"""Autoregressive inference and ROI gradients for Py4Cast models.

Pure-PyTorch port of the inference loop of ``AutoRegressiveLightning`` in
Py4Cast (https://github.com/meteofrance/py4cast), extended to return the
gradient of the forecast over a region of interest (ROI) with respect to the
initial state. The batches are Py4Cast ``ItemBatch`` objects.

py4cast and mfai are imported inside the functions that need them, so that the
explainers (e.g. ``wasserstein_barycenter``) can be imported without Py4Cast.
"""

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

import torch
from torch.utils.checkpoint import checkpoint


@dataclass
class ForecastContext:
    """Everything ``predict_step`` needs besides the model and the batch."""

    cfg_model: dict  # ``model`` section of the Py4Cast model config
    dataset_info: Any  # py4cast DatasetInfo: stats, diff_stats, statics
    output_feature_names: list  # from the Py4Cast checkpoint
    output_dim_names: list  # from the Py4Cast checkpoint
    output_dtype: Any  # from the Py4Cast checkpoint
    input_name: str  # explained input channel, e.g. "aro_u_250hpa"
    output_name: str  # explained output channel, e.g. "aro_tp_0m"
    target: Sequence[float] | None = None  # ROI (lon_min, lon_max, lat_min, lat_max)
    extent: Sequence[float] | None = None  # grid (lon_min, lon_max, lat_min, lat_max)

    def roi(self, H: int, W: int):
        """Pixel box (i_min, i_max, j_min, j_max) of the ROI, or the whole grid."""
        if not self.target:
            return 0, H, 0, W
        return roi_indices(self.target, self.extent, H, W)


def roi_indices(target, extent, H, W):
    """Pixel box (i_min, i_max, j_min, j_max) of a lon/lat box on a regular H x W grid."""
    lon0, lon1, lat0, lat1 = target
    lon_min, lon_max, lat_min, lat_max = extent
    lat_step = (lat_max - lat_min) / H
    lon_step = (lon_max - lon_min) / W
    return (
        int((lat0 - lat_min) / lat_step),
        int((lat1 - lat_min) / lat_step),
        int((lon0 - lon_min) / lon_step),
        int((lon1 - lon_min) / lon_step),
    )


def roi_mean(tensor, roi, t_idx, feat_idx):
    """Mean of a (B, T, H, W, F) tensor over the ROI for one time step and feature."""
    i_min, i_max, j_min, j_max = roi
    return tensor[:, t_idx, i_min:i_max, j_min:j_max, feat_idx].mean().item()


def event_intensity(batch, ctx):
    """Mean ground truth of the explained output over the ROI at the first lead time.

    Used to select events (e.g. heavy precipitation over the ROI).
    """
    roi = ctx.roi(*batch.inputs.tensor.shape[2:4])
    output_idx = batch.outputs.feature_names_to_idx[ctx.output_name]
    return roi_mean(batch.outputs.tensor, roi, 0, output_idx)


def batch_to_device(batch, device):
    batch.inputs.tensor = batch.inputs.tensor.to(device)
    batch.forcing.tensor = batch.forcing.tensor.to(device)
    if batch.outputs is not None:
        batch.outputs.tensor = batch.outputs.tensor.to(device)
    return batch


def predict_step(model, batch, ctx, compute_grads=False, enable_checkpoint=True):
    """Roll the model out over ``batch.num_pred_steps`` autoregressive steps.

    Returns ``(prediction, gradient)``. ``prediction`` is the de-normalised
    forecast (NamedTensor with all lead times). If ``compute_grads`` is set,
    ``gradient`` is the gradient of the ``ctx.output_name`` field at the last
    lead time, summed over the ROI, with respect to ``batch.inputs.tensor``
    (same shape as the inputs). Otherwise it is None.
    """
    from py4cast.datasets.base import NamedTensor

    gradient = None
    with torch.set_grad_enabled(compute_grads):
        if compute_grads:
            batch.inputs.tensor.requires_grad_(True)
        states = _rollout(model, batch, ctx, enable_checkpoint)
        if compute_grads:
            i_min, i_max, j_min, j_max = ctx.roi(*batch.inputs.tensor.shape[2:4])
            output_idx = batch.outputs.feature_names_to_idx[ctx.output_name]
            roi_field = states[-1][..., i_min:i_max, j_min:j_max, output_idx]
            gradient = torch.autograd.grad(roi_field.sum(), batch.inputs.tensor)[0]

    prediction = NamedTensor(
        torch.stack(states, dim=1).detach().type(ctx.output_dtype),
        ctx.output_dim_names,
        ctx.output_feature_names,
    )
    stats = ctx.dataset_info.stats
    for name in prediction.feature_names:
        idx = prediction.feature_names_to_idx[name]
        prediction.tensor[..., idx] *= torch.asarray(stats[name]["std"])
        prediction.tensor[..., idx] += torch.asarray(stats[name]["mean"])
    return prediction, gradient


def _rollout(model, batch, ctx, enable_checkpoint):
    """Normalised predicted states (B, H, W, F), one per lead time.

    Supports the ``scaled_ar`` (state + y * diff_std + diff_mean) and ``diff_ar``
    (state + y) strategies of Py4Cast. Boundary forcing is not used at inference.
    """
    from mfai.pytorch.models.utils import (
        expand_to_batch,
        features_last_to_second,
        features_second_to_last,
    )
    from py4cast.datasets.base import NamedTensor

    cfg = ctx.cfg_model
    strategy = cfg["training_strategy"]
    num_inter_steps = cfg["num_inter_steps"]
    if strategy not in ("scaled_ar", "diff_ar") or cfg["mask_ratio"] != 0:
        raise NotImplementedError(f"Unsupported model: {strategy}, mask_ratio={cfg['mask_ratio']}.")
    if strategy == "diff_ar" and num_inter_steps != 1:
        raise ValueError("Diff AR strategy requires exactly 1 intermediary step.")
    scale_y = strategy == "scaled_ar"

    device = batch.inputs.tensor.device
    if scale_y:
        diff_stats = ctx.dataset_info.diff_stats
        names = ctx.output_feature_names
        diff_std = diff_stats.to_list("std", names).to(device, non_blocking=True)
        diff_mean = diff_stats.to_list("mean", names).to(device, non_blocking=True)
    grid_statics = expand_to_batch(
        ctx.dataset_info.statics.grid_statics.tensor, batch.batch_size
    ).to(device)

    def forward(x):
        if model.features_second:
            return features_second_to_last(model(features_last_to_second(x)))
        return model(x)

    prev_states = batch.inputs
    states = []
    for i in range(batch.num_pred_steps):
        for k in range(num_inter_steps):
            # Model input: previous states, static fields and forcings of step i.
            forcing = batch.forcing.select_dim("timestep", i)
            inputs = [
                prev_states.select_tensor_dim("timestep", t) for t in range(batch.num_input_steps)
            ]
            x = torch.cat(
                inputs + [grid_statics[: batch.batch_size], forcing.tensor],
                dim=forcing.dim_index("features"),
            )
            if cfg["channels_last"]:
                x = x.to(memory_format=torch.channels_last)
            y = checkpoint(forward, x, use_reentrant=False) if enable_checkpoint else forward(x)

            last_state = prev_states.select_tensor_dim("timestep", -1)
            new_state = last_state + y * diff_std + diff_mean if scale_y else last_state + y

            # Slide the input window, except after the very last step.
            if i < batch.num_pred_steps - 1 or k < num_inter_steps - 1:
                t_dim = batch.inputs.dim_index("timestep")
                kept = prev_states.index_select_tensor_dim(
                    "timestep", range(1, prev_states.dim_size("timestep"))
                )
                prev_states = NamedTensor.new_like(
                    torch.cat([kept, new_state.unsqueeze(t_dim)], dim=t_dim), prev_states
                )
        states.append(new_state)
    return states
