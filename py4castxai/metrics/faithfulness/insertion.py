import os
from typing import Dict, List, Optional
from dataclasses import dataclass

import numpy as np
import torch

import numpy as np
import torch
import torch.nn.functional as F
import os
from typing import Dict, List, Optional, Tuple, NamedTuple
from scipy.ndimage import gaussian_filter

from py4castxai.explainers import explainers_registry
from py4castxai.utils_test.infer_utils import predict_step

# ---------------------------------------------------------------------------
# Result containers (mirror ROADEventResult / ROADAggregateResult)
# ---------------------------------------------------------------------------

@dataclass
class InsDelEventResult:
    runtime: str
    ins_auc: float          # insertion AUC   (higher = better)
    del_auc: float          # deletion AUC    (lower  = better)
    score: float            # ins_auc - del_auc  (higher = better)
    ins_curve: np.ndarray   # ROI prediction vs fraction inserted
    del_curve: np.ndarray   # ROI prediction vs fraction deleted
    fractions: np.ndarray
    pred_clean: float
    precip_mean: float


@dataclass
class InsDelAggregateResult:
    mean_ins: float
    mean_del: float
    mean_score: float
    sem_score: float
    fractions: np.ndarray
    ins_curve_mean: np.ndarray
    del_curve_mean: np.ndarray
    ins_curve_std: np.ndarray
    del_curve_std: np.ndarray
    n_events: int
    event_results: List[InsDelEventResult]

def compute_roi_indices(
    target: List[float],
    extent: List[float],
    H: int,
    W: int
) -> Tuple[int, int, int, int]:
    """Geographic bounding box → pixel indices."""
    lon0, lon1 = target[0], target[1]
    lat0, lat1 = target[2], target[3]
    lat_min, lat_max = extent[2], extent[3]
    lon_min, lon_max = extent[0], extent[1]

    lat_step = (lat_max - lat_min) / H
    lon_step = (lon_max - lon_min) / W

    i_min = int((lat0 - lat_min) / lat_step)
    i_max = int((lat1 - lat_min) / lat_step)
    j_min = int((lon0 - lon_min) / lon_step)
    j_max = int((lon1 - lon_min) / lon_step)

    return i_min, i_max, j_min, j_max


def get_roi_mean(
    tensor: torch.Tensor,
    i_min: int, i_max: int,
    j_min: int, j_max: int,
    t_idx: int,
    feat_idx: int
) -> float:
    """Mean model output over Paris ROI."""
    
    return tensor[
        :, t_idx, i_min:i_max, j_min:j_max, feat_idx
    ].mean().item()
def normalize_expl(a: torch.Tensor, abs_val: bool = True) -> torch.Tensor:
    """Normalise attribution map to [0, 1]."""
    if isinstance(a, np.ndarray):
        a = torch.from_numpy(a).to(torch.float32)
    if abs_val:
        return a.abs() / (a.abs().max() + 1e-8)
    return a / (a.abs().max() + 1e-8)


def spatial_blur(x: torch.Tensor, sigma: float = 10.0) -> torch.Tensor:
    """
    Gaussian blur over spatial field — PIC lower bound baseline.
    Args:
        x:     [B, H, W, 1]
        sigma: blur radius in pixels (~25km at 2.5km resolution for sigma=10)
    """
    x_np = x.detach().cpu().squeeze().numpy()
    blurred_np = gaussian_filter(x_np, sigma=sigma)
    return torch.from_numpy(blurred_np).to(x.dtype).to(x.device).reshape(x.shape)

class InsertionDeletion:
    """
    Insertion / Deletion AUC for meteorological regression, ROI-scalar readout.

    WHY THIS METRIC (vs the binarized ROAD)
    ---------------------------------------
    ROAD masks a TOP-K set, so it uses only the RANK of attribution values and
    the score inherits a spatial-contiguity / imputation bias that rewards
    smooth maps independently of faithfulness. Insertion/Deletion instead
    reveals pixels IN ORDER OF ATTRIBUTION MAGNITUDE, one fraction at a time,
    over a fixed baseline. No fixed binary mask boundary, so:
      - magnitude ordering is used at every step (not just a single cut)
      - the contiguity/imputation artifact does not enter the same way
      - a smoother map has no mechanical advantage unless it is actually
        ordering pixels better

    DELETION: start from the clean field, progressively REPLACE the most
    salient pixels with a baseline. If attribution is faithful, the ROI
    prediction should drop FAST -> low deletion AUC is good.

    INSERTION: start from an all-baseline field, progressively RESTORE the most
    salient pixels. Faithful attribution -> ROI prediction recovers FAST ->
    high insertion AUC is good.

    Combined score = ins_auc - del_auc (higher = better).

    BASELINE
    --------
    The baseline field is what a "removed" pixel becomes. Options:
      "blur"  : heavy Gaussian blur of the clean input (destroys spatial
                structure, preserves local mean; the recommended default and
                the analogue of ROAD's PIC lower bound)
      "mean"  : spatial mean of the input channel
      "zero"  : normalized zero (physical zero after de-normalization differs;
                usually a poor baseline for physical fields)

    Curves are read out as the ROI-mean prediction, identical to ROAD, so the
    two metrics are directly comparable on the same events.
    """

    def __init__(
        self,
        model: torch.nn.Module,
        explainer: str = "BaseGrad",
        precip_threshold: float = 5.0,
        baseline: str = "blur",
        blur_sigma: float = 10.0,
        n_steps: int = 20,
        seed: int = 42,
        **expl_params,
    ):
        self.model = model
        self.precip_threshold = precip_threshold
        self.baseline = baseline
        self.blur_sigma = blur_sigma
        self.n_steps = n_steps
        self.seed = seed
        # fractions of pixels revealed/removed: 0 .. 1 inclusive
        self.fractions = np.linspace(0.0, 1.0, n_steps + 1)

        self.explainer_name = explainer
        self.explainer = self._init_explainer(explainer, expl_params)

    # ------------------------------------------------------------------
    # Helpers (same shapes/conventions as the ROAD class)
    # ------------------------------------------------------------------

    def _init_explainer(self, name: str, params: Dict):
        cls = explainers_registry.get(name)
        if cls is None:
            raise ValueError(
                f"Unknown explainer: {name}. Available: {list(explainers_registry.keys())}"
            )
        return cls(self.model, **params)

    def _get_runtime(self, batch_idx: int, infer_ds) -> str:
        return infer_ds.sample_list[batch_idx].timestamps.datetime.strftime("%Y%m%d%H")

    def _move_batch_to_device(self, batch, device):
        batch.inputs.tensor = batch.inputs.tensor.to(device)
        batch.forcing.tensor = batch.forcing.tensor.to(device)
        if batch.outputs is not None:
            batch.outputs.tensor = batch.outputs.tensor.to(device)
        return batch

    def _make_baseline(self, x: torch.Tensor) -> torch.Tensor:
        """
        x : (B, H, W, 1) single-channel input field (clean).
        Returns a baseline field of the same shape.
        """
        if self.baseline == "blur":
            return spatial_blur(x, sigma=self.blur_sigma).to(x.device)
        if self.baseline == "mean":
            return torch.full_like(x, float(x.mean()))
        if self.baseline == "zero":
            return torch.zeros_like(x)
        raise ValueError(f"unknown baseline {self.baseline}")

    def _predict_roi(
        self, batch, input_clean, input_idx, output_idx,
        i_min, i_max, j_min, j_max, t_step,
        batch_idx, checkpoint, cfg_model, cfg_dataset, cfg_xai,
        dataset_info, infer_ds, list_run_hour, use_old_weights,
        forcing_input, field_2d,
    ) -> float:
        """
        Overwrite the explained channel with field_2d, run the model, read the
        ROI-mean prediction, restore the clean tensor. field_2d is (B, H, W).
        """
        if forcing_input:
            new = batch.forcing.tensor.clone()
            new[..., input_idx] = field_2d.unsqueeze(1)  # (B,1,H,W) time dim
            batch.forcing.tensor = new
        else:
            new = batch.inputs.tensor.clone()
            new[..., input_idx] = field_2d.unsqueeze(1)
            batch.inputs.tensor = new

        with torch.no_grad():
            pred = predict_step(
                self.model, batch, batch_idx, checkpoint,
                cfg_model, cfg_dataset, cfg_xai, dataset_info,
                infer_ds, list_run_hour, use_old_weights,
            )

        if forcing_input:
            batch.forcing.tensor = input_clean
        else:
            batch.inputs.tensor = input_clean

        return get_roi_mean(
            pred.tensor, i_min, i_max, j_min, j_max, t_idx=t_step, feat_idx=output_idx
        )

    # ------------------------------------------------------------------
    # Main evaluation
    # ------------------------------------------------------------------

    def evaluate(
        self,
        dataloader,
        checkpoint,
        cfg_model,
        cfg_dataset,
        cfg_xai,
        dataset_info,
        infer_ds,
        list_run_hour,
        use_old_weights,
        target,
        plot_explanations,
        extent,
        fig_path,
        precip_thresh=None,
    ) -> InsDelAggregateResult:

        torch.manual_seed(self.seed)
        np.random.seed(self.seed)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        os.makedirs(cfg_xai["eval"]["eval_plot"]["fig_path"] + "/insdel", exist_ok=True)

        if precip_thresh:
            self.precip_threshold = precip_thresh
            print("Precipitation threshold override:", self.precip_threshold)

        forcing_input = cfg_xai["explain"]["forcing_input"]
        n_pred_steps = int(cfg_dataset["num_pred_steps_val_test"])
        t_step = n_pred_steps - 1

        results: List[InsDelEventResult] = []

        with torch.autograd.set_grad_enabled(True):
            for batch_idx, batch in enumerate(dataloader):

                H, W = batch.inputs.tensor.shape[2:4]
                i_min, i_max, j_min, j_max = compute_roi_indices(target, extent, H, W)
                output_idx = batch.outputs.feature_names_to_idx[cfg_xai["explain"]["output"]]

                precip_mean = get_roi_mean(
                    batch.outputs.tensor, i_min, i_max, j_min, j_max,
                    t_idx=0, feat_idx=output_idx,
                )
                if precip_mean <= self.precip_threshold:
                    continue

                batch = self._move_batch_to_device(batch, device)
                batch.inputs.tensor.requires_grad_()
                runtime = self._get_runtime(batch_idx, infer_ds)

                explain_clean, output_clean = self.explainer.compute_explanations(
                    batch, batch_idx, checkpoint,
                    cfg_model, cfg_dataset, cfg_xai, dataset_info,
                    infer_ds, list_run_hour, use_old_weights, target,
                    plot_explanations=plot_explanations, extent=extent,
                    fig_path=fig_path, return_output=True,
                )
                if explain_clean is None:
                    continue

                if forcing_input:
                    input_clean = batch.forcing.tensor.clone()
                    input_idx = batch.forcing.feature_names_to_idx[cfg_xai["explain"]["input"]]
                else:
                    input_clean = batch.inputs.tensor.clone()
                    input_idx = batch.inputs.feature_names_to_idx[cfg_xai["explain"]["input"]]

                # clean single-channel field, shape (B, H, W, 1)
                x_clean = (
                    batch.forcing.tensor[:, 0, ..., input_idx].unsqueeze(-1)
                    if forcing_input
                    else batch.inputs.tensor[:, 0, ..., input_idx].unsqueeze(-1)
                )
                base_field = self._make_baseline(x_clean)          # (B,H,W,1)

                x_clean_2d = x_clean.squeeze(-1)                    # (B,H,W)
                base_2d = base_field.squeeze(-1)                    # (B,H,W)

                pred_clean_roi = get_roi_mean(
                    output_clean.tensor, i_min, i_max, j_min, j_max,
                    t_idx=t_step, feat_idx=output_idx,
                )

                # --- attribution -> descending pixel order ---
                explain_t = explain_clean[0]
                explain_t = normalize_expl(
                    explain_t 
                )
                order = torch.argsort(explain_t.reshape(-1), descending=True)  # (H*W,)

                # --- insertion & deletion curves ---
                ins_curve, del_curve = [], []
                total = H * W

                for frac in self.fractions:
                    k = int(round(frac * total))

                    # DELETION: clean field, top-k salient pixels -> baseline
                    del_field = x_clean_2d.clone().reshape(1, -1)
                    if k > 0:
                        sel = order[:k]
                        del_field[0, sel] = base_2d.reshape(1, -1)[0, sel]
                    del_field = del_field.reshape(1, H, W)
                    del_curve.append(self._predict_roi(
                        batch, input_clean, input_idx, output_idx,
                        i_min, i_max, j_min, j_max, t_step,
                        batch_idx, checkpoint, cfg_model, cfg_dataset, cfg_xai,
                        dataset_info, infer_ds, list_run_hour, use_old_weights,
                        forcing_input, del_field,
                    ))

                    # INSERTION: baseline field, top-k salient pixels -> clean
                    ins_field = base_2d.clone().reshape(1, -1)
                    if k > 0:
                        sel = order[:k]
                        ins_field[0, sel] = x_clean_2d.reshape(1, -1)[0, sel]
                    ins_field = ins_field.reshape(1, H, W)
                    ins_curve.append(self._predict_roi(
                        batch, input_clean, input_idx, output_idx,
                        i_min, i_max, j_min, j_max, t_step,
                        batch_idx, checkpoint, cfg_model, cfg_dataset, cfg_xai,
                        dataset_info, infer_ds, list_run_hour, use_old_weights,
                        forcing_input, ins_field,
                    ))

                ins_curve = np.array(ins_curve, dtype=float)
                del_curve = np.array(del_curve, dtype=float)

                # Normalize curves to [clean, baseline] span so AUCs are
                # comparable across events with different prediction scales.
                pred_base_roi = del_curve[-1]  # fully baselined == deletion end
                span = (pred_clean_roi - pred_base_roi)
                if abs(span) < 1e-8:
                    continue  # ROI prediction insensitive to this channel; skip
                ins_norm = (ins_curve - pred_base_roi) / span
                del_norm = (del_curve - pred_base_roi) / span

                ins_auc = float(np.trapezoid(ins_norm, self.fractions))
                del_auc = float(np.trapezoid(del_norm, self.fractions))

                results.append(InsDelEventResult(
                    runtime=runtime,
                    ins_auc=ins_auc,
                    del_auc=del_auc,
                    score=ins_auc - del_auc,
                    ins_curve=ins_norm,
                    del_curve=del_norm,
                    fractions=self.fractions.copy(),
                    pred_clean=pred_clean_roi,
                    precip_mean=precip_mean,
                ))

                print(
                    f"[InsDel] {runtime} | precip={precip_mean:.3f} | "
                    f"ins={ins_auc:.4f} del={del_auc:.4f} score={ins_auc-del_auc:.4f}"
                )

        return self._aggregate(results)

    # ------------------------------------------------------------------
    # Aggregation (mirrors ROAD._aggregate)
    # ------------------------------------------------------------------

    def _aggregate(self, event_results: List[InsDelEventResult]) -> InsDelAggregateResult:
        if not event_results:
            raise ValueError("No valid events. Check precip_threshold / channel sensitivity.")

        ins = np.array([r.ins_auc for r in event_results])
        dels = np.array([r.del_auc for r in event_results])
        score = np.array([r.score for r in event_results])
        ins_curves = np.stack([r.ins_curve for r in event_results], axis=0)
        del_curves = np.stack([r.del_curve for r in event_results], axis=0)
        n = len(score)

        print(f"\n[InsDel] Aggregated over {n} valid events")
        print(f"  Insertion AUC : {ins.mean():.4f}  (higher better)")
        print(f"  Deletion  AUC : {dels.mean():.4f}  (lower  better)")
        print(f"  Score (I - D) : {score.mean():.4f} +/- {score.std()/np.sqrt(n):.4f}")

        return InsDelAggregateResult(
            mean_ins=float(ins.mean()),
            mean_del=float(dels.mean()),
            mean_score=float(score.mean()),
            sem_score=float(score.std() / np.sqrt(n)),
            fractions=event_results[0].fractions,
            ins_curve_mean=ins_curves.mean(axis=0),
            del_curve_mean=del_curves.mean(axis=0),
            ins_curve_std=ins_curves.std(axis=0),
            del_curve_std=del_curves.std(axis=0),
            n_events=n,
            event_results=event_results,
        )