import logging

import numpy as np
import torch
import torch.nn.functional as F
from scipy.integrate import trapezoid

from ..inference import predict_step, roi_mean
from .base import Metric, normalize_attribution

logger = logging.getLogger(__name__)


def linear_imputation(x, mask, noise_std, seed):
    """Replace masked pixels by their 3x3 neighbourhood mean plus Gaussian noise.

    ``x`` and ``mask`` have shape (B, H, W, 1); ``mask`` is 1 on the removed pixels.
    """
    torch.manual_seed(seed)
    kernel = torch.ones(1, 1, 3, 3, device=x.device) / 9.0
    neighbor_mean = F.conv2d(x.permute(0, 3, 1, 2), kernel, padding=1).permute(0, 2, 3, 1)
    noise = noise_std * torch.randn_like(x)
    return x * (1 - mask) + (neighbor_mean + noise) * mask


class ROAD(Metric):
    """Faithfulness: binary ROAD for regression (Rong et al., 2022; Eqs. 20-21).

    For each p in ``percentages`` the top-p % pixels of the attribution map are
    removed from the explained input channel by linear imputation (noise std
    ``std_noise * (max - min)``). The step scores 1 if the ROI forecast at the
    last lead time changes more than on average over ``n_random`` random masks of
    the same size. The score is the normalised area under this binary curve.
    """

    def __init__(
        self, model, explainer, seed=42, std_noise=0.1, n_random=5, percentages=tuple(range(1, 16))
    ):
        super().__init__(model, explainer, seed)
        self.std_noise = std_noise
        self.n_random = n_random
        self.percentages = list(percentages)

    def evaluate(self, events, ctx):
        torch.manual_seed(self.seed)
        np.random.seed(self.seed)
        scores = []
        for batch_idx, runtime, batch in events:
            scores.append(self._event_score(batch_idx, batch, ctx))
            logger.info("[ROAD] %s | AUC=%.4f", runtime, scores[-1])
        return {"ROAD": scores}

    def _event_score(self, batch_idx, batch, ctx):
        attribution, prediction = self.explainer.explain(batch, ctx)
        H, W = batch.inputs.tensor.shape[2:4]
        roi = ctx.roi(H, W)
        input_idx = batch.inputs.feature_names_to_idx[ctx.input_name]
        output_idx = batch.outputs.feature_names_to_idx[ctx.output_name]

        clean_inputs = batch.inputs.tensor.clone()
        x_in = batch.inputs.tensor[..., input_idx]
        noise_std = self.std_noise * (x_in.max() - x_in.min()).item()
        x = batch.inputs.tensor[:, 0, ..., input_idx].unsqueeze(-1)  # (1, H, W, 1)
        clean_forecast = roi_mean(prediction.tensor, roi, -1, output_idx)
        saliency = normalize_attribution(attribution).reshape(1, -1)

        def forecast_change(pixels, seed):
            """Change of the ROI forecast when ``pixels`` (flat indices) are removed."""
            mask = torch.zeros(1, H * W, device=x.device)
            mask.scatter_(1, pixels.to(x.device), 1.0)
            x_masked = linear_imputation(x, mask.view(1, H, W, 1), noise_std, seed)
            inputs = clean_inputs.clone()
            inputs[..., input_idx] = x_masked.unsqueeze(1).squeeze(-1)
            batch.inputs.tensor = inputs
            masked_prediction, _ = predict_step(self.model, batch, ctx)
            batch.inputs.tensor = clean_inputs
            return abs(clean_forecast - roi_mean(masked_prediction.tensor, roi, -1, output_idx))

        event_seed = self.seed + batch_idx * 1000
        curve = []
        for p_idx, p in enumerate(self.percentages):
            top_k = int(p / 100 * H * W)
            if top_k == 0:
                curve.append(0)
                continue
            _, salient = torch.topk(saliency, top_k, dim=1)
            salient_change = forecast_change(salient, event_seed + p_idx)
            random_changes = []
            for r in range(self.n_random):
                random_seed = event_seed + p_idx * 100 + r
                torch.manual_seed(random_seed)
                random_pixels = torch.randperm(H * W)[:top_k].unsqueeze(0)
                random_changes.append(forecast_change(random_pixels, random_seed + 999))
            curve.append(1 if salient_change > np.mean(random_changes) else 0)

        x_axis = np.array(self.percentages, dtype=float)
        auc = trapezoid(np.array(curve, dtype=float), x_axis) / (x_axis.max() - x_axis.min())
        return float(auc)
