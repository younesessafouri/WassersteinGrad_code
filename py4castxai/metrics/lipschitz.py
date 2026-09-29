import logging

import numpy as np
import torch

from .base import Metric, normalize_attribution

logger = logging.getLogger(__name__)


def l2_distance(a, b):
    return torch.norm(a.reshape(-1).float() - b.reshape(-1).float()).item()


def cosine_distance(a, b):
    """l2 distance between the two maps projected on the unit sphere (scale invariant)."""
    a = a.reshape(-1).float()
    b = b.reshape(-1).float()
    return torch.norm(a / (torch.norm(a) + 1e-8) - b / (torch.norm(b) + 1e-8)).item()


class LocalLipschitzEstimate(Metric):
    """Robustness: local Lipschitz estimate (Alvarez-Melis & Jaakkola, 2018; Eqs. 22-23).

    max over ``num_iter`` noisy inputs of ||G(x) - G(x + eps)|| / ||eps||, where eps
    perturbs the explained channel with std ``std_noise * (max - min)``. Maps are
    normalised by their maximum (LLE_l2) and also by their l2 norm (LLE_cos). The
    explainer's own noise is re-seeded identically for the clean and noisy inputs.
    """

    def __init__(self, model, explainer, seed=42, std_noise=0.1, num_iter=7):
        super().__init__(model, explainer, seed)
        self.std_noise = std_noise
        self.num_iter = num_iter

    def evaluate(self, events, ctx):
        torch.manual_seed(self.seed)
        np.random.seed(self.seed)
        scores_l2, scores_cos = [], []
        for batch_idx, runtime, batch in events:
            input_idx = batch.inputs.feature_names_to_idx[ctx.input_name]
            x_in = batch.inputs.tensor[..., input_idx].clone()
            noise_std = self.std_noise * (x_in.max() - x_in.min()).item()
            explainer_seed = self.seed + batch_idx * 10000

            torch.manual_seed(explainer_seed)
            attribution, _ = self.explainer.explain(batch, ctx)
            clean_map = normalize_attribution(attribution)
            clean_inputs = batch.inputs.tensor.clone().detach()

            ratios_l2, ratios_cos = [], []
            for i in range(self.num_iter):
                torch.manual_seed(self.seed + batch_idx * 1000 + i)
                delta = torch.randn_like(x_in) * noise_std
                noisy = clean_inputs.clone()
                noisy[..., input_idx] = (x_in + delta).detach()
                batch.inputs.tensor = noisy
                torch.manual_seed(explainer_seed)
                noisy_map = normalize_attribution(self.explainer.explain(batch, ctx)[0])
                input_distance = torch.norm(delta).item() + 1e-8
                ratios_l2.append(l2_distance(clean_map, noisy_map) / input_distance)
                ratios_cos.append(cosine_distance(clean_map, noisy_map) / input_distance)
            batch.inputs.tensor = clean_inputs

            scores_l2.append(max(ratios_l2))
            scores_cos.append(max(ratios_cos))
            logger.info("[LLE] %s | l2=%.5f | cos=%.7f", runtime, scores_l2[-1], scores_cos[-1])
        return {"LLE_l2": scores_l2, "LLE_cos": scores_cos}
