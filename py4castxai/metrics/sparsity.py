import logging

import numpy as np
import torch

from .base import Metric, normalize_attribution

logger = logging.getLogger(__name__)


def gini(x):
    """Gini index of each row of a non-negative (B, N) array (Eq. 24)."""
    x = np.sort(x + 1e-7, axis=-1)
    n = x.shape[-1]
    k = np.arange(1, n + 1)[np.newaxis, :]
    return ((2 * k - n - 1) * x).sum(axis=-1) / (n * x.sum(axis=-1))


class Sparsity(Metric):
    """Complexity: Gini index of |attribution| (Chalasani et al., 2020). Higher is sparser."""

    def evaluate(self, events, ctx):
        torch.manual_seed(self.seed)
        np.random.seed(self.seed)
        scores = []
        for _, runtime, batch in events:
            attribution, _ = self.explainer.explain(batch, ctx)
            flat = normalize_attribution(attribution).reshape(1, -1)
            scores.append(float(gini(flat.detach().cpu().numpy())[0]))
            logger.info("[Sparsity] %s | gini=%.4f", runtime, scores[-1])
        return {"Sparsity": scores}
