"""WassersteinGrad (Section 3.3 and Algorithm 1 of the paper).

The N gradient maps obtained from noisy inputs are turned into probability
measures |G_i| / sum|G_i| and aggregated by their entropic Wasserstein
barycenter instead of their pointwise mean (SmoothGrad).

* ``WassersteinGrad``: the barycenter itself (WGBary in the paper).
* ``WassersteinGradMask``: the barycenter used as a spatial mask on the clean
  gradient, which restores the sign (WGBary x Grad in the paper).
"""

import ot
import torch

from ..inference import predict_step
from .base import Explainer


def wasserstein_barycenter(gradients, reg=1e-3):
    """Entropic Wasserstein barycenter of 2D attribution maps.

    Args:
        gradients: N maps of shape (H, W) (a list or an (N, H, W) tensor).
        reg: entropic regularisation (lambda in the paper).

    Returns:
        The (H, W) barycenter of the measures |G_i| / sum|G_i|, computed with the
        convolutional Sinkhorn algorithm of POT (Solomon et al., 2015).
    """
    measures = torch.stack([g.abs() / (g.abs().sum() + 1e-8) for g in gradients])
    return ot.bregman.convolutional_barycenter2d(measures, reg=reg)


class WassersteinGrad(Explainer):
    """WGBary: Wasserstein barycenter of the perturbed gradients."""

    def __init__(self, model, std_perturbations=0.2, num_perturbations=20, reg=1e-3):
        super().__init__(model)
        self.std_perturbations = std_perturbations
        self.num_perturbations = num_perturbations
        self.reg = reg

    def explain(self, batch, ctx):
        prediction, _ = predict_step(self.model, batch, ctx)
        gradients = self.perturbed_gradients(
            batch, ctx, self.std_perturbations, self.num_perturbations
        )
        return wasserstein_barycenter(gradients, self.reg), prediction


class WassersteinGradMask(WassersteinGrad):
    """WGBary x Grad: clean gradient masked by the Wasserstein barycenter."""

    def explain(self, batch, ctx):
        prediction, _ = predict_step(self.model, batch, ctx)
        clean_gradient, _ = self.gradient(batch, ctx)
        gradients = self.perturbed_gradients(
            batch, ctx, self.std_perturbations, self.num_perturbations
        )
        return clean_gradient * wasserstein_barycenter(gradients, self.reg), prediction
