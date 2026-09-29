import torch

from ..inference import predict_step
from .base import Explainer


class SmoothGrad(Explainer):
    """SmoothGrad (Smilkov et al., 2017): pointwise mean of the perturbed gradients."""

    def __init__(self, model, std_perturbations=0.2, num_perturbations=20):
        super().__init__(model)
        self.std_perturbations = std_perturbations
        self.num_perturbations = num_perturbations

    def explain(self, batch, ctx):
        prediction, _ = predict_step(self.model, batch, ctx)
        gradients = self.perturbed_gradients(
            batch, ctx, self.std_perturbations, self.num_perturbations
        )
        return torch.stack(gradients).mean(dim=0), prediction
