import torch

from ..inference import predict_step
from .base import Explainer


class IntegratedGrad(Explainer):
    """Integrated Gradients (Sundararajan et al., 2017) with a zero baseline.

    On standardised inputs the zero baseline is the climatological mean state.
    The path integral is approximated by a left Riemann sum of ``num_steps`` terms.
    """

    def __init__(self, model, num_steps=20):
        super().__init__(model)
        self.num_steps = num_steps

    def explain(self, batch, ctx):
        prediction, _ = predict_step(self.model, batch, ctx)
        clean = batch.inputs.tensor
        input_idx = batch.inputs.feature_names_to_idx[ctx.input_name]
        x_in = clean[..., input_idx].detach()
        baseline = torch.zeros_like(x_in)

        gradients = []
        for step in range(self.num_steps):
            alpha = step / self.num_steps
            path_point = clean.detach().clone()
            path_point[..., input_idx] = baseline + alpha * (x_in - baseline)
            batch.inputs.tensor = path_point
            gradients.append(self.gradient(batch, ctx)[0])
        batch.inputs.tensor = clean

        attribution = (x_in - baseline)[0, 0] * torch.stack(gradients).mean(dim=0)
        return attribution, prediction
