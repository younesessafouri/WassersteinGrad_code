from abc import ABC, abstractmethod

import torch

from ..inference import predict_step


class Explainer(ABC):
    """Base class of the gradient-based attribution methods.

    ``explain(batch, ctx)`` returns ``(attribution, prediction)``: an (H, W) map
    explaining the ``ctx.output_name`` field summed over the ROI at the last
    lead time with respect to the ``ctx.input_name`` channel of the initial
    state, and the de-normalised forecast. Batches hold one sample with one
    input time step.
    """

    def __init__(self, model):
        self.model = model

    @abstractmethod
    def explain(self, batch, ctx): ...

    def gradient(self, batch, ctx):
        """``(gradient, prediction)``: gradient of the ROI target w.r.t. the explained channel."""
        prediction, gradient = predict_step(self.model, batch, ctx, compute_grads=True)
        input_idx = batch.inputs.feature_names_to_idx[ctx.input_name]
        return gradient[0, 0, ..., input_idx], prediction

    def perturbed_gradients(self, batch, ctx, std, num):
        """Gradients at ``num`` noisy copies of the input (Algorithm 1, lines 2-7).

        Only the explained channel is perturbed, with Gaussian noise of standard
        deviation ``std * (max - min)`` of that channel.
        """
        clean = batch.inputs.tensor
        input_idx = batch.inputs.feature_names_to_idx[ctx.input_name]
        x_in = clean[..., input_idx].detach()
        sigma = std * (x_in.max() - x_in.min())
        gradients = []
        for _ in range(num):
            noisy = clean.detach().clone()
            noisy[..., input_idx] = x_in + torch.randn_like(x_in) * sigma
            batch.inputs.tensor = noisy
            gradients.append(self.gradient(batch, ctx)[0])
        batch.inputs.tensor = clean
        return gradients
