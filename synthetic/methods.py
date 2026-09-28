"""BaseGrad, SmoothGrad and WassersteinGrad for any differentiable map f: (B, H, W) -> (B,).

All three methods take the model ``f`` and one input field ``x`` of shape (H, W)
and return an (H, W) attribution map. SmoothGrad and WassersteinGrad aggregate
the same perturbed gradients (``noisy_gradients``) in two different ways:
pointwise averaging, or an entropic Wasserstein barycenter of |G_i| / sum|G_i|.
"""

import ot
import torch


def input_gradients(f, x):
    """Gradients of f w.r.t. each field of a batch x of shape (B, H, W)."""
    x = x.detach().requires_grad_(True)
    return torch.autograd.grad(f(x).sum(), x)[0]


def noisy_gradients(f, x, noise_level, n_samples, generator=None):
    """Gradients at ``n_samples`` copies of x with Gaussian noise of std noise_level * (max - min)."""
    sigma = noise_level * (x.max() - x.min())
    noise = torch.randn((n_samples, *x.shape), generator=generator, dtype=x.dtype)
    return input_gradients(f, x + sigma * noise)


def wasserstein_barycenter(gradients, reg=1e-3):
    """Entropic Wasserstein barycenter of the measures |G_i| / sum|G_i|, gradients of shape (N, H, W).

    Computed with the convolutional Sinkhorn algorithm of POT, as in the paper. ``reg`` is
    defined on the grid rescaled to [0, 1]^2 (Gaussian kernel of std sqrt(reg / 2) * (n - 1) pixels).
    """
    measures = torch.stack([g.abs() / g.abs().sum() for g in gradients])
    return ot.bregman.convolutional_barycenter2d(measures, reg=reg)


def base_grad(f, x):
    return input_gradients(f, x[None])[0]


def smooth_grad(f, x, noise_level=0.2, n_samples=20, generator=None):
    return noisy_gradients(f, x, noise_level, n_samples, generator).mean(dim=0)


def wasserstein_grad(f, x, noise_level=0.2, n_samples=20, reg=1e-3, generator=None):
    return wasserstein_barycenter(noisy_gradients(f, x, noise_level, n_samples, generator), reg)
