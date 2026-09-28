"""Checks of the physics and of the gradients of the toy model.

    python -m synthetic.sanity_checks
"""

import sys

import torch

from .advection import DTYPE, Event, Nowcast
from .methods import base_grad, input_gradients, noisy_gradients
from .metrics import centroid, distance

EVENTS = [Event((32.0, 32.0), (0.8, 0.35), 3.0), Event((31.3, 32.6), (-0.52, 0.71), 2.5),
          Event((32.4, 31.2), (0.13, -0.97), 4.0)]


def check(name, error, tolerance):
    ok = error <= tolerance
    print(f"{'PASS' if ok else 'FAIL'}  {name}: {error:.2e} (tolerance {tolerance:.0e})")
    return ok


def check_event(event, model):
    q, q_prev = model.observation(event), model.observation(event, time=-1)
    velocity = torch.tensor([event.velocity], dtype=DTYPE)
    forecast = model.extrapolate(q[None], velocity)[0]
    results = []

    # Advection-diffusion of a Gaussian: Gaussian at source + lead * velocity, variance s^2 + 2 nu t
    # (up to its periodic images, below 1e-9 here).
    s2 = event.radius**2 + 2 * model.diffusion * model.lead
    dx, dy = model.offsets([c + model.lead * v for c, v in zip(event.source, event.velocity)])
    exact = event.radius**2 / s2 * torch.exp(-(dx**2 + dy**2) / (2 * s2))
    results.append(check("forecast = analytic solution (max abs error)", (forecast - exact).abs().max().item(),
                         1e-8))
    results.append(check("mass conservation (relative error)", abs(forecast.sum().item() / q.sum().item() - 1),
                         1e-12))

    # Block matching recovers the velocity to the resolution of its search grid.
    motion = model.estimate_motion(q[None], q_prev)[0]
    results.append(check("estimated - true velocity (px/step)", (motion - velocity[0]).abs().max().item(),
                         0.5 / model.upsample))

    # Gradient = ROI weights carried back by the adjoint (velocity -m, same diffusion).
    f = model.target(event)
    gradient = base_grad(f, q)
    adjoint = model.extrapolate(model.roi(event)[None], -motion[None])[0]
    results.append(check("gradient = adjoint solution (max abs error)", (gradient - adjoint).abs().max().item(),
                         1e-12))

    # Gradient = central finite difference along a random direction.
    direction = torch.randn(q.shape, generator=torch.Generator().manual_seed(0), dtype=DTYPE)
    eps = 1e-6
    finite_difference = (f((q + eps * direction)[None]) - f((q - eps * direction)[None])).item() / (2 * eps)
    results.append(check("directional derivative, autograd vs finite difference (relative error)",
                         abs((gradient * direction).sum().item() / finite_difference - 1), 1e-6))

    # With the motion fixed, the model is linear: input noise leaves the gradient unchanged.
    weights = model.roi(event)

    def fixed_motion(x):
        return (model.extrapolate(x, velocity.expand(len(x), 2)) * weights).sum(dim=(-2, -1))

    gradients = noisy_gradients(fixed_motion, q, 0.4, 5, torch.Generator().manual_seed(0))
    results.append(check("fixed motion: perturbed - clean gradient (max abs)",
                         (gradients - input_gradients(fixed_motion, q[None])).abs().max().item(), 1e-15))

    # With estimated motion, each perturbed gradient is the clean one moved by -lead * (motion error).
    noisy = q + 0.2 * torch.randn((5, *q.shape), generator=torch.Generator().manual_seed(1), dtype=DTYPE)
    cx, cy = centroid(gradient)
    predicted = [(cx - model.lead * ex, cy - model.lead * ey)
                 for ex, ey in (model.estimate_motion(noisy, q_prev) - motion).tolist()]
    moved = [centroid(g) for g in input_gradients(f, noisy)]
    results.append(check("perturbed gradient centroid - (clean centroid - lead * motion error) (px)",
                         max(distance(p, m, q.shape) for p, m in zip(predicted, moved)), 1e-4))
    return results


def main():
    results = [ok for event in EVENTS for ok in check_event(event, Nowcast(lead=24))]
    print(f"{sum(results)}/{len(results)} checks passed")
    return all(results)


if __name__ == "__main__":
    sys.exit(0 if main() else 1)
