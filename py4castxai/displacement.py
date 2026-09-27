"""Geometric displacement of gradient attributions under input noise (Section 4.2).

* ``displacement_curves``: centroid and peak displacement of the gradient map
  for increasing noise levels, with the relative forecast error (Fig. 4, App. E.2).
* ``transport_plan``: optimal transport plan between a clean and a perturbed
  gradient map, without the mass that stays in place (Fig. 1c, App. E.1).
"""

import numpy as np
import ot
import torch
from scipy.ndimage import center_of_mass
from skimage.transform import resize

from .explainers import BaseGrad
from .inference import predict_step


def _normalized_abs(g):
    g = np.abs(g)
    return g / (np.max(g) + 1e-8)


def centroid(g):
    """Centre of mass (col, row) of |g|."""
    g = np.abs(g)
    if g.sum() == 0:
        H, W = g.shape
        return float(H) / 2, float(W) / 2
    row, col = center_of_mass(g)
    return float(col), float(row)


def peak(g):
    """Location (col, row) of the maximum of |g|."""
    row, col = np.unravel_index(int(np.argmax(np.abs(g))), g.shape)
    return float(col), float(row)


def _distance(p, q):
    return np.sqrt((p[0] - q[0]) ** 2 + (p[1] - q[1]) ** 2)


def displacement_curves(model, batch, ctx, sigma_step=0.1, num_noise_levels=10, num_monte_carlo=10):
    """Displacement of the gradient map of one event for increasing input noise.

    For each noise level sigma_p = p * sigma_step (p = 1..num_noise_levels, as a
    fraction of the range of the explained channel), ``num_monte_carlo`` noisy
    inputs are drawn. For each of them we record the distance (in pixels) between
    the centroids and between the peaks of the noisy and clean |gradient| maps, and
    the forecast RMSE at the last lead time divided by the clean one.

    Returns a dict with ``noise`` (num_noise_levels,) and ``centroid``, ``peak``,
    ``rmse`` arrays of shape (num_noise_levels, num_monte_carlo).
    """
    explainer = BaseGrad(model)
    output_idx = batch.outputs.feature_names_to_idx[ctx.output_name]
    truth = batch.outputs.tensor[0, -1, ..., output_idx]
    prediction, _ = predict_step(model, batch, ctx)
    rmse_ref = torch.sqrt(torch.mean((truth - prediction.tensor[0, -1, ..., output_idx]) ** 2))

    clean_gradient, _ = explainer.gradient(batch, ctx)
    clean_map = _normalized_abs(clean_gradient.detach().cpu().numpy())
    clean_centroid, clean_peak = centroid(clean_map), peak(clean_map)

    clean = batch.inputs.tensor
    input_idx = batch.inputs.feature_names_to_idx[ctx.input_name]
    x_in = clean[..., input_idx].detach()
    value_range = float(x_in.max() - x_in.min())

    shape = (num_noise_levels, num_monte_carlo)
    result = {
        "noise": np.zeros(num_noise_levels),
        "centroid": np.full(shape, np.nan),
        "peak": np.full(shape, np.nan),
        "rmse": np.full(shape, np.nan),
    }
    for p in range(num_noise_levels):
        noise = (p + 1) * sigma_step
        result["noise"][p] = noise
        for m in range(num_monte_carlo):
            noisy = clean.detach().clone()
            noisy[..., input_idx] = x_in + torch.randn_like(x_in) * (noise * value_range)
            batch.inputs.tensor = noisy
            gradient, noisy_prediction = explainer.gradient(batch, ctx)
            noisy_map = _normalized_abs(gradient.detach().cpu().numpy())
            result["centroid"][p, m] = _distance(centroid(noisy_map), clean_centroid)
            result["peak"][p, m] = _distance(peak(noisy_map), clean_peak)
            rmse = torch.sqrt(
                torch.mean((truth - noisy_prediction.tensor[0, -1, ..., output_idx]) ** 2)
            )
            result["rmse"][p, m] = float((rmse / rmse_ref).detach().cpu())
    batch.inputs.tensor = clean
    return result


def aggregate_displacement(results):
    """Average each event over its Monte Carlo draws, then mean and SEM across events."""
    summary = {"noise": results[0]["noise"], "n_events": len(results)}
    for key in ("centroid", "peak", "rmse"):
        per_event = np.nanmean(np.stack([r[key] for r in results]), axis=2)
        n = np.sum(~np.isnan(per_event), axis=0)
        summary[key] = np.nanmean(per_event, axis=0)
        summary[f"{key}_sem"] = np.nanstd(per_event, axis=0, ddof=1) / np.sqrt(np.maximum(n, 1))
    return summary


def transport_plan(clean_gradient, perturbed_gradient, scale=16, quantile=90):
    """Exact OT plan from a perturbed to the clean gradient map (App. E.1).

    Both maps are reduced to |G| above their ``quantile``-th percentile,
    downsampled by ``scale`` and normalised to probability measures; the cost is
    the Euclidean distance between grid points, scaled to [0, 1]. The diagonal of
    the plan (mass that does not move) is set to zero.

    Returns the plan, of shape (h * w, h * w), and the coarse grid shape (h, w).
    """

    def measure(g):
        g = np.abs(g)
        g = np.where(g > np.percentile(g, quantile), g, 0)
        g = resize(g, (g.shape[0] // scale, g.shape[1] // scale), anti_aliasing=True)
        a = g.flatten().astype(np.float64)
        return a / a.sum(), g.shape

    a, (h, w) = measure(perturbed_gradient)
    b, _ = measure(clean_gradient)
    rows, cols = np.meshgrid(np.arange(h), np.arange(w), indexing="ij")
    coords = np.stack([rows.flatten(), cols.flatten()], axis=1).astype(np.float64)
    cost = ot.dist(coords, coords, metric="euclidean")
    cost /= cost.max()
    plan = ot.emd(a, b, cost)
    np.fill_diagonal(plan, 0)
    return plan, (h, w)
