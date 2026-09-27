"""Figures: attribution maps (Fig. 2), displacement curves (Fig. 4), OT mass transport (Fig. 1c)."""

import os

import cartopy.crs as ccrs
import cartopy.feature as cfeature
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Patch, Rectangle
from skimage.transform import resize

PROJ = ccrs.PlateCarree()

STYLE = {
    "font.family": "serif",
    "font.serif": ["Times New Roman", "DejaVu Serif"],
    "font.size": 9,
    "axes.titlesize": 9,
    "axes.labelsize": 8,
    "xtick.labelsize": 7,
    "ytick.labelsize": 7,
    "legend.fontsize": 7,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.02,
}

DISPLAY_NAMES = {
    "WassersteinGrad": r"WG$_{\mathrm{Bary}}$ (Ours)",
    "WassersteinGradMask": r"WG$_{\mathrm{Bary}}\times$Grad (Ours)",
}


def _numpy(a):
    return a.detach().cpu().numpy() if hasattr(a, "detach") else np.asarray(a)


def _signed_unit(a):
    """Scale a map to [-1, 1] by its maximum absolute value."""
    a = _numpy(a)
    return a / (np.nanmax(np.abs(a)) + 1e-12)


def _unit_range(a):
    return (a - np.nanmin(a)) / (np.nanmax(a) - np.nanmin(a) + 1e-12)


def _map_axis(fig, position, target):
    """Map axis with coastlines, borders and the ROI box."""
    ax = fig.add_subplot(position, projection=PROJ)
    ax.add_feature(cfeature.COASTLINE, linewidth=0.5, edgecolor="0.3")
    ax.add_feature(cfeature.BORDERS, linewidth=0.4, edgecolor="0.5", linestyle="--")
    if target:
        lon0, lon1, lat0, lat1 = target
        ax.add_patch(
            Rectangle(
                (lon0, lat0),
                lon1 - lon0,
                lat1 - lat0,
                linewidth=1.0,
                edgecolor="black",
                facecolor="none",
                transform=PROJ,
                zorder=5,
            )
        )
    return ax


def _show(fig, ax, data, extent, **kwargs):
    im = ax.imshow(data, origin="lower", extent=extent, transform=PROJ, **kwargs)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.03, shrink=0.88).ax.tick_params(labelsize=6)
    return im


def _save(fig, *paths):
    for path in paths:
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        fig.savefig(path)
    plt.close(fig)


def plot_attribution(attribution, input_field, truth, forecast, extent, target, title, path):
    """Input channel, ground truth, forecast (shared colour scale) and attribution map."""
    vmin, vmax = min(truth.min(), forecast.min()), max(truth.max(), forecast.max())
    panels = [
        ("Input", input_field, {}),
        ("Ground truth", truth, {"vmin": vmin, "vmax": vmax}),
        ("Forecast", forecast, {"vmin": vmin, "vmax": vmax}),
        ("Attribution", _signed_unit(attribution), {"vmin": -1, "vmax": 1}),
    ]
    with plt.rc_context(STYLE):
        fig = plt.figure(figsize=(14, 3.2))
        for k, (name, data, limits) in enumerate(panels):
            ax = _map_axis(fig, 141 + k, target)
            _show(fig, ax, data, extent, cmap="RdBu_r", **limits)
            ax.set_title(name)
        fig.suptitle(title)
        _save(fig, path)


def plot_comparison(attributions, extent, target, path, max_cols=4, panel_size=(2.6, 2.2)):
    """Attribution maps of several explainers, each scaled to [-1, 1] (Fig. 2).

    Saved as ``path``.png and ``path``.pdf.
    """
    names = list(attributions)
    n_cols = min(len(names), max_cols)
    n_rows = -(-len(names) // n_cols)
    with plt.rc_context(STYLE):
        fig = plt.figure(figsize=(n_cols * panel_size[0] + 0.4, n_rows * panel_size[1] + 0.4))
        grid = fig.add_gridspec(n_rows, n_cols, wspace=0.22, hspace=0.4)
        for k, name in enumerate(names):
            ax = _map_axis(fig, grid[k // n_cols, k % n_cols], target)
            _show(fig, ax, _signed_unit(attributions[name]), extent, cmap="RdBu_r", vmin=-1, vmax=1)
            ax.set_title(DISPLAY_NAMES.get(name, name))
        _save(fig, f"{path}.png", f"{path}.pdf")


def plot_displacement(summary, path, km_per_pixel=2.5):
    """Peak (left) and centroid (right) displacement of the gradient map against the
    noise level, with the relative forecast error on the right axis (Fig. 4).

    Bands are +/- 1 SEM across events; ``summary`` is ``aggregate_displacement`` output.
    """
    noise = summary["noise"]

    def band(ax, key, color, label, scale=1.0, linestyle="-"):
        mean, sem = summary[key] * scale, summary[f"{key}_sem"] * scale
        ax.fill_between(noise, mean - sem, mean + sem, color=color, alpha=0.2)
        ax.plot(noise, mean, color=color, linewidth=1.8, linestyle=linestyle, label=label)
        ax.tick_params(axis="y", labelcolor=color)

    with plt.rc_context(STYLE):
        fig, (ax_peak, ax_centroid) = plt.subplots(1, 2, figsize=(11.0, 4.0), sharex=True)
        band(ax_peak, "peak", "#ff7f0e", "Peak displacement", km_per_pixel, "--")
        ax_peak.set_ylabel("Peak displacement (km)", color="#ff7f0e")
        band(ax_centroid, "centroid", "#1f77b4", "Centroid displacement", km_per_pixel)
        ax_centroid.set_ylabel("Centroid displacement (km)", color="#1f77b4")
        ax_error = ax_centroid.twinx()
        band(ax_error, "rmse", "#d62728", "Relative prediction error")
        ax_error.set_ylabel("Relative prediction error", color="#d62728")

        for ax in (ax_peak, ax_centroid):
            ax.set_xlabel(r"Noise level $\sigma$ (fraction of input range)")
            ax.grid(True, axis="y", linestyle=":", linewidth=0.5, alpha=0.6)
        ax_peak.legend(loc="upper left")
        handles = (
            ax_centroid.get_legend_handles_labels()[0] + ax_error.get_legend_handles_labels()[0]
        )
        ax_centroid.legend(handles=handles, loc="upper left")
        fig.text(
            0.99,
            0.01,
            f"Mean ± SEM across n={summary['n_events']} events",
            ha="right",
            fontsize=7,
            color="gray",
            style="italic",
        )
        fig.tight_layout()
        _save(fig, path)


def plot_transport(clean_gradient, perturbed_gradient, plan, grid_shape, extent, target, path):
    """Clean and perturbed gradients and the OT mass flow between them (Fig. 1c).

    ``plan`` and ``grid_shape`` come from ``displacement.transport_plan``. Red:
    mass leaving the perturbed map; green: mass arriving on the clean map.
    """
    H, W = clean_gradient.shape
    h, w = grid_shape
    sent = resize(plan.sum(axis=1).reshape(h, w), (H, W), anti_aliasing=True)
    received = resize(plan.sum(axis=0).reshape(h, w), (H, W), anti_aliasing=True)

    with plt.rc_context(STYLE):
        fig = plt.figure(figsize=(7, 2.6))
        grid = fig.add_gridspec(1, 3, wspace=0.25, left=0.04, right=0.96, top=0.88, bottom=0.04)
        axes = [_map_axis(fig, grid[0, k], target) for k in range(3)]
        panels = {"Clean gradient": clean_gradient, "Perturbed gradient": perturbed_gradient}
        for k, (title, gradient) in enumerate(panels.items()):
            im = axes[k].imshow(
                _signed_unit(gradient),
                origin="lower",
                extent=extent,
                transform=PROJ,
                cmap="RdBu_r",
                vmin=-1,
                vmax=1,
            )
            axes[k].set_title(title)
        fig.colorbar(im, ax=axes[:2], fraction=0.03, pad=0.005, shrink=0.65).ax.tick_params(
            labelsize=6
        )

        for mass, cmap in ((sent, "Reds"), (received, "Greens")):
            axes[2].imshow(
                _unit_range(mass),
                origin="lower",
                extent=extent,
                transform=PROJ,
                cmap=cmap,
                alpha=0.5,
            )
        axes[2].set_title("OT mass transport")
        axes[2].legend(
            handles=[
                Patch(facecolor="red", alpha=0.75, label="Sent (perturbed)"),
                Patch(facecolor="green", alpha=0.75, label="Received (clean)"),
            ],
            loc="lower left",
            fontsize=6,
        )
        _save(fig, path)
