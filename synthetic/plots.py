"""The three figures of the toy experiment."""

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle

from .metrics import centroid

COLORS = {"SmoothGrad": "tab:blue", "WassersteinGrad": "tab:red", "BaseGrad": "0.35"}
CROP = 20  # attribution maps are shown in a window of +-CROP pixels around the source
SOURCE_STYLE = {"edgecolor": "tab:green", "linewidth": 1.2, "linestyle": "--"}


def _show(ax, a, title, cmap="Reds", window=None):
    """|A| scaled to its maximum, origin at the bottom left, optionally within +-CROP pixels of ``window``."""
    a = np.abs(a.detach().numpy())
    ax.imshow(a / a.max(), origin="lower", cmap=cmap, vmin=0, vmax=1, interpolation="nearest")
    ax.set_title(title, fontsize=9)
    ax.set_xticks([])
    ax.set_yticks([])
    if window is not None:
        ax.set_xlim(window[0] - CROP, window[0] + CROP)
        ax.set_ylim(window[1] - CROP, window[1] + CROP)


def _box(ax, centre, half_width, **kwargs):
    """Outline of the square of side 2 * half_width + 1 pixels centred on ``centre``."""
    corner = (centre[0] - half_width - 0.5, centre[1] - half_width - 0.5)
    ax.add_patch(Rectangle(corner, 2 * half_width + 1, 2 * half_width + 1, fill=False, **kwargs))


def plot_setup(model, event, forecast, clean, perturbed, noise_level, path):
    """Observation, forecast and ROI, clean gradient, and gradients at perturbed inputs."""
    source, h = model.source_centre(event), model.roi_half_width
    fig, axes = plt.subplots(2, 4, figsize=(11, 6.4))
    (a, b, c, d), row = axes

    _show(a, model.observation(event), "(a) observation $q$, previous one (dashed)\nand estimated motion",
          cmap="Blues")
    a.contour(model.observation(event, time=-1).numpy(), levels=[0.5], colors="0.3", linestyles="--",
              linewidths=0.8)
    a.arrow(*event.source, 8 * event.velocity[0], 8 * event.velocity[1], color="k", width=0.4)
    _box(a, source, CROP - 0.5, edgecolor="0.5", linewidth=0.8, linestyle=":")
    _show(b, forecast, f"(b) forecast at lead {model.lead} steps\nand ROI", cmap="Blues")
    _box(b, model.roi_centre(event), h, edgecolor="k", linewidth=1.2)
    _show(c, clean, "(c) BaseGrad, clean input\n(= ROI carried back)", window=source)
    _box(c, source, h, **SOURCE_STYLE)

    centroids = np.array([centroid(g) for g in perturbed])
    _show(d, clean, f"(d) centroids of {len(perturbed)} perturbed\ngradients, $\\alpha$ = {noise_level}",
          window=source)
    d.scatter(centroids[:, 0], centroids[:, 1], s=12, color="tab:blue", edgecolors="w", linewidths=0.5)
    _box(d, source, h, **SOURCE_STYLE)

    for i, ax in enumerate(row):
        _show(ax, perturbed[i], f"BaseGrad, perturbed input #{i + 1}", window=source)
        _box(ax, source, h, **SOURCE_STYLE)
    fig.suptitle("Input noise changes the selected motion and displaces the gradient "
                 "(green: true source region; (c)-(d) and bottom row: dotted window of (a))", fontsize=10)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_comparison(model, event, clean, maps, n_samples, path):
    """SmoothGrad (top) against WassersteinGrad (bottom) at increasing noise levels (columns).

    ``maps`` maps each noise level to {"SmoothGrad": map, "WassersteinGrad": map}, both
    aggregating the same ``n_samples`` perturbed gradients.
    """
    source, h = model.source_centre(event), model.roi_half_width
    levels = list(maps)
    fig, axes = plt.subplots(2, len(levels) + 1, figsize=(2.6 * (len(levels) + 1), 5.6))
    _show(axes[0, 0], clean, "BaseGrad, clean input\n(reference)", window=source)
    axes[1, 0].axis("off")
    for j, level in enumerate(levels, start=1):
        for i, name in enumerate(("SmoothGrad", "WassersteinGrad")):
            _show(axes[i, j], maps[level][name], f"{name}\n$\\alpha$ = {level}", window=source)
    for ax in axes.flat:
        if ax.images:
            _box(ax, source, h, **SOURCE_STYLE)
    fig.suptitle(f"Aggregation of the same {n_samples} perturbed gradients, lead {model.lead} steps "
                 "(green: true source region)", fontsize=10)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_metrics(summary, path):
    """Mean +- SEM over events of the displacement and localisation metrics against the noise level.

    ``summary`` maps each lead time to {"noise_levels": [...], group: {metric: {"mean": [...], "sem": [...]}}}.
    """
    fig, axes = plt.subplots(2, 2, figsize=(10, 8))
    (a, b), (c, d) = axes
    leads = sorted(summary, key=int)
    styles = dict(zip(leads, ("--", "-")))
    for lead in leads:
        stats = summary[lead]
        _curve(a, stats, "perturbed", "displacement", COLORS["BaseGrad"], styles[lead])
        for name in ("SmoothGrad", "WassersteinGrad"):
            _curve(b, stats, name, "centroid_error", COLORS[name], styles[lead])
            _curve(c, stats, name, "peak_error", COLORS[name], styles[lead])
            _curve(d, stats, name, "rms_radius", COLORS[name], styles[lead])
        d.axhline(stats["clean"]["rms_radius"]["mean"][0], color=COLORS["BaseGrad"], linestyle=styles[lead],
                  linewidth=0.8)
    a.set_title("(a) displacement of the perturbed gradients\n(distance of their centroid to the clean one)")
    b.set_title("(b) centroid error to the true source")
    c.set_title("(c) peak error to the true source")
    d.set_title("(d) RMS distance of attribution mass to the source\n(thin grey: BaseGrad, clean input)")
    for ax in axes.flat:
        ax.title.set_fontsize(10)
        ax.set_xlabel(r"noise level $\alpha$ (noise std / input range)")
        ax.set_ylabel("pixels")
        ax.set_ylim(bottom=0)
        ax.grid(alpha=0.3)
    handles = [Line2D([], [], color=COLORS[name], label=name) for name in ("SmoothGrad", "WassersteinGrad")]
    handles += [Line2D([], [], color="k", linestyle=styles[lead], label=f"lead {lead}") for lead in leads]
    fig.legend(handles=handles, loc="upper center", ncol=len(handles), frameon=False, handlelength=3)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(path, dpi=150)
    plt.close(fig)


def _curve(ax, stats, group, metric, color, style):
    x = np.array(stats["noise_levels"])
    mean, sem = np.array(stats[group][metric]["mean"]), np.array(stats[group][metric]["sem"])
    ax.plot(x, mean, linestyle=style, marker="o", markersize=3, color=color)
    ax.fill_between(x, mean - sem, mean + sem, color=color, alpha=0.2, linewidth=0)
