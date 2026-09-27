"""
Visualization Module for Py4CastXai
====================================

Contains all plotting and visualization functions organized by type.
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from matplotlib.patches import Rectangle
import torch
import copy
from typing import Dict, List, Optional



#####SV plot ##############
import os
import numpy as np
import torch
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from matplotlib.patches import Rectangle

# Match your existing palette
BLUE   = "#1f77b4"
ORANGE = "#ff7f0e"
RED    = "#d62728"
GREY   = "#7f7f7f"
# py4castxai/plotting/wavelet_plots.py
import numpy as np
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature


def plot_scale_spectrum(spectra, bands, labels=None, ax=None, title=None):
    """Figure-3 analogue: normalized importance per wavelength band.

    spectra : list of lists — one spectrum per model/config/lead time
    bands   : list of str   — from attribution.bands_km()
    """
    if ax is None:
        _, ax = plt.subplots(figsize=(8, 4))
    n = len(spectra)
    w = 0.8 / n
    x = np.arange(len(bands))
    for i, s in enumerate(spectra):
        ax.bar(x + (i - (n - 1) / 2) * w, s, w,
               label=labels[i] if labels else None)
    ax.set_xticks(x)
    ax.set_xticklabels(bands, rotation=45, ha="right")
    ax.set_xlabel("Wavelength band (km)   —   fine → coarse")
    ax.set_ylabel("Normalized importance [-]")
    if title:
        ax.set_title(title)
    if labels:
        ax.legend()
    ax.figure.tight_layout()
    return ax


DEFAULT_GROUPS = (
    ("Fine\n5–20 km",          [0, 1]),
    ("Intermediate\n20–80 km", [2, 3]),
    ("Coarse\n80–320 km",      [4, 5]),
    ("Synoptic\n>320 km",      [6]),
)


def plot_scale_maps(maps, extent, groups=DEFAULT_GROUPS, title=None,
                    cmap="magma"):
    """Figure-2d / Figure-4 analogue: where each scale band matters.

    maps : (H, W, J+1) array or tensor — from attribution.maps()
    """
    if isinstance(maps, torch.Tensor):
        maps = maps.detach().cpu().numpy()

    proj = ccrs.PlateCarree()
    fig, axes = plt.subplots(1, len(groups), figsize=(4.2 * len(groups), 4.2),
                             subplot_kw={"projection": proj})
    axes = np.atleast_1d(axes)
    for ax, (name, idxs) in zip(axes, groups):
        m = maps[..., idxs].sum(-1)
        m = m / (m.max() + 1e-12)
        im = ax.imshow(m, extent=extent, origin="lower", transform=proj,
                       cmap=cmap, vmin=0, vmax=1)
        ax.coastlines(resolution="50m", linewidth=0.6, color="white")
        ax.add_feature(cfeature.BORDERS, linewidth=0.4, edgecolor="white")
        ax.set_title(name, fontsize=10)
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    if title:
        fig.suptitle(title)
    fig.tight_layout()
    return fig


def plot_orientation(orient, level, dx_km=2.5, title=None):
    """Zonal vs meridional structure at one scale.

    orient : (3, h_j, w_j) — from attribution.orientation(level)
    """
    if isinstance(orient, torch.Tensor):
        orient = orient.detach().cpu().numpy()
    names = ["Horizontal detail\n(zonal structure)",
             "Vertical detail\n(meridional structure)",
             "Diagonal detail"]
    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    vmax = orient.max() + 1e-12
    for ax, m, nm in zip(axes, orient, names):
        im = ax.imshow(m, origin="lower", cmap="magma", vmin=0, vmax=vmax)
        ax.set_title(nm, fontsize=10)
        ax.axis("off")
        fig.colorbar(im, ax=ax, fraction=0.046)
    lo, hi = 2 ** (level + 1) * dx_km, 2 ** (level + 2) * dx_km
    fig.suptitle(title or f"Level {level + 1}  (~{lo:.0f}–{hi:.0f} km)")
    fig.tight_layout()
    return fig


def wavelet_mosaic(attribution, sample=0, t=0):
    """Classic nested-quadrant layout. Returns (H, W) array to imshow.
    Debugging view — shows all bands at native resolution at once."""
    B, T, _, _ = attribution.shape
    n = sample * T + t
    canvas = attribution.yl[n, 0].abs()
    canvas = canvas / (canvas.max() + 1e-12)
    for h in reversed(attribution.yh):                 # coarse → fine
        lh, hl, hh = (h[n, 0, i].abs() for i in range(3))
        lh, hl, hh = (v / (v.max() + 1e-12) for v in (lh, hl, hh))
        H, W = lh.shape
        up = F.interpolate(canvas[None, None], size=(H, W), mode="nearest")[0, 0]
        canvas = torch.cat([torch.cat([up, lh], dim=1),
                            torch.cat([hl, hh], dim=1)], dim=0)
    return canvas.detach().cpu().numpy()
def _caption(fig, text):
    """Your house-style traceability caption."""
    fig.text(0.99, 0.01, text, ha="right", fontsize=7,
             color="gray", style="italic")


def _add_geo(ax, extent, target=None, projection=ccrs.PlateCarree()):
    """Coastlines/borders + optional target ROI box, matching _run_introduction."""
    ax.add_feature(cfeature.COASTLINE, linewidth=0.6, edgecolor="black")
    ax.add_feature(cfeature.BORDERS, linewidth=0.4, edgecolor="black", linestyle="--")
    ax.set_extent(extent, crs=projection)
    ax.gridlines(linewidth=0.2, linestyle="--", color="gray", alpha=0.4)
    if target is not None:
        lon0, lon1, lat0, lat1 = target[0], target[1], target[2], target[3]
        ax.add_patch(Rectangle(
            (lon0, lat0), lon1 - lon0, lat1 - lat0,
            fill=False, edgecolor=RED, linewidth=1.6,
            transform=projection, zorder=10,
        ))


# ============================================================================
# 1. SPECTRUM DIAGNOSTIC  —  the Gelaro Fig. 2 analog (most important)
# ============================================================================
def plot_sv_spectrum(sigma, alpha, output_path,
                     reweighted_alpha=None, n_events=None):
    """
    Two-panel diagnostic, directly mirroring Fig. 2 of Gelaro et al. (1998).

    (a) Ritz singular values sigma_i vs index  — the amplification spectrum.
        A clear leading structure + gap => low-rank projection is justified.
        A flat, gapless tail => truncation discards real signal.

    (b) |projection coefficients| alpha_i vs index — how the gradient distributes
        over the SV subspace. Roughly WHITE (flat) coefficients are the paper's
        key evidence that the energy/Euclidean metric is a reasonable proxy for
        the error-covariance metric (isotropic projection). A red (decaying)
        distribution after sigma^2 reweighting is expected for the sensitivity.

    Parameters
    ----------
    sigma : 1D array (k,)   Ritz singular values, descending.
    alpha : 1D array (k,)   gradient projection coefficients onto leading SVs.
    reweighted_alpha : optional 1D array, e.g. alpha/sigma**2 (the tilde_e0 form),
                       overlaid as a dashed line for the S vs S^-1 comparison.
    """
    sigma = np.asarray(torch.as_tensor(sigma).detach().cpu())
    alpha = np.abs(np.asarray(torch.as_tensor(alpha).detach().cpu()))
    idx = np.arange(1, len(sigma) + 1)

    fig, (axa, axb) = plt.subplots(1, 2, figsize=(10.0, 3.8))

    # (a) amplification spectrum
    axa.plot(idx, sigma, marker="o", ms=3.5, color=BLUE, linewidth=1.6)
    axa.set_xlabel("SV index $i$")
    axa.set_ylabel(r"Ritz singular value $\hat{\sigma}_i$", color=BLUE)
    axa.set_ylim(bottom=0)
    axa.grid(True, axis="y", linestyle=":", linewidth=0.5, alpha=0.6)
    axa.set_title("(a) Amplification spectrum", fontsize=11)

    # (b) projection coefficients
    axb.plot(idx, alpha, marker="s", ms=3.5, color=BLUE,
             linewidth=1.6, label=r"$|\alpha_i|$ (gradient)")
    # if reweighted_alpha is not None:
    #     rw = np.abs(np.asarray(torch.as_tensor(reweighted_alpha).detach().cpu()))
    #     axb.plot(idx, rw, marker="^", ms=3.5, color=ORANGE, linewidth=1.4,
    #              linestyle="--", label=r"$|\alpha_i|/\hat{\sigma}_i^2$ ($\tilde{e}_0$)")
    #     axb.legend(fontsize=8, loc="upper right")
    axb.set_xlabel("SV index $i$")
    axb.set_ylabel(r"$|\alpha_i|$  (projection coeff.)")
    axb.set_ylim(bottom=0)
    axb.grid(True, axis="y", linestyle=":", linewidth=0.5, alpha=0.6)
    axb.set_title("(b) Gradient projection coefficients", fontsize=11)

    # quick quantitative read-out: cumulative variance captured
    cum = np.cumsum(alpha ** 2) / (np.sum(alpha ** 2) + 1e-12)
    axb.text(0.97, 0.04,
             f"top-{min(5,len(idx))} capture {cum[min(4,len(idx)-1)]*100:.0f}% of $\\|\\alpha\\|^2$",
             ha="right", va="bottom", transform=axb.transAxes,
             fontsize=7.5, color=GREY)

    cap = "Ritz approximation from Lanczos on $L^{T}L$"
    if n_events:
        cap += f" | mean over n={n_events} events"
    _caption(fig, cap)

    plt.tight_layout()
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    plt.savefig(output_path, bbox_inches="tight", dpi=300)
    plt.close(fig)


# ============================================================================
# 2. ATTRIBUTION MAP  —  rank-k projected attribution on the geographic grid
# ============================================================================
def plot_sv_attribution(attr, extent, output_path,
                        target=None, title=None, km_per_pixel=2.5,
                        symmetric=True):
    """
    Plot a single SV-projected attribution map (the rank-k a_k from the explainer),
    geolocated with Cartopy and the target ROI boxed, matching your house style.

    attr : 2D tensor/array (H, W) — the returned attribution for one input channel.
    """
    attr = np.asarray(torch.as_tensor(attr).detach().cpu(), dtype=float)
    attr /=np.max(np.abs(attr))
    proj = ccrs.PlateCarree()

    fig, ax = plt.subplots(figsize=(6.2, 5.0), subplot_kw={"projection": proj})

    lon = np.linspace(extent[0], extent[1], attr.shape[1])
    lat = np.linspace(extent[2], extent[3], attr.shape[0])
    lon2d, lat2d = np.meshgrid(lon, lat)

    if symmetric:
        v = np.percentile(np.abs(attr), 99) + 1e-12
        vmin, vmax, cmap = -1, 1, "RdBu_r"
    else:
        vmin, vmax = np.percentile(attr, [2, 98]); cmap = "viridis"

    # vmin,vmax = np.min(attr),np.max(attr)
    pc = ax.pcolormesh(lon2d, lat2d, attr, cmap="RdBu_r", vmin=vmin, vmax=vmax,
                       transform=proj, shading="auto")
    _add_geo(ax, extent, target=target, projection=proj)
    ax.axis("off")

    cb = fig.colorbar(pc, ax=ax, fraction=0.046, pad=0.04)
    cb.ax.tick_params(labelsize=8)
    cb.set_label("attribution (a.u.)", fontsize=9)

    if title:
        ax.set_title(title, fontsize=11)

    _caption(fig, f"rank-$k$ SV projection | {km_per_pixel} km/pixel")
    plt.tight_layout()
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    plt.savefig(output_path, bbox_inches="tight", dpi=150)
    plt.close(fig)


# ============================================================================
# 3. LEADING SV GALLERY  —  the Gelaro Fig. 5 / Fig. 13 analog
def plot_sv_gallery(
    V,
    sigma,
    in_shape,
    in_channel_idx,
    extent,
    output_path,
    target=None,
    n_show=6,
    km_per_pixel=2.5,
    dpi=150,
    stride=1,
):
    """
    Plot leading singular vectors while keeping PDF size manageable.

    Parameters
    ----------
    stride : int
        Spatial subsampling factor. 1 = no downsampling.
        2 = keep every other pixel, etc.
    dpi : int
        Resolution used for rasterized artists.
    """

    V = torch.as_tensor(V).detach().cpu()
    sigma = np.asarray(torch.as_tensor(sigma).detach().cpu())

    k = min(n_show, V.shape[1])
    proj = ccrs.PlateCarree()

    ncol = 3
    nrow = int(np.ceil(k / ncol))

    fig, axes = plt.subplots(
        nrow,
        ncol,
        figsize=(4.0 * ncol, 3.2 * nrow),
        subplot_kw={"projection": proj},
        constrained_layout=True,
    )

    axes = np.atleast_1d(axes).ravel()

    lon = np.linspace(extent[0], extent[1], in_shape[3])
    lat = np.linspace(extent[2], extent[3], in_shape[2])

    lon2d, lat2d = np.meshgrid(lon, lat)

    if stride > 1:
        lon2d = lon2d[::stride, ::stride]
        lat2d = lat2d[::stride, ::stride]

    # max_field = 0
    # for i in range(k):
    #     ax = axes[i]

    #     field = (
    #         V[:, i]
    #         .view(*in_shape)[0, 0, :, :, in_channel_idx]
    #         .numpy()
    #     )
    #     m = np.max(np.abs(field))
    #     if max_field<m:
    #         max_field = m
    
    for i in range(k):
        ax = axes[i]

        field = (
            V[:, i]
            .view(*in_shape)[0, 0, :, :, in_channel_idx]
            .numpy()
        )
        field/= np.max(np.abs(field))
        # field = field / np.max(np.abs(field))   # optional global normalization for stability

        # thr = np.quantile(np.abs(field), 0.99)
        # masked = np.where(np.abs(field) >= thr, field, 0)

        # # normalize AFTER masking, but only by max of kept values
        # field = masked / np.max(np.abs(masked))
        if stride > 1:
            field = field[::stride, ::stride]

        vmax = np.percentile(np.abs(field), 99) + 1e-12
        # vmin,vmax = np.min(field),np.max(field)

        ax.pcolormesh(
            lon2d,
            lat2d,
            field,
            cmap="RdBu_r",
            vmin=-1,
            vmax=1,
            transform=proj,
            shading="auto",
            rasterized=True,   # key for small PDFs
        )

        _add_geo(ax, extent, target=target, projection=proj)

        ax.set_title(
            rf"$u_{{{i+1}}}$, $\hat{{\sigma}}={sigma[i]:.1f}$",
            fontsize=10,
        )

        ax.axis("off")

    for j in range(k, len(axes)):
        axes[j].axis("off")

    _caption(fig, f"leading right singular vectors | {km_per_pixel} km/pixel")

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)

    plt.savefig(
        output_path,
        dpi=dpi,
        bbox_inches="tight",
    )

    plt.close(fig)

# ============================================================================
# EVALUATION PLOTS (evaluation_plots.py)
# ============================================================================

def plot_boxplot(results: Dict, metric_name: str, path: str, 
                 num_steps: int, save_figs: bool = True):
    """Plot boxplot comparison of metrics across explainers."""
    for i in range(num_steps):
        data = []
        names = []
       
        for expl, vals in results.items():
            if metric_name in vals:
                names.append(expl)
                data.append(vals[metric_name]["vals"][i])
            mean_val = np.median(data) 
            # print(f"{expl}: {mean_val}")       
        plt.figure(figsize=(7, 4))
        sns.boxplot(data=data, palette="pastel")
        plt.xticks(range(len(names)), names)
        plt.ylabel(metric_name)
        plt.title(f'{metric_name} Comparison - Step {i+1}')
        plt.tight_layout()

        if save_figs:
            os.makedirs(path, exist_ok=True)
            plt.savefig(f"{path}/{i+1}_step.png", dpi=300, bbox_inches='tight')
            plt.close()
        else:
            plt.show()


def plot_bar(results: Dict, metric_name: str, path: str, 
             num_steps: int, save_figs: bool = True):
    """Plot bar chart comparison of metrics."""
    for i in range(num_steps):
        names = []
        means = []
        stds = []

        for expl, vals in results.items():
            if metric_name in vals:
                names.append(expl)
                means.append(vals[metric_name]["mean"][i])
                stds.append(vals[metric_name]["std"][i])

        plt.figure(figsize=(7, 4))
        plt.bar(names, means, yerr=stds, capsize=4, color='skyblue')
        plt.title(f"{metric_name} Comparison - Step {i+1}")
        plt.ylabel("Mean Score ± Std")
        plt.tight_layout()
        
        if save_figs:
            os.makedirs(path, exist_ok=True)
            plt.savefig(f"{path}/{i+1}_step.png", dpi=300, bbox_inches='tight')
            plt.close()
        else:
            plt.show()


def plot_bar_ROAD(results: Dict, metric_name: str, path: str, 
                  num_steps: int = 1, save_figs: bool = True):
    """Plot ROAD metric comparison."""
    for i in range(num_steps):
        names = []
        auc = []

        for expl, vals in results.items():
            if metric_name in vals:
                names.append(expl)
                auc.append(vals[metric_name]["auc"][i])
        
        plt.figure(figsize=(7, 4))
        plt.bar(names, auc, capsize=4, color='skyblue')
        plt.title(f"{metric_name} Comparison - Step {i+1}")
        plt.ylabel("AUC Score")
        plt.tight_layout()
        
        if save_figs:
            os.makedirs(path, exist_ok=True)
            plt.savefig(f"{path}/{i+1}_step.png", dpi=300, bbox_inches='tight')
            plt.close()
        else:
            plt.show()


def plot_spider(spider_data: Dict, metric_names: List[str], 
                metric_directions: Dict, fig_path: Optional[str] = None, 
                save: bool = False):
    """Create spider/radar plot for multi-metric comparison."""
    def normalize(values, lower_better=True):
        values = np.array(values)
        if lower_better:
            values = -values
        min_val, max_val = values.min(), values.max()
        return (values - min_val) / (max_val - min_val + 1e-8)

    angles = np.linspace(0, 2 * np.pi, len(metric_names), endpoint=False).tolist()
    angles += angles[:1]

    fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(polar=True))

    for explainer_name, vals in spider_data.items():
        norm_vals = []
        for i, metric in enumerate(metric_names):
            norm = normalize(
                [spider_data[e][i] for e in spider_data], 
                lower_better=metric_directions[metric]
            )
            norm_vals.append(norm[list(spider_data.keys()).index(explainer_name)])

        norm_vals += norm_vals[:1]
        ax.plot(angles, norm_vals, 'o-', linewidth=2, label=explainer_name)
        ax.fill(angles, norm_vals, alpha=0.15)

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(metric_names, size=10)
    ax.set_ylim(0, 1)
    ax.set_yticks([0.25, 0.5, 0.75, 1.0])
    ax.set_yticklabels(['0.25', '0.5', '0.75', '1.0'])
    ax.set_title("Explainer Comparison (Normalized)", size=14, pad=20)
    ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1))
    ax.grid(True)

    plt.tight_layout()
    
    if save and fig_path:
        os.makedirs(fig_path, exist_ok=True)
        plt.savefig(os.path.join(fig_path, "spider_plot.png"), 
                   dpi=300, bbox_inches="tight")
        plt.close()
    else:
        plt.show()


# ============================================================================
# COMPARISON PLOTS (comparison_plots.py)
# ============================================================================

def plot_explainer_comparison(results_all: List[Dict], dataset_info, 
                              cfg_xai: Dict, save_path: str):
    """Plot side-by-side comparison of different explainers."""
    num_examples = len(results_all)
    explainer_names = [k for k in results_all[0].keys() 
                      if k not in ['batch', 'pred']]
    
    extent = cfg_xai["extent"]
    stats = dataset_info.stats
    input_name = cfg_xai["explain"]["input"]
    output_name = cfg_xai["explain"]["output"]
    
    # Determine if forcing input
    batch = results_all[0]["batch"]
    if cfg_xai["explain"]["forcing_input"]:
        input_idx = batch.forcing.feature_names_to_idx[input_name]
    else:
        input_idx = batch.inputs.feature_names_to_idx[input_name]
    
    means_input = torch.asarray(stats[input_name]["mean"])
    std_input = torch.asarray(stats[input_name]["std"])
    means_output = torch.asarray(stats[output_name]["mean"])
    std_output = torch.asarray(stats[output_name]["std"])
    
    # Create figure
    n_rows = len(explainer_names) + 2  # input + pred + explainers
    n_cols = num_examples
    
    fig, axes = plt.subplots(
        n_rows, n_cols,
        figsize=(4.5 * n_cols, 2.5 * n_rows),
        subplot_kw={'projection': ccrs.PlateCarree()}
    )
    
    # Ensure axes is 2D
    if n_cols == 1:
        axes = axes.reshape(-1, 1)
    
    target = cfg_xai.get("target")
    
    for c, example_results in enumerate(results_all):
        batch = example_results["batch"]
        
        # Get input
        if cfg_xai["explain"]["forcing_input"]:
            inp = batch.forcing.tensor[0, 0, :, :, input_idx].detach().cpu()
        else:
            inp = batch.inputs.tensor[0, 0, :, :, input_idx].detach().cpu()
        
        gt = batch.outputs.tensor[0, 0, :, :, 0].detach().cpu()
        pred = example_results["pred"][0, 0, :, :, 0]
        
        # Denormalize
        inp = inp * std_input + means_input
        gt = gt * std_output + means_output
        pred = pred * std_output + means_output
        
        # Row 1: Input
        im0 = axes[0, c].imshow(inp, origin="lower", cmap="RdBu_r", extent=extent)
        if c == 0:
            axes[0, c].set_ylabel("Input (t)", fontsize=10)
        axes[0, c].set_title(f"Example {c+1}", fontsize=11)
        
        # Row 2: Prediction
        im1 = axes[1, c].imshow(pred, origin="lower", cmap="RdBu_r", extent=extent)
        if c == 0:
            axes[1, c].set_ylabel("Prediction (t+1)", fontsize=10)
        
        # Rows 3+: Explainers
        for r, explainer_name in enumerate(explainer_names, start=2):
            attr = example_results[explainer_name][0].squeeze(1)
            
            if cfg_xai["explain"]["forcing_input"]:
                attr = attr[..., input_idx].numpy()
            else:
                attr = attr[0, ..., input_idx].numpy()
            
            im = axes[r, c].imshow(
                attr, origin="lower", cmap="RdBu_r", 
                extent=extent, vmin=-1, vmax=1
            )
            
            if c == 0:
                axes[r, c].set_ylabel(explainer_name, fontsize=10)
            
            if c == n_cols - 1:
                plt.colorbar(im, ax=axes[r, c], orientation="vertical", 
                           fraction=0.046, pad=0.04)
        
        # Add colorbars for first two rows
        if c == n_cols - 1:
            plt.colorbar(im0, ax=axes[0, c], orientation="vertical", 
                        fraction=0.046, pad=0.04)
            plt.colorbar(im1, ax=axes[1, c], orientation="vertical", 
                        fraction=0.046, pad=0.04)
    
    # Add features and target box to all axes
    for r in range(n_rows):
        for c in range(n_cols):
            ax = axes[r, c]
            ax.set_extent(extent, crs=ccrs.PlateCarree())
            ax.add_feature(cfeature.COASTLINE, linewidth=0.3)
            ax.set_xticks([])
            ax.set_yticks([])
            
            if target is not None:
                lon0, lon1 = target[0], target[1]
                lat0, lat1 = target[2], target[3]
                rect = Rectangle(
                    (lon0, lat0), lon1 - lon0, lat1 - lat0,
                    linewidth=2, edgecolor='black', facecolor='none',
                    transform=ccrs.PlateCarree(), zorder=5
                )
                ax.add_patch(copy.deepcopy(rect))
    
    plt.subplots_adjust(wspace=0.05, hspace=0.1)
    plt.suptitle(f"Explainer Comparison: {input_name} → {output_name}", 
                fontsize=14, y=0.995)
    
    # Save
    fig_path = os.path.join(save_path, "multiple_examples_comparison")
    os.makedirs(fig_path, exist_ok=True)
    plt.savefig(
        os.path.join(fig_path, f"{input_name}_{output_name}.png"),
        dpi=300, bbox_inches="tight"
    )
    plt.close()


# ============================================================================
# SPATIAL ANALYSIS PLOTS (spatial_plots.py)
# ============================================================================

def plot_radial_distribution(all_explanations: Dict, extent: List, 
                             target: List, num_bins: int = 50,
                             save_path: Optional[str] = None):
    """Plot radial distribution of attributions from target center."""
    import torch
    
    def normalize_expl(a):
        if isinstance(a, np.ndarray):
            a = torch.from_numpy(a).to(torch.float32)
        return a.abs() / (a.abs().max() + 1e-8)
    
    def haversine(lon1, lat1, lon2, lat2):
        R = 6371.0  # Earth radius in km
        lon1, lat1, lon2, lat2 = map(np.radians, [lon1, lat1, lon2, lat2])
        dlon = lon2 - lon1
        dlat = lat2 - lat1
        a = np.sin(dlat/2)**2 + np.cos(lat1)*np.cos(lat2)*np.sin(dlon/2)**2
        return 2 * R * np.arcsin(np.sqrt(a))
    
    def compute_distance_map(lon_grid, lat_grid, cx, cy):
        return haversine(lon_grid, lat_grid, cx, cy)
    
    def compute_lonlat_grid(H, W, extent):
        lon_min, lon_max, lat_min, lat_max = extent
        lons = np.linspace(lon_min, lon_max, W)
        lats = np.linspace(lat_min, lat_max, H)
        lon_grid, lat_grid = np.meshgrid(lons, lats)
        return lon_grid, lat_grid
    
    def radial_profile_single(attr, distance_map, num_bins=50):
        arr = attr.detach().cpu().numpy()
        arr = normalize_expl(arr)
        
        distances = distance_map.flatten()
        attributions = arr.flatten()
       
        bins = np.linspace(0, distances.max(), num_bins + 1)
        centers = 0.5 * (bins[:-1] + bins[1:])
        radial_mean = np.zeros(num_bins)

        for i in range(num_bins):
            mask = (distances >= bins[i]) & (distances < bins[i+1])
            if mask.sum() > 0:
                radial_mean[i] = attributions[mask].mean()
            else:
                radial_mean[i] = np.nan

        return centers, radial_mean
    
    # Get target center
    lon0, lon1, lat0, lat1 = target
    center_lon = (lon0 + lon1) / 2
    center_lat = (lat0 + lat1) / 2
    
    # Get shape from first explanation
    # first_exp = list(all_explanations.values())[0][0][0]
    # _, _, H, W, _ = first_exp.shape
    H,W = 512,640
    # Compute distance map
    lon_grid, lat_grid = compute_lonlat_grid(H, W, extent)
    distance_map = compute_distance_map(lon_grid, lat_grid, center_lon, center_lat)
    
    # Plot
    plt.figure(figsize=(12, 7))
    
    for var_name, explanations in all_explanations.items():
        profiles = []
        for exp in explanations:
            radii, profile = radial_profile_single(exp[0], distance_map, num_bins)
            profiles.append(profile)
        
        profiles = np.array(profiles)
        mean_profile = np.nanmean(profiles, axis=0)
        
        plt.plot(radii, mean_profile, linewidth=3, label=var_name)
    
    plt.xlabel("Distance from target center (km)", fontsize=12)
    plt.xscale("log")
    plt.ylabel("Normalized Attribution", fontsize=12)
    plt.title("Radial Attribution Distribution", fontsize=14)
    plt.grid(True, alpha=0.3)
    plt.legend(fontsize=10)
    plt.tight_layout()
    
    if save_path:
        os.makedirs(save_path, exist_ok=True)
        plt.savefig(os.path.join(save_path, "radial_distribution.png"), 
                   dpi=300, bbox_inches="tight")
        plt.close()
    else:
        plt.show()


# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

def normalize_attr(a):
    """Normalize attribution for visualization."""
    if isinstance(a, np.ndarray):
        a = torch.from_numpy(a).float()
    return a / (a.abs().max() + 1e-8)


def denormalize(tensor, mean, std):
    """Denormalize tensor using mean and std."""
    return tensor * std + mean


def add_target_box(ax, target, projection=ccrs.PlateCarree()):
    """Add target region box to axis."""
    if target is None:
        return
    
    lon0, lon1, lat0, lat1 = target
    rect = Rectangle(
        (lon0, lat0), lon1 - lon0, lat1 - lat0,
        linewidth=2, edgecolor='black', facecolor='none',
        transform=projection, zorder=5
    )
    ax.add_patch(rect)