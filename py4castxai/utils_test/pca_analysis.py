
import os
import logging
import torch
import numpy as np
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from matplotlib.patches import Rectangle
import copy
from typing import Dict, List, Optional, Tuple

from py4castxai.explainers import explainers_registry
import os
import logging
import copy
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.ticker as mticker
from matplotlib.patches import Rectangle
from matplotlib.lines import Line2D
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from cartopy.mpl.gridliner import LONGITUDE_FORMATTER, LATITUDE_FORMATTER

# ── Publication style ──────────────────────────────────────────────────────────
matplotlib.rcParams.update({
    "font.family":      "serif",
    "font.serif":       ["Computer Modern Roman", "Times New Roman", "DejaVu Serif"],
    "font.size":        12,
    "axes.titlesize":   16,
    "axes.labelsize":   11,
    "xtick.labelsize":  9,
    "ytick.labelsize":  9,
    "legend.fontsize":  11,
    "figure.dpi":       150,
    "savefig.dpi":      300,
    "savefig.bbox":     "tight",
    "text.usetex":      False,   # flip to True if LaTeX is available
})

_CMAP      = "seismic"
_ELLIPSE_COLOR  = "#E63946"
_BARY_COLOR     = "#E63946"
_TARGET_COLOR   = "#2B2D42"
_LAND_COLOR     = "#F5F5EE"
_OCEAN_COLOR    = "#D6E8F0"
_BORDER_COLOR   = "#888888"
_COAST_COLOR    = "#444444"


# ──────────────────────────────────────────────────────────────────────────────
# Internal helpers
# ──────────────────────────────────────────────────────────────────────────────

def _get_ellipse_coords(
    center: List[float],
    a: float,
    b: float,
    theta: float,
    lon_mean: float,
    lon_std: float,
    lat_mean: float,
    lat_std: float,
    N_points: int = 300,
    evec_R: Optional[np.ndarray] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """Return lon/lat arrays tracing the 95 % confidence ellipse."""
    t = np.linspace(0, 2 * np.pi, N_points)
    x_scaled = a * np.cos(t)
    y_scaled = b * np.sin(t)

    if evec_R is not None:
        R = evec_R
    else:
        R = np.array([[np.cos(theta), -np.sin(theta)],
                      [np.sin(theta),  np.cos(theta)]])

    coords    = R @ np.vstack([x_scaled, y_scaled])
    lon_coords = (coords[0] + center[0]) * float(lon_std) + float(lon_mean)
    lat_coords = (coords[1] + center[1]) * float(lat_std) + float(lat_mean)
    return lon_coords, lat_coords


def _style_ax(ax, extent, gridlines= True,left_labels=True):
    """Apply common cartographic styling to a GeoAxes."""
    ax.set_extent(extent, crs=ccrs.PlateCarree())
    ax.add_feature(cfeature.LAND,       facecolor=_LAND_COLOR,  zorder=0,rasterized=True)
    ax.add_feature(cfeature.OCEAN,      facecolor=_OCEAN_COLOR, zorder=0,rasterized=True)
    ax.add_feature(cfeature.COASTLINE,  linewidth=0.8, edgecolor=_COAST_COLOR,  zorder=2)
    ax.add_feature(cfeature.BORDERS,    linewidth=0.5, edgecolor=_BORDER_COLOR, zorder=2,
                   linestyle="--")
    if gridlines:
        gl = ax.gridlines(
            crs=ccrs.PlateCarree(),
            draw_labels=True,
            linewidth=0.4,
            color="gray",
            alpha=0.5,
            linestyle=":",
        )
        gl.left_labels = left_labels   # ← add this
        gl.top_labels    = False
        gl.right_labels  = False
        gl.xformatter    = LONGITUDE_FORMATTER
        gl.yformatter    = LATITUDE_FORMATTER
        gl.xlabel_style  = {"size": 8}
        gl.ylabel_style  = {"size": 8}
    return ax


def _overlay_ellipse_and_bary(
    ax,
    barycenter: torch.Tensor,
    ellipse_params: Dict,
    lon_mean, lon_std, lat_mean, lat_std,
    projection,
):
    """Draw ellipse and barycenter marker on ax."""
    lon_ell, lat_ell = _get_ellipse_coords(
        barycenter.tolist(),
        ellipse_params["a"], ellipse_params["b"],
        ellipse_params["theta"],
        lon_mean, lon_std, lat_mean, lat_std,
        evec_R=ellipse_params["evec_R"],
    )
    bary_lon = barycenter[0].item() * float(lon_std) + float(lon_mean)
    bary_lat = barycenter[1].item() * float(lat_std) + float(lat_mean)

    ax.plot(lon_ell, lat_ell,
            color=_ELLIPSE_COLOR, linewidth=1.8, linestyle="-",
            transform=projection, zorder=5)
    ax.plot(bary_lon, bary_lat,
            marker="+", color=_BARY_COLOR,
            markersize=9, markeredgewidth=2,
            transform=projection, zorder=6)


def _add_target_box(ax, target, projection):
    """Draw target bounding box."""
    if target is None:
        return
    lon0, lon1, lat0, lat1 = target
    rect = Rectangle(
        (lon0, lat0), lon1 - lon0, lat1 - lat0,
        linewidth=1.5, edgecolor=_TARGET_COLOR,
        facecolor="none", linestyle="--",
        transform=projection, zorder=7,
    )
    ax.add_patch(rect)


def _add_shared_colorbar(fig, im, cax, label: str = ""):
    """Add a tidy colorbar."""
    cb = fig.colorbar(im, cax=cax, orientation="vertical")
    cb.set_label(label, fontsize=11)
    cb.ax.tick_params(labelsize=8)
    cb.outline.set_linewidth(0.5)
    return cb


def _legend_handles():
    return [
        Line2D([0], [0], color=_ELLIPSE_COLOR, linewidth=1.8,
               label="95 % influence ellipse"),
        Line2D([0], [0], marker="+", color=_BARY_COLOR, linewidth=0,
               markersize=9, markeredgewidth=2, label="Attribution barycenter"),
    ]


# ──────────────────────────────────────────────────────────────────────────────
# Public plotting functions
# ──────────────────────────────────────────────────────────────────────────────

def plot_single_pca_mode(
    pc_map: torch.Tensor,
    barycenter: torch.Tensor,
    ellipse_params: Dict,
    extent: List[float],
    lon_mean, lon_std, lat_mean, lat_std,
    target=None,
    title: str = "First PCA Mode",
    save_path: str = "pca_mode1.png",
    cbar_label: str = "PCA loading",
):
    """
    Plot a single PCA mode with the influence ellipse.

    Parameters
    ----------
    pc_map       : (H, W) tensor — the PCA spatial map.
    barycenter   : (2,) tensor in *standardised* coordinates.
    ellipse_params: dict returned by _compute_barycenter_ellipse.
    extent       : [lon_min, lon_max, lat_min, lat_max].
    title        : axes title (e.g. variable name + mode number).
    save_path    : full path including filename.
    """
    projection = ccrs.PlateCarree()

    fig = plt.figure(figsize=(8, 6))
    gs  = gridspec.GridSpec(1, 2, width_ratios=[1, 0.035], wspace=0.05)
    ax  = fig.add_subplot(gs[0, 0], projection=projection)
    cax = fig.add_subplot(gs[0, 1])

    _style_ax(ax, extent)

    vmax = float(pc_map.abs().cpu().max())
    im = ax.imshow(
        pc_map.cpu().numpy(),
        transform=projection,
        origin="lower",
        extent=extent,
        cmap=_CMAP,
        vmin=-vmax, vmax=vmax,
        interpolation="bilinear",
        zorder=1,rasterized=True)


    _overlay_ellipse_and_bary(
        ax, barycenter, ellipse_params,
        lon_mean, lon_std, lat_mean, lat_std, projection,
    )
    _add_target_box(ax, target, projection)
    _add_shared_colorbar(fig, im, cax, cbar_label)

    ax.legend(handles=_legend_handles(), loc="lower left",
              framealpha=0.85, edgecolor="0.7", fontsize=11)
    ax.set_title(title, pad=6)

    os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
    fig.savefig(save_path)
    plt.close(fig)
    print(f"Saved → {save_path}")


def plot_two_pca_modes(
    pc1_map: torch.Tensor,
    pc2_map: torch.Tensor,
    barycenter: torch.Tensor,
    ellipse_params: Dict,
    extent: List[float],
    lon_mean, lon_std, lat_mean, lat_std,
    explained_variance_ratio: np.ndarray,
    target=None,
    variable_name: str = "",
    save_path: str = "pca_modes.png",
    cbar_label: str = "PCA loading",
):
    """
    Side-by-side plot of PCA modes 1 and 2 with a shared colorbar.
    """
    projection = ccrs.PlateCarree()

    fig = plt.figure(figsize=(14, 6))
    gs  = gridspec.GridSpec(
        1, 3,
        width_ratios=[1, 1, 0.04],
        wspace=0.08,
    )
    ax1 = fig.add_subplot(gs[0, 0], projection=projection)
    ax2 = fig.add_subplot(gs[0, 1], projection=projection)
    cax = fig.add_subplot(gs[0, 2])

    vmax = max(
        float(pc1_map.abs().cpu().max()),
        float(pc2_map.abs().cpu().max()),
    )
    imshow_kw = dict(
        transform=projection, origin="lower", extent=extent,
        cmap=_CMAP, vmin=-vmax, vmax=vmax,
        interpolation="bilinear", zorder=1,
    )

    for ax, pc_map, mode_idx in zip(
        [ax1, ax2], [pc1_map, pc2_map], [1, 2]
    ):
        _style_ax(ax, extent, gridlines=True)  # labels only on left panel
       
        im = ax.imshow(pc_map.cpu().numpy(), **imshow_kw)
        _overlay_ellipse_and_bary(
            ax, barycenter, ellipse_params,
            lon_mean, lon_std, lat_mean, lat_std, projection,
        )
        _add_target_box(ax, target, projection)

        # evr = explained_variance_ratio[mode_idx - 1] * 100
        ax.set_title(f"PC{mode_idx} ", pad=6)

    ax1.legend(handles=_legend_handles(), loc="lower left",
               framealpha=0.85, edgecolor="0.7", fontsize=11)

    _add_shared_colorbar(fig, im, cax, cbar_label)

    suptitle = f"PCA Analysis — {variable_name}" if variable_name else "PCA Analysis"
    fig.suptitle(suptitle, fontsize=13, y=1.01)

    os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
    fig.savefig(save_path)
    plt.close(fig)
    print(f"Saved → {save_path}")


def plot_multi_variable_pca(
    variables: Dict[str, Dict],
    extent: List[float],
    target=None,
    n_modes: int = 1,
    save_path: str = "pca_multi_variable.png",
    cbar_label: str = "PCA loading",
):
    plt.rcParams.update({
        "font.size":        18,
        "axes.titlesize":   18,
        "axes.labelsize":   17,
        "xtick.labelsize":  15,
        "ytick.labelsize":  15,
        "legend.fontsize":  16,
    })

    n_vars      = len(variables)
    n_cols      = n_vars * n_modes
    projection  = ccrs.PlateCarree()
    col_widths  = [1] * n_cols + [0.04]
    fig = plt.figure(figsize=(5.5 * n_cols + 0.5, 5))
    gs  = gridspec.GridSpec(
        1, n_cols + 1,
        width_ratios=col_widths,
        wspace=0.06,
    )

    last_im = None
    col_idx = 0

    for var_name, res in variables.items():
        pca      = res["pca_results"]
        bary     = res["barycenter"]
        ell      = res["ellipse_params"]
        lon_mean = res["lon_mean"]
        lon_std  = res["lon_std"]
        lat_mean = res["lat_mean"]
        lat_std  = res["lat_std"]

        pc_maps = [pca["pc_1"]] if n_modes == 1 else [pca["pc_1"], pca["pc_2"]]
        vmax    = max(float(m.abs().cpu().max()) for m in pc_maps)
        evr     = pca["explained_variance_ratio"]

        imshow_kw = dict(
            transform=projection, origin="lower", extent=extent,
            cmap=_CMAP, vmin=-vmax, vmax=vmax,
            interpolation="bilinear", zorder=1,
        )

        for mode_idx, pc_map in enumerate(pc_maps, start=1):
            ax = fig.add_subplot(gs[0, col_idx], projection=projection)
            _style_ax(ax, extent, gridlines=True, left_labels=(col_idx == 0))
            last_im = ax.imshow(pc_map.cpu().numpy(), **imshow_kw)
            _overlay_ellipse_and_bary(
                ax, bary, ell, lon_mean, lon_std, lat_mean, lat_std, projection,
            )
            _add_target_box(ax, target, projection)

            evr_pct = evr[mode_idx - 1] * 100
            if n_modes == 1:
                title = f"{var_name}"
            else:
                title = f"{var_name} — PC{mode_idx}\n({evr_pct:.1f} % var.)"

            ax.set_title(title, pad=5)  # inherits axes.titlesize=14
            col_idx += 1

    # Shared colorbar
    cax = fig.add_subplot(gs[0, -1])
    cb  = fig.colorbar(last_im, cax=cax, orientation="vertical")
    cb.set_label(cbar_label)           # inherits font.size=14
    cb.ax.tick_params(labelsize=11)    # ↑ was 8
    cb.outline.set_linewidth(0.4)

    # Shared legend
    fig.legend(
        handles=_legend_handles(),
        loc="lower center",
        ncol=2,
        framealpha=0.85,
        edgecolor="0.7",
        bbox_to_anchor=(0.5, -0.08),   # inherits legend.fontsize=12
    )

    os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
    fig.savefig(save_path)
    plt.close(fig)
    print(f"Saved → {save_path}")


# ──────────────────────────────────────────────────────────────────────────────
# Patched PCAAnalyzer._visualize_pca  (drop-in replacement)
# ──────────────────────────────────────────────────────────────────────────────
class PCAAnalyzer:
    """Performs PCA analysis on attribution maps."""
    
    def __init__(self, model, config, dataloader, dataset_info, infer_ds):
        self.model = model
        self.config = config
        self.dataloader = dataloader
        self.dataset_info = dataset_info
        self.infer_ds = infer_ds
        self.min_lat = 37.5
        self.max_lat = 55.4
        self.min_lon = -12.0
        self.max_lon = 16.0
        self.new_res = 0.025  # The target resolution
        self.extent = self.config.cfg_xai["extent"]
        self.logger = logging.getLogger(__name__)
 
    def get_explainer(self, explainer_name: str, params: Dict):
            """Initialize explainer by name."""
            explainer_cls = explainers_registry.get(explainer_name)
            self.explainer_name = explainer_name
            if explainer_cls is None:
                raise ValueError(f"Unknown explainer: {explainer_name}")
            return explainer_cls(self.model, **params)   
               
    def run(self):
        cfg_xai = self.config.cfg_xai
        all_variables = {}
        import pandas as pd
        metrics_list = [] # ---> NEW: Accumulator for our physical metrics

        for input_var in cfg_xai["multiple_explain"]:
            cfg_xai["explain"]["input"] = input_var
            self.logger.info(f"Processing variable: {input_var}")

            # ── 1. Collect explanations for this variable ──────────────────────

            # ── 2. Prepare data matrix ─────────────────────────────────────────
            X_matrix, X_matrix_pca, coords = self._collect_explanations()


            #threshold
            threshold = 0.05
            print("threshold: ",threshold)
            # # only keep weights above the threshold
            Mask = X_matrix > threshold
            Mask_pca = X_matrix.abs() > threshold

            X_matrix = X_matrix * Mask
            X_matrix_pca = X_matrix_pca * Mask_pca

            # ---> NEW: Calculate Sparsity (% of grid actively used)
            sparsity_pct = Mask.float().mean().item() * 100.0
            
            # ── 3. Normalise coordinates ───────────────────────────────────────
            lon_mean, lon_std = coords[0].mean(), coords[0].std()
            lat_mean, lat_std = coords[1].mean(), coords[1].std()

            Scaling_Std = torch.tensor(
                [lon_std.item(), lat_std.item()], device=coords.device
            ).unsqueeze(1)
            Mean_Orig = torch.tensor(
                [lon_mean.item(), lat_mean.item()], device=coords.device
            ).unsqueeze(1)
            coords_norm = (coords - Mean_Orig) / Scaling_Std

            # ── 4. Barycenter + ellipse ────────────────────────────────────────
            barycenter, ellipse_params = self._compute_barycenter_ellipse(
                X_matrix, coords_norm
            )

            # ── 5. PCA ─────────────────────────────────────────────────────────
            pca_results = self._perform_pca(X_matrix_pca)
            
            
            # ---> NEW: Extract Physical Metrics for the Table
            # Un-normalize the barycenter to get real Lat/Lon coordinates
            bary_lon = barycenter[0].item() * lon_std.item() + lon_mean.item()
            bary_lat = barycenter[1].item() * lat_std.item() + lat_mean.item()
            
            # Extract Ellipse shape parameters
            a = ellipse_params["a"]
            b = ellipse_params["b"]
            theta_rad = ellipse_params["theta"]
            
            # Calculate physical properties
            area_normalized = np.pi * a * b
            orientation_deg = np.degrees(theta_rad)
            anisotropy = a / b if b > 0 else 0
            pc1_variance = pca_results["explained_variance_ratio"][0] * 100.0
            
            # Append to our metrics list
            metrics_list.append({
                "Variable": input_var,
                "Center (Lon, Lat)": f"({bary_lon:.2f}°, {bary_lat:.2f}°)",
                "Spatial Spread (Area)": f"{area_normalized:.2f}",
                "Main Axis (Deg)": f"{orientation_deg:.1f}°",
                "Anisotropy (a/b)": f"{anisotropy:.2f}",
                "Sparsity (% > 0.1)": f"{sparsity_pct:.2f}%",
                "PC1 Dominance": f"{pc1_variance:.1f}%"
            })
            # ── 6. Accumulate ──────────────────────────────────────────────────
            all_variables[input_var] = {
                "pca_results":    pca_results,
                "barycenter":     barycenter,
                "ellipse_params": ellipse_params,
                "lon_mean": lon_mean,
                "lon_std":  lon_std,
                "lat_mean": lat_mean,
                "lat_std":  lat_std,
            }

        # ── 7. Single multi-variable figure ────────────────────────────────────
        save_path = os.path.join(
            cfg_xai["explain"]["exp_plot"]["fig_path"],
            "pca_analysis"
        )
        
        
        # ── 7. Single multi-variable figure and CSV Save ───────────────────────
        save_path = os.path.join(
            cfg_xai["explain"]["exp_plot"]["fig_path"],
            "pca_analysis"
        )
        os.makedirs(save_path, exist_ok=True)
        
        # ---> NEW: Save and Print the Quantitative Metrics Table
        metrics_df = pd.DataFrame(metrics_list)
        csv_path = os.path.join(save_path, "quantitative_metrics.csv")
        metrics_df.to_csv(csv_path, index=False)
        
        print("\n" + "="*80)
        print("EXTRACTED PHYSICAL METRICS:")
        print("="*80)
        print(metrics_df.to_markdown(index=False)) # Prints a beautiful markdown table to the console
        print("="*80 + "\n")

        plot_multi_variable_pca(
            variables=all_variables,
            extent=self.extent,
            target=cfg_xai.get("target"),
            n_modes=1,          
            save_path=os.path.join(save_path, "pca_multi_variable.pdf"),
            cbar_label="PCA loading (normalised)",
        )

        self.logger.info(f"Multi-variable PCA figure saved to {save_path}")
        self.logger.info(f"Quantitative metrics saved to {csv_path}")
        return all_variables
        
    def _collect_explanations(self):
        """
        Single pass over batches — accumulates only what PCA needs.
        Never stores the full list of explanation maps simultaneously.
        """
        cfg_xai   = self.config.cfg_xai
        explainer_name = list(cfg_xai["explain"]["explainers"].keys())[0]
        explainer = self.get_explainer(explainer_name, cfg_xai["explain"]["explainers"][explainer_name])
        device    = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Accumulators — stay on CPU to spare GPU memory
        rows_abs  = []   # for barycenter/ellipse  (|attribution|, normalised)
        rows_pca  = []   # for PCA                 (signed,        normalised)
        extent = self.config.cfg_xai["extent"]
        target = self.config.cfg_xai["target"]
        for batch_idx, batch in enumerate(self.dataloader):
            H, W = batch.inputs.tensor.shape[2:4]
            lon0, lon1 = target[0], target[1]
            lat0, lat1 = target[2], target[3]
            lat_min, lat_max = extent[2], extent[3]
            lon_min, lon_max = extent[0], extent[1]
            lat_step = (lat_max - lat_min) / H
            lon_step = (lon_max - lon_min) / W
            i_min = int((lat0 - lat_min) / lat_step)
            i_max = int((lat1 - lat_min) / lat_step)
            j_min = int((lon0 - lon_min) / lon_step)
            j_max = int((lon1 - lon_min) / lon_step)

            output_idx = batch.outputs.feature_names_to_idx[
                cfg_xai["explain"]["output"]
            ]

            # ── Precipitation filter ───────────────────────
            precip_mean = batch.outputs.tensor[
                :, 0, i_min:i_max, j_min:j_max, output_idx
            ].mean().item()
            # if precip_mean < 1:
            #     continue
            batch.inputs.tensor  = batch.inputs.tensor.to(device)
            batch.forcing.tensor = batch.forcing.tensor.to(device)
            if batch.outputs is not None:
                batch.outputs.tensor = batch.outputs.tensor.to(device)

            with torch.autograd.set_grad_enabled(True):
                batch.inputs.tensor.requires_grad_()
                batch.forcing.tensor.requires_grad_()

                explanation = explainer.compute_explanations(
                    batch, batch_idx,
                    self.config.checkpoint,
                    self.config.cfg_model["model"],
                    self.config.cfg_dataset["data"],
                    cfg_xai,
                    self.dataset_info,
                    self.infer_ds,
                    cfg_xai["list_run_hour"],
                    cfg_xai["use_old_weights"],
                    cfg_xai["target"],
                    plot_explanations=False,
                    extent=cfg_xai["extent"],
                    fig_path=None,
                )


            if explanation is not None:
                # ── extract the flat map ───────────────────────────────────────
                if self.explainer_name == "WassersteinGrad":
                    raw = explanation[0].flatten()
                else:
                    raw = explanation[0][0].flatten()

                norm = raw.abs().max() + 1e-8

                # move to CPU immediately — free GPU allocation
                rows_abs.append((raw.abs() / norm).detach().cpu())
                rows_pca.append((raw       / norm).detach().cpu())

                # explicitly delete GPU tensors
                del explanation, raw, norm

            torch.cuda.empty_cache()

        X_matrix     = torch.stack(rows_abs, dim=0)   # (N, H*W)
        X_matrix_pca = torch.stack(rows_pca, dim=0)
        del rows_abs, rows_pca
        lat = np.arange(self.min_lat, self.max_lat + self.new_res, self.new_res)
        lon = np.arange(self.min_lon, self.max_lon + self.new_res, self.new_res)
        lon_flipped = lon[240:880]
        lat_flipped = lat[100:612]
        lon_flipped,lat_flipped= np.meshgrid(lon_flipped,lat_flipped)
        
        coords = torch.stack([
            torch.tensor(lon_flipped.copy()).flatten(),
            torch.tensor(lat_flipped.copy()).flatten()
        ], dim=0)
        return X_matrix, X_matrix_pca, coords
    
    
    def _compute_barycenter_ellipse(self, X_matrix: torch.Tensor, 
                                   coords: torch.Tensor) -> Tuple:
        """Compute weighted barycenter and covariance ellipse."""
        # Barycenter
        A_total = X_matrix.sum(dim=0)
        A_total_expanded = A_total.unsqueeze(0)
        Weighted_Pos = coords.to(device=A_total_expanded.device) * A_total_expanded
        
        Numerator = Weighted_Pos.sum(dim=1)
        Denominator = A_total.sum()
        Barycenter = Numerator / Denominator
        
        # Covariance matrix
        M = A_total.size(0)
        P_centered = coords.to(Barycenter.device) - Barycenter.unsqueeze(1)
        
        Numerator_Sum = torch.zeros(2, 2, device=Barycenter.device)
        for i in range(M):
            P_diff_i = P_centered[:, i].unsqueeze(1)
            prod = P_diff_i @ P_diff_i.T
            Numerator_Sum += A_total[i] * prod
        
        S_w = Numerator_Sum / (A_total.sum() - 1)
        
        # Eigen-decomposition
        evals, evecs = torch.linalg.eig(S_w)
        evals = evals.real
        evecs = evecs.real
        
        lambda_max = torch.max(evals)
        lambda_min = torch.min(evals)
        
        # 95% confidence ellipse
        g_squared = torch.tensor(5.991, device=Barycenter.device)
        a = torch.sqrt(lambda_max * g_squared).item()
        b = torch.sqrt(lambda_min * g_squared).item()
        
        # Rotation
        major_evec_index = torch.argmax(evals)
        major_evec = evecs[:, major_evec_index]
        minor_evec_index = torch.argmin(evals)
        minor_evec = evecs[:, minor_evec_index]
        
        evec_R = np.column_stack((major_evec.cpu(), minor_evec.cpu()))
        theta = torch.atan2(major_evec[1], major_evec[0]).item()
        
        ellipse_params = {
            "a": a, "b": b, "theta": theta,
            "evec_R": evec_R,
            "eigenvalues": (lambda_max.item(), lambda_min.item())
        }
        
        return Barycenter, ellipse_params
    
    def _perform_pca(self, X_matrix_pca: torch.Tensor) -> Dict:
        """Perform PCA on attribution maps."""
        # Center data
        mu = X_matrix_pca.mean(dim=0, keepdim=True)
        Xc = X_matrix_pca - mu
        
        # SVD
        U, S, Vt = torch.linalg.svd(Xc, full_matrices=False)
        
          
        # Compute eigenvalues (variances)
        N = Xc.shape[0]
        del U, Xc 
        eigenvalues = (S ** 2) / (N - 1)
        
        # Explained variance ratio
        evr = eigenvalues / eigenvalues.sum()
        
        # Get grid shape from dataset
        H, W = 512, 640 
        
        return {
            "pc_1": Vt[0].reshape(H, W),
            "pc_2": Vt[1].reshape(H, W),
            "explained_variance_ratio": evr.cpu().numpy(),
            "eigenvalues": eigenvalues.cpu().numpy()
        }
    
   

    def _visualize_pca(
        self,
        pca_results: Dict,
        barycenter: torch.Tensor,
        ellipse_params: Dict,
        coords: torch.Tensor,
        save_path: str,
        lon_mean, lon_std, lat_mean, lat_std,
        mode: str = "single",           # "single" | "both"
        variable_name: str = "",
    ):
        """
        Drop-in replacement for PCAAnalyzer._visualize_pca.

        mode = "single" → only PC1 + ellipse (ideal for posters / single-column figures)
        mode = "both"   → PC1 and PC2 side-by-side (for supplementary / full-width figures)
        """
        cfg_xai    = self.config.cfg_xai
        target     = cfg_xai.get("target")
        input_name = variable_name or cfg_xai["explain"]["input"]
        os.makedirs(save_path, exist_ok=True)

        common = dict(
            barycenter=barycenter,
            ellipse_params=ellipse_params,
            extent=self.extent,
            lon_mean=lon_mean, lon_std=lon_std,
            lat_mean=lat_mean, lat_std=lat_std,
            target=target,
            cbar_label="PCA loading (normalised)",
        )

        if mode == "single":
            plot_single_pca_mode(
                pc_map=pca_results["pc_1"],
                title=f"First PCA Mode — {input_name}",
                save_path=os.path.join(save_path, f"pca_pc1_{input_name}.pdf"),
                **common,
            )
        else:
            plot_two_pca_modes(
                pc1_map=pca_results["pc_1"],
                pc2_map=pca_results["pc_2"],
                explained_variance_ratio=pca_results["explained_variance_ratio"],
                variable_name=input_name,
                save_path=os.path.join(save_path, f"pca_{input_name}.pdf"),
                **common,
            )
    def _get_ellipse_coords(self, center: List, a: float, b: float, 
                        theta: float,lon_mean,lon_std,lat_mean,lat_std, N_points: int = 100, 
                        evec_R=None) -> Tuple[np.ndarray, np.ndarray]:
        """Generate ellipse coordinates."""
        t = np.linspace(0, 2 * np.pi, N_points)
        x_scaled = a * np.cos(t)
        y_scaled = b * np.sin(t)
        
        if evec_R is not None:
            R = evec_R
        else:
            R = np.array([
                [np.cos(theta), -np.sin(theta)],
                [np.sin(theta), np.cos(theta)]
            ])
        
        coords = np.dot(R, np.vstack([x_scaled, y_scaled]))
        lon_coords = coords[0, :] + center[0]
        lat_coords = coords[1, :] + center[1]
        lon_coords = (lon_coords * lon_std.cpu().numpy()) + lon_mean.cpu().numpy()
        lat_coords = (lat_coords * lat_std.cpu().numpy()) + lat_mean.cpu().numpy()
        return lon_coords, lat_coords