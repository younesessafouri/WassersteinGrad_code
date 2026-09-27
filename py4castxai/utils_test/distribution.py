import os
import logging
import copy

import numpy as np
import scipy.signal
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from matplotlib.colors import TwoSlopeNorm
from matplotlib.patches import Rectangle
import matplotlib.tri as tri

import torch
from sklearn.mixture import GaussianMixture
from typing import Dict, List, Optional, Tuple

LOG = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Plot style — applied globally once
# ---------------------------------------------------------------------------
plt.rcParams.update({
    "font.family": "sans-serif",
    "font.size": 10,
    "axes.titlesize": 11,
    "axes.titleweight": "bold",
    "axes.labelsize": 9,
     "figure.facecolor": "white",
    "axes.facecolor":   "white",
    "text.color":       "black",
    "axes.labelcolor":  "black",
    "xtick.color":      "black",
    "ytick.color":      "black",
    "axes.edgecolor":   "#cccccc",
    "figure.dpi": 120,
})

CMAP_ATTR = "RdBu_r"
CMAP_FFT  = "inferno"
NORM_ATTR = TwoSlopeNorm(vmin=-1, vcenter=0, vmax=1)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def normalize_expl(a: torch.Tensor | np.ndarray, abs: bool = False) -> torch.Tensor:
    """Normalise to [-1, 1] (or [0, 1] when abs=True)."""
    if isinstance(a, np.ndarray):
        a = torch.from_numpy(a).to(torch.float32)
    if abs:
        return a.abs() / (a.abs().max() + 1e-8)
    return a / (a.abs().max() + 1e-8)


def normalize_sum(a: torch.Tensor) -> torch.Tensor:
    """Normalise to a probability simplex (non-negative, sums to 1)."""
    a = torch.abs(a)
    return a / (a.sum() + 1e-8)


def fft_log_power(map_2d: np.ndarray) -> np.ndarray:
    """Return log-power spectrum of *map_2d* after Hann windowing."""
    H, W = map_2d.shape
    window_2d = np.outer(
        scipy.signal.windows.hann(H),
        scipy.signal.windows.hann(W),
    )
    F_shift = np.fft.fftshift(np.fft.fft2(map_2d * window_2d))
    return np.log(np.abs(F_shift) ** 2 + 1e-8)


def _add_geo_features(ax: plt.Axes) -> None:
    """Add coastlines and borders to a Cartopy axes."""
    ax.add_feature(cfeature.COASTLINE, linewidth=1.2, edgecolor="#aaaaaa")
    ax.add_feature(cfeature.BORDERS,   linewidth=0.8, edgecolor="#777777")


def _shared_colorbar(fig: plt.Figure, im, axes, label: str = "") -> None:
    cbar = fig.colorbar(im, ax=axes, fraction=0.025, pad=0.02)
    cbar.set_label(label, color="#e0e0e0")
    cbar.ax.yaxis.set_tick_params(color="#e0e0e0")


# ---------------------------------------------------------------------------
# Main class
# ---------------------------------------------------------------------------

class DistributionRunner:
    """Wasserstein barycenter + FFT analysis on XAI attribution maps."""

    def __init__(self, model, config, dataloader, dataset_info, infer_ds):
        self.model        = model
        self.config       = config
        self.dataloader   = dataloader
        self.dataset_info = dataset_info
        self.infer_ds     = infer_ds
        self.min_lat  = 37.5
        self.max_lat  = 55.4
        self.min_lon  = -12.0
        self.max_lon  = 16.0
        self.new_res  = 0.025
        self.extent   = self.config.cfg_xai["extent"]
        self.logger   = logging.getLogger(__name__)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def run(self):
        """Full pipeline: collect attributions → barycenter → FFT + geo plots."""
        import ot

        print("=" * 72)
        print("  DISTRIBUTION ANALYSIS")
        print("=" * 72)

        print("\n[1/2] Collecting explanations …")
        explanations= self._collect_explanations()
        print(f"      → {len(explanations)} samples collected")

        # coords, lon_mean, lon_std, lat_mean, lat_std = self._build_coords()

        print("\n[2/2] Computing barycenters …")
        for idx, (explanation) in enumerate((explanations)):
            print(f"  Sample {idx + 1}/{len(explanations)}")

           
         
           
            # --- plots ---
            self.plot_fft(explanation,"Gradient")

    # ------------------------------------------------------------------
    # Plots
    # ------------------------------------------------------------------
    def plot_fft(self, gradient: torch.Tensor | np.ndarray, title: str = "Gradient") -> None:
        """Plot the log-power FFT spectrum of a single 2D gradient map.

        Args:
            gradient: Tensor or array of shape (H, W), (1, H, W), or (1, 1, H, W, ...).
            title:    Label shown in the figure title.
        """
        # --- squeeze to 2D ---
        if isinstance(gradient, torch.Tensor):
            g = gradient.detach().cpu().numpy()
        else:
            g = np.array(gradient)
        g = np.squeeze(g)
        if g.ndim != 2:
            raise ValueError(f"Expected a 2-D map after squeezing, got shape {g.shape}")

        log_power = fft_log_power(g)

        fig, ax = plt.subplots(figsize=(4, 4))
        im = ax.imshow(log_power, cmap=CMAP_FFT)
        # ax.set_title(f"FFT Log-Power — {title}", pad=10)
        ax.axis("off")
        cbar = fig.colorbar(im, ax=ax, fraction=0.042, pad=0.04,shrink=0.6)
        cbar.set_label("log power", color="#e0e0e0")
        plt.tight_layout()
        plt.savefig("egu_poster/fft.pdf",dpi=300, bbox_inches="tight")
        plt.show()
    def _plot_fft_comparison(
        self,
        sg_map:       np.ndarray,
        weighted_map: np.ndarray,
        barycentre:   np.ndarray,
    ) -> None:
        """Three-panel log-power FFT comparison."""
        maps   = [sg_map, weighted_map, barycentre]
        titles = ["SmoothGrad", "Weighted SmoothGrad", "Wasserstein Barycenter"]
        ffts   = [fft_log_power(m) for m in maps]

        vmin = min(f.min() for f in ffts)
        vmax = max(f.max() for f in ffts)

        fig, axes = plt.subplots(1, 3, figsize=(16, 5))
        fig.suptitle("Fourier Log-Power Spectrum", fontsize=13, color="#e0e0e0", y=1.01)

        for ax, fft_map, title in zip(axes, ffts, titles):
            im = ax.imshow(fft_map, cmap=CMAP_FFT, vmin=vmin, vmax=vmax)
            ax.set_title(title, pad=8)
            ax.axis("off")

        _shared_colorbar(fig, im, axes, label="log power")
        plt.tight_layout()
        plt.show()

    def _plot_geo_comparison(
        self,
        sg_map:       np.ndarray,
        weighted_map: np.ndarray,
        barycentre:   np.ndarray,
    ) -> None:
        """Three-panel geographic attribution map."""
        maps   = [sg_map, weighted_map, barycentre]
        titles = ["Raw Attributions (SmoothGrad)",
                  "Weighted Attributions",
                  "Wasserstein Barycenter"]

        fig, axes = plt.subplots(
            1, 3, figsize=(15, 5),
            subplot_kw={"projection": ccrs.PlateCarree()},
        )
        fig.suptitle("Attribution Maps", fontsize=13, color="#e0e0e0", y=1.01)

        for ax, data, title in zip(axes, maps, titles):
            norm_data = normalize_expl(torch.tensor(data)).numpy()
            im = ax.imshow(
                norm_data,
                cmap=CMAP_ATTR,
                origin="lower",
                transform=ccrs.PlateCarree(),
                extent=self.extent,
                norm=NORM_ATTR,
            )
            _add_geo_features(ax)
            ax.set_title(title, pad=8)

        _shared_colorbar(fig, im, axes, label="Normalised attribution")
        plt.tight_layout()
        plt.show()

    def _plot_gmm_histogram(self, attribution_map: np.ndarray, save_path: str = None) -> None:
        """Attribution value histogram with GMM signal / noise decomposition."""
        vals = attribution_map.ravel()

        gmm = GaussianMixture(n_components=2, random_state=0).fit(vals.reshape(-1, 1))
        means        = gmm.means_.flatten()
        signal_comp  = int(np.argmax(np.abs(means)))
        labels       = gmm.predict(vals.reshape(-1, 1))

        mu     = gmm.means_.flatten()
        sigmas = np.sqrt(gmm.covariances_.flatten())
        noise  = int(np.argmin(np.abs(mu)))
        signal = 1 - noise
        sep    = abs(mu[signal] - mu[noise]) / sigmas[noise]
        print(f"  Signal/noise separation: {sep:.2f}σ")

        fig, axes = plt.subplots(1, 2, figsize=(11, 4.5), sharey=True)

        axes[0].hist(vals, bins=80, density=True, alpha=0.75, color="#5b8dee")
        for m in means:
            axes[0].axvline(m, linestyle="--", linewidth=1.5, color="#ff6b6b")
        axes[0].set_yscale("log")
        axes[0].set_title("Attribution Distribution")
        axes[0].set_xlabel("Attribution value")
        axes[0].set_ylabel("Density (log)")

        axes[1].hist(vals[labels != signal_comp], bins=80, density=True,
                     alpha=0.65, label="Noise", color="#aaaaaa")
        axes[1].hist(vals[labels == signal_comp], bins=80, density=True,
                     alpha=0.65, label="Signal", color="#f4a261")
        axes[1].set_yscale("log")
        axes[1].set_title("GMM Decomposition")
        axes[1].set_xlabel("Attribution value")
        axes[1].legend(framealpha=0.3)

        plt.suptitle("Attribution GMM Analysis", fontsize=12, color="#e0e0e0")
        plt.tight_layout()

        if save_path:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            plt.savefig(save_path, dpi=150, bbox_inches="tight")
            plt.close(fig)
        else:
            plt.show()

    def plot_gmm_fourier_comparison(
        self,
        attribution_map: torch.Tensor,
        save_path: str = None,
    ) -> None:
        """2×2 panel: spatial / frequency before and after GMM masking."""
        map_2d = attribution_map.detach().cpu().numpy()
        vals   = map_2d.ravel().reshape(-1, 1)

        gmm        = GaussianMixture(n_components=2, random_state=42).fit(vals)
        signal_idx = int(np.argmax(np.abs(gmm.means_.flatten())))
        mask_2d    = (gmm.predict(vals) == signal_idx).reshape(map_2d.shape)

        filtered_map         = map_2d.copy()
        filtered_map[~mask_2d] = 0.0

        orig_spec     = fft_log_power(map_2d)
        filtered_spec = fft_log_power(filtered_map)

        fig, axes = plt.subplots(2, 2, figsize=(10, 10))
        panel_data   = [map_2d, orig_spec, filtered_map, filtered_spec]
        panel_titles = [
            "Original (Spatial)",
            "Original (Frequency)",
            "GMM Filtered (Spatial)",
            "GMM Filtered (Frequency)",
        ]

        for ax, data, title in zip(axes.flat, panel_data, panel_titles):
            im = ax.imshow(data, cmap=CMAP_FFT)
            ax.set_title(title, pad=8)
            ax.axis("off")
            plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

        plt.suptitle("GMM Filtering vs. Frequency Content", fontsize=13, color="#e0e0e0")
        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches="tight")
            plt.close(fig)
        else:
            plt.show()

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _build_coords(self):
        """Build and normalise the (lon, lat) coordinate grid."""
        lat = np.arange(self.min_lat, self.max_lat + self.new_res, self.new_res)
        lon = np.arange(self.min_lon, self.max_lon + self.new_res, self.new_res)
        lon_grid, lat_grid = np.meshgrid(lon[240:880], lat[100:612])

        coords = torch.stack([
            torch.tensor(lon_grid.copy()).flatten(),
            torch.tensor(lat_grid.copy()).flatten(),
        ], dim=0)

        lon_mean, lon_std = coords[0].mean(), coords[0].std()
        lat_mean, lat_std = coords[1].mean(), coords[1].std()

        scale  = torch.tensor([lon_std.item(), lat_std.item()]).unsqueeze(1)
        origin = torch.tensor([lon_mean.item(), lat_mean.item()]).unsqueeze(1)
        coords = (coords - origin) / scale

        return coords, lon_mean, lon_std, lat_mean, lat_std

    def _collect_explanations(self) -> Tuple[List, List]:
        """Iterate the dataloader and collect (explanation, smoothgrad) pairs."""
        from py4castxai.explainers import (
            explainers_registry
        )

        cfg_xai = self.config.cfg_xai
        explainer_name = list(cfg_xai["explain"]["explainers"].keys())[0]
        expl_params    = cfg_xai["explain"]["explainers"][explainer_name]

       
        explainer = explainers_registry.get(explainer_name)(self.model, **expl_params)

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        explanations, sg_total = [], []

        for batch_idx, batch in enumerate(self.dataloader):
            batch.inputs.tensor  = batch.inputs.tensor.to(device)
            batch.forcing.tensor = batch.forcing.tensor.to(device)
            if batch.outputs is not None:
                batch.outputs.tensor = batch.outputs.tensor.to(device)

            with torch.autograd.set_grad_enabled(True):
                batch.inputs.tensor.requires_grad_()
                batch.forcing.tensor.requires_grad_()

                result = explainer.compute_explanations(
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
                    fig_path="./test_dddd",
                )

                if result is not None:
                    explanations.append(result[0])
                    # sg_total.append(result[1])

            torch.cuda.empty_cache()

        return explanations#, sg_total