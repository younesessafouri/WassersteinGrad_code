from abc import ABC, abstractmethod
from py4castxai.utils_test.grad_utils import compute_grad

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import numpy as np
import torch
import os
import copy
from matplotlib.patches import Rectangle


# ─────────────────────────────────────────────
#  Helpers
# ─────────────────────────────────────────────

def normalize_expl(a: np.ndarray | torch.Tensor) -> torch.Tensor:
    """Normalize attribution map to [-1, 1]."""
    if isinstance(a, np.ndarray):
        a = torch.from_numpy(a).to(torch.float32)
    return a / (a.abs().max() + 1e-8)


def _denormalize(tensor: torch.Tensor, stats: dict, name: str) -> np.ndarray:
    """Reverse z-score normalization using stored mean/std."""
    mean = torch.as_tensor(stats[name]["mean"])
    std  = torch.as_tensor(stats[name]["std"])
    return (tensor * std + mean).numpy()


def _add_geo_features(ax):
    ax.add_feature(cfeature.COASTLINE, linewidth=0.6)
    ax.add_feature(cfeature.BORDERS,   linewidth=0.6)


def _add_target_box(axes, target):
    """Draw a bounding-box rectangle on every axis."""
    lon0, lon1, lat0, lat1 = target
    for ax in axes:
        rect = Rectangle(
            (lon0, lat0), lon1 - lon0, lat1 - lat0,
            linewidth=0.6, edgecolor="black", facecolor="none",
            transform=ccrs.PlateCarree(), zorder=5,
        )
        ax.add_patch(rect)


# ─────────────────────────────────────────────
#  Base class
# ─────────────────────────────────────────────

class XGrad(ABC):

    def __init__(self, model):
        self.model         = model
        self.compute_grad  = compute_grad

    @abstractmethod
    def compute_explanations(self, *args, **kwargs):
        pass

    # ------------------------------------------------------------------
    def plot_explanations(
        self,
        explanations,
        input,
        pred,
        extent,
        method,
        fig_path,
        runtime,
        stats,
        step         = 0,
        input_name   = "aro_u_250hpa",
        output_name  = "aro_tp_0m",
        input_idx    = 0,
        output_idx   = 0,
        target       = None,
        forcing      = None,
        save_figs    = True,
    ):
        # ── 1. Pull raw tensors ──────────────────────────────────────
        attr = normalize_expl(explanations.detach().cpu()).numpy()
        attr = attr.reshape(512,640)
        if forcing:
            inp_raw = input.forcing.tensor[0, step, :, :, input_idx].detach().cpu()
        elif step == 0:
            inp_raw = input.inputs.tensor[0, 0, :, :, input_idx].detach().cpu()
        else:
            inp_raw = pred.tensor[0, step - 1, :, :, input_idx].detach().cpu()

        pred_arr = pred.tensor[0, step, :, :, output_idx].detach().cpu().numpy()
        gt_raw   = input.outputs.tensor[0, step, :, :, output_idx].detach().cpu()

        # ── 2. Denormalize ───────────────────────────────────────────
        inp_arr = _denormalize(inp_raw, stats, input_name)
        gt_arr  = _denormalize(gt_raw,  stats, output_name)

        # ── 3. Shared colour limits for prediction panels ────────────
        vmin = min(gt_arr.min(), pred_arr.min())
        vmax = max(gt_arr.max(), pred_arr.max())

        # ── 4. Build figure ──────────────────────────────────────────
        fig, axes = plt.subplots(
            1, 4,
            figsize=(40, 16),
            subplot_kw={"projection": ccrs.PlateCarree()},
        )
        # fig.subplots_adjust(wspace=0.1, top=0.88)

        for ax in axes:
            _add_geo_features(ax)
        
        
        v = np.percentile(np.abs(attr), 99) + 1e-12
        vmin, vmax, cmap = -v, v, "RdBu_r"
        panel_cfg = [
            (axes[0], inp_arr,   "RdBu_r", None,  None,  f"Input  \n  {input_name}  (t)"),
            (axes[1], gt_arr,    "RdBu_r", vmin,  vmax,  f"Ground truth  \n  {output_name}  (t+{step+1})"),
            (axes[2], pred_arr,  "RdBu_r", vmin,  vmax,  f"Prediction  \n (t+{step+1})"),
            (axes[3], attr,      "RdBu_r", vmin,    vmax,     f"Attribution  \n {method}"),
        ]

        for ax, data, cmap, lo, hi, title in panel_cfg:
            kw = dict(origin="lower", cmap=cmap, extent=extent,
                      transform=ccrs.PlateCarree())
            if lo is not None:
                kw.update(vmin=lo, vmax=hi)
            im = ax.imshow(data, **kw)
            ax.set_title(title, fontsize=8, pad=5)
            cb = fig.colorbar(im, ax=ax, orientation="vertical",
                              fraction=0.046, pad=0.04)
            # if ax is axes[3]:
            #     cb.set_label("Normalised attribution", fontsize=9)

        # ── 5. Optional target bounding box ─────────────────────────
        if target is not None:
            _add_target_box(axes, target)   # expects (lon0, lon1, lat0, lat1)

     

        # ── 6. Save / show ───────────────────────────────────────────
        out_dir = os.path.join(fig_path, method)
        os.makedirs(out_dir, exist_ok=True)

        if save_figs:
            plt.savefig(os.path.join(out_dir, runtime), dpi=150, bbox_inches="tight")
            plt.close(fig)
        else:
            plt.show()