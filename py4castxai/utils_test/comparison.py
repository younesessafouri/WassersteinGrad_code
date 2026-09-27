"""
Comparison Plots Module
=======================

Complete visualization functions for explainer and model comparisons.
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from matplotlib.patches import Rectangle
import torch
import copy
from typing import Dict, List, Optional, Tuple
from datetime import datetime


"""
Comparison Module for Py4CastXai
=================================

Complete module for comparing different explainer methods and models.
Handles data collection, statistical analysis, and orchestration.
"""

import os
import numpy as np
import torch
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.patches import Rectangle
import cartopy.crs as ccrs
import cartopy.feature as cfeature
 
# ── Publication style (NeurIPS-compliant) ────────────────────────────────────
matplotlib.rcParams.update({
    "font.family":        "serif",
    "font.serif":         ["Times New Roman", "DejaVu Serif"],
    "font.size":          9,
    "axes.titlesize":     9,
    "axes.titlepad":      5,
    "axes.labelsize":     8,
    "xtick.labelsize":    6,
    "ytick.labelsize":    6,
    "legend.fontsize":    7,
    "legend.framealpha":  0.85,
    "legend.edgecolor":   "0.7",
    "figure.dpi":         300,
    "savefig.dpi":        300,
    "savefig.bbox":       "tight",
    "savefig.pad_inches": 0.02,
    "axes.linewidth":     0.5,
    "xtick.major.width":  0.4,
    "ytick.major.width":  0.4,
})
 
_CMAP_FIELD   = "RdBu_r"
_CMAP_ATTR    = "RdBu_r"
_PANEL_LABELS = "abcdefghijklmnopqrstuvwxyz"
 
 
def _add_geo_features(ax):
    ax.add_feature(cfeature.COASTLINE, linewidth=0.5, edgecolor="0.3")
    ax.add_feature(cfeature.BORDERS,   linewidth=0.4, edgecolor="0.5",
                   linestyle="--")
 
 
def _colorbar(fig, im, ax, label=""):
    cb = fig.colorbar(im, ax=ax, orientation="vertical",
                      fraction=0.046, pad=0.03, shrink=0.88)
    cb.ax.tick_params(labelsize=6, width=0.4, length=2)
    if label:
        cb.set_label(label, fontsize=6)
    return cb
 
 
def _panel_label(ax, letter):
    ax.text(-0.01, 1.05, f"({letter})", transform=ax.transAxes,
            fontsize=9, fontweight="bold", va="bottom", ha="right")
 
 
def _add_target_box(axes, target, proj):
    if target is None:
        return
    lon0, lon1, lat0, lat1 = target
    for ax in axes:
        ax.add_patch(Rectangle(
            (lon0, lat0), lon1 - lon0, lat1 - lat0,
            linewidth=1.2, edgecolor="black", facecolor="none",
            transform=proj, zorder=5,
        ))


def plot_qualitative(
    explanations,          # dict[str, tensor]  — output of your loop
    input,                 # model input object
    pred,                  # model prediction object
    extent,                # (lon0, lon1, lat0, lat1)
    stats,                 # normalisation stats dict
    fig_path,              # root output directory
    runtime,               # filename stem
    input_name="aro_tp_0m",
    output_name="aro_tp_0m",
    input_idx=0,
    output_idx=0,
    target=None,           # optional bounding box (lon0,lon1,lat0,lat1)
    forcing=None,step = 1,
    save_figs=True,max_cols=4, panel_size=(2.6, 2.2)
):
    """
    Two-row qualitative figure for NeurIPS.

    Row 1 (fixed, 2 panels + empty spacers):
        Input field  |  Ground truth  |  Prediction  |  (Error map)

    Row 2 (dynamic, N panels — one per explainer):
        Attribution_1  |  Attribution_2  |  ...  |  Attribution_N
    """

    proj    = ccrs.PlateCarree()
    kw_base = dict(origin="lower", extent=extent, transform=proj)

    # ── Denormalise input fields ──────────────────────────────────────────────
    means_input  = torch.as_tensor(stats[input_name]["mean"])
    std_input    = torch.as_tensor(stats[input_name]["std"])
    means_output = torch.as_tensor(stats[output_name]["mean"])
    std_output   = torch.as_tensor(stats[output_name]["std"])


    inp       = (input * std_input  + means_input).numpy()
    # gt        = (input.outputs.tensor[0, step, :, :, output_idx]
    #              .detach().cpu() * std_output + means_output).numpy()
    pred_data = pred
    
    vmin_field =  pred_data.min()
    vmax_field = pred_data.max()
    sym_inp  = np.nanmax(np.abs(inp))
    sym_pred_max = pred_data.max()
    sym_pred_min = pred_data.min()
    # ── Attribution arrays ────────────────────────────────────────────────────
    method_names = list(explanations.keys())
    n_methods    = len(method_names)
 
    attrs = {}
    for name in method_names:
        a = explanations[name]
        if isinstance(a, torch.Tensor):
           
            a = a.reshape(512,640).detach().cpu().numpy()
        abs_max = np.nanmax(np.abs(a))
        attrs[name] = a / (abs_max + 1e-12)
 
    # ── Grid dimensions ───────────────────────────────────────────────────────
    n_top      = 2                              # Input + Prediction
    n_attr_cols = min(n_methods, max_cols)      # columns in attribution block
    n_attr_rows = int(np.ceil(n_methods / n_attr_cols))
 
    pw, ph = panel_size
    fig_w  = n_attr_cols * pw + 0.4            # width driven by bottom row
    fig_h  = ph + n_attr_rows * ph + 0.6       # top row + attr rows + margins
 
    fig = plt.figure(figsize=(fig_w, fig_h))
 
    # Two independent GridSpecs: top and bottom sections
    # Top section occupies upper ~45 % of figure height
    top_frac  = 1.0 / (1 + n_attr_rows)        # fraction for the top section
    attr_frac = 1.0 - top_frac
 
    # gs_top = gridspec.GridSpec(
    #     1, n_top,
    #     figure=fig,
    #     left=0.5 - (n_top * pw / 2) / fig_w,   # center the 2 panels
    #     right=0.5 + (n_top * pw / 2) / fig_w,
    #     top=0.97,
    #     bottom=0.97 - top_frac * 0.88,
    #     wspace=0.30,
    # )
 
    gs_attr = gridspec.GridSpec(
        n_attr_rows, n_attr_cols,
        figure=fig,
        left=0.03,
        right=0.97,
        top=0.97 - top_frac * 0.88 - 0.06,     # leave gap for separator line
        bottom=0.03,
        wspace=0.22,
        hspace=0.40,
    )
 
    # ── Top axes ──────────────────────────────────────────────────────────────
    # ax_inp  = fig.add_subplot(gs_top[0, 0], projection=proj)
    # ax_pred = fig.add_subplot(gs_top[0, 1], projection=proj)
    # top_axes = [ax_inp, ax_pred]
 
    # ── Attribution axes ──────────────────────────────────────────────────────
    attr_axes = []
    for k in range(n_methods):
        r, c = divmod(k, n_attr_cols)
        attr_axes.append(fig.add_subplot(gs_attr[r, c], projection=proj))
 
    # all_visible = top_axes + attr_axes
    all_visible = attr_axes
    for ax in all_visible:
        _add_geo_features(ax)
 
    # ── Row 1: field panels ───────────────────────────────────────────────────
    # top_cfg = [
    #     (ax_inp,  inp,       _CMAP_FIELD, None,  None,
    #      f"Input — {input_name}", "a"),
    #     (ax_pred, pred_data, _CMAP_FIELD, sym_pred_min, sym_pred_max,
    #      f"Prediction (t+ {step}) — {output_name}",     "b"),
    # ]
    # for ax, data, cmap, lo, hi, title, letter in top_cfg:
    #     im = ax.imshow(data, cmap=cmap, vmin=lo, vmax=hi, **kw_base)
    #     ax.set_title(title, fontsize=9)
    #     _colorbar(fig, im, ax)
    #     _panel_label(ax, letter)
 
    # ── Row 2+: attribution panels ────────────────────────────────────────────
    DISPLAY_NAMES = {
    "BaseGrad":            "BaseGrad",
    "InputxGrad":          "Input$\\times$Grad",
    "SmoothGrad":          "SmoothGrad",
    "SmoothGrad2":         "SmoothGrad$^2$",
    "VarGrad":             "VarGrad",
    "WassersteinGrad":     "WG$_{\\mathrm{Bary}}$ (Ours)",
    "WassersteinGradMask": "WG$_{\\mathrm{Bary}\\times\\mathrm{Grad}}$ (Ours)",
}

    for j, (name, ax) in enumerate(zip(method_names, attr_axes)):
        display = DISPLAY_NAMES.get(name, name)

        im = ax.imshow(attrs[name], cmap=_CMAP_ATTR, vmin=-1, vmax=1, **kw_base)
        # display = name.replace("_", " ").title()
        # ax.set_title(display, fontsize=10)
        _colorbar(fig, im, ax)
        # _panel_label(ax, _PANEL_LABELS[j])
 
    # ── Target bounding box ───────────────────────────────────────────────────
    _add_target_box(all_visible, target, proj)
 
    # ── Separator line between rows ───────────────────────────────────────────
    # sep_y = gs_attr.top + 0.025
    # fig.add_artist(plt.Line2D(
    #     [0.01, 0.99], [sep_y, sep_y],
    #     transform=fig.transFigure,
    #     color="0.6", linewidth=0.6, linestyle="--",
    # ))
 
    # # ── Section labels ────────────────────────────────────────────────────────
    # fig.text(0.01, gs_top.top + 0.01,   "Meteorological fields",
    #          fontsize=7, fontstyle="italic", color="0.45", va="bottom")
    # fig.text(0.01, gs_attr.top + 0.01, "Attribution methods",
    #          fontsize=7, fontstyle="italic", color="0.45", va="bottom")
 
    # # ── Overall title ─────────────────────────────────────────────────────────
    # fig.suptitle(
    #     f"{input_name} $\\rightarrow$ {output_name} — qualitative comparison",
    #     fontsize=10, fontweight="bold", y=1.01,
    # )
 
    # ── Save ──────────────────────────────────────────────────────────────────
    out_dir = os.path.join(fig_path, "qualitative")
    os.makedirs(out_dir, exist_ok=True)
 
    if save_figs:
        base = os.path.join(out_dir, runtime)
        fig.savefig(base + ".png", dpi=300, bbox_inches="tight", pad_inches=0.02)
        fig.savefig(base + ".pdf",          bbox_inches="tight", pad_inches=0.02)
        plt.close(fig)
        print(f"Saved → {base}.png / .pdf")
    else:
        plt.show()
 



def plot_qualitative_multidates(
    all_explanations,      # list of dicts: [{method: tensor, ...}, ...]
    all_inputs,            # list of tensors
    all_preds,             # list of tensors
    all_runtimes,          # list of str  e.g. ["2023010112", ...]
    extent,
    stats,
    fig_path,
    input_name="aro_u_250hpa",
    output_name="aro_tp_0m",
    input_idx=0,
    output_idx=0,
    target=None,
    panel_size=(2.6, 2.2),
    save_figs=True,
    filename="multidates_qualitative",
):
    """
    Appendix figure: one row per date, one column per attribution method.
    
    Layout:
        Col 0        | Col 1      | Col 2      | ... | Col N
        Date label   | Method 1   | Method 2   | ... | Method N
        (text only)  | attr map   | attr map   | ... | attr map
    
    Each row is labelled on the left with the runtime date string.
    """

    proj    = ccrs.PlateCarree()
    kw_base = dict(origin="lower", extent=extent, transform=proj)

    means_output = torch.as_tensor(stats[output_name]["mean"])
    std_output   = torch.as_tensor(stats[output_name]["std"])
    means_input  = torch.as_tensor(stats[input_name]["mean"])
    std_input    = torch.as_tensor(stats[input_name]["std"])

    # ── Collect method names from first sample ────────────────────────────────
    method_names = list(all_explanations[0].keys())
    n_methods    = len(method_names)
    n_dates      = len(all_runtimes)

    DISPLAY_NAMES = {
        "BaseGrad":            "BaseGrad",
        "InputxGrad":          "Input$\\times$Grad",
        "SmoothGrad":          "SmoothGrad",
        "SmoothGrad2":         "SmoothGrad$^2$",
        "VarGrad":             "VarGrad",
        "WassersteinGrad":     "WG$_{\\mathrm{Bary}}$ (Ours)",
        "WassersteinGradMask": "WG$_{\\mathrm{Bary}\\times\\mathrm{Grad}}$ (Ours)",
    }

    # ── Figure dimensions ─────────────────────────────────────────────────────
    pw, ph   = panel_size
    date_col = 0.85                         # width of the date label column
    fig_w    = date_col + n_methods * pw + 0.4
    fig_h    = n_dates  * ph + 0.3         # rows + top margin for col headers

    fig = plt.figure(figsize=(fig_w, fig_h))

    # One GridSpec: n_dates rows x n_methods cols
    # Leave left margin for date labels
    gs = gridspec.GridSpec(
        n_dates, n_methods,
        figure=fig,
        left=date_col / fig_w + 0.01,
        right=0.97,
        top=0.93,
        bottom=0.02,
        wspace=0.18,
        hspace=0.15,
    )

    # ── Column headers (method names) — drawn once at the top ─────────────────
    for j, name in enumerate(method_names):
        display = DISPLAY_NAMES.get(name, name)
        # Compute x-center of each column in figure coordinates
        col_left  = gs.left  + j       * (gs.right - gs.left) / n_methods
        col_right = gs.left  + (j + 1) * (gs.right - gs.left) / n_methods
        x_center  = (col_left + col_right) / 2
        fig.text(
            x_center, 0.96,
            display,
            ha="center", va="bottom",
            fontsize=9, fontweight="bold",
        )

    # ── Row loop ──────────────────────────────────────────────────────────────
    for i, (explanations, inp_tensor, pred_tensor, runtime) in enumerate(
        zip(all_explanations, all_inputs, all_preds, all_runtimes)
    ):
        # ── Normalise attributions for this date ──────────────────────────────
        attrs = {}
        for name in method_names:
            a = explanations[name]
            if isinstance(a, torch.Tensor):
                a = a.reshape(512, 640).detach().cpu().numpy()
            abs_max = np.nanmax(np.abs(a))
            attrs[name] = a / (abs_max + 1e-12)

        # ── Date label on the left ────────────────────────────────────────────
        # Parse runtime string "YYYYMMDDHH" → readable
        try:
            dt = datetime.strptime(runtime, "%Y%m%d%H")
            date_str = dt.strftime("%d %b %Y\n%H UTC")
        except ValueError:
            date_str = runtime

        # Compute y-center of this row in figure coordinates
        row_top    = gs.top  - i       * (gs.top - gs.bottom) / n_dates
        row_bottom = gs.top  - (i + 1) * (gs.top - gs.bottom) / n_dates
        y_center   = (row_top + row_bottom) / 2

        fig.text(
            date_col / fig_w - 0.01,
            y_center,
            date_str,
            ha="right", va="center",
            fontsize=8,
            color="0.2",
            linespacing=1.4,
        )

        # ── Attribution panels for this row ───────────────────────────────────
        for j, name in enumerate(method_names):
            ax = fig.add_subplot(gs[i, j], projection=proj)
            _add_geo_features(ax)

            im = ax.imshow(
                attrs[name],
                cmap=_CMAP_ATTR,
                vmin=-1, vmax=1,
                **kw_base,
            )
            _colorbar(fig, im, ax)

            # Only label first row panels (headers already drawn above)
            # Draw bounding box target
            if target is not None:
                _add_target_box([ax], target, proj)

    # ── Row index labels (a), (b), ... on the left of each row ───────────────
    
    # for i in range(n_dates):
    #     row_top    = gs.top  - i       * (gs.top - gs.bottom) / n_dates
    #     row_bottom = gs.top  - (i + 1) * (gs.top - gs.bottom) / n_dates
    #     y_center   = (row_top + row_bottom) / 2

    #     try:
    #         dt = datetime.strptime(all_runtimes[i], "%Y%m%d%H")
    #         date_str = dt.strftime("%d %b %Y\n%H UTC")
    #     except ValueError:
    #         date_str = all_runtimes[i]

    #     # Panel letter ABOVE the date string — stacked vertically, not side by side
    #     fig.text(
    #         date_col / fig_w - 0.01,
    #         y_center + 0.02,              # slightly above center
    #         f"({_PANEL_LABELS[i]})",
    #         ha="right", va="bottom",
    #         fontsize=8, fontweight="bold",
    #         color="0.1",
    #     )
    #     fig.text(
    #         date_col / fig_w - 0.01,
    #         y_center - 0.02,              # slightly below center
    #         date_str,
    #         ha="right", va="top",
    #         fontsize=7,
    #         color="0.35",
    #         linespacing=1.4,
    #     )
    # ── Save ──────────────────────────────────────────────────────────────────
    out_dir = os.path.join(fig_path, "qualitative_multidates")
    os.makedirs(out_dir, exist_ok=True)

    if save_figs:
        base = os.path.join(out_dir, filename)
        fig.savefig(base + ".png", dpi=300,
                    bbox_inches="tight", pad_inches=0.02)
        fig.savefig(base + ".pdf",
                    bbox_inches="tight", pad_inches=0.02)
        plt.close(fig)
        print(f"Saved → {base}.png / .pdf")
    else:
        plt.show()










# ============================================================================
# EXPLAINER COMPARATOR
# ============================================================================

class ExplainerComparator:
    """Compares multiple explainers on the same examples."""
    
    def __init__(self, model, config, dataloader, dataset_info, infer_ds):
        self.model = model
        self.config = config
        self.dataloader = dataloader
        self.dataset_info = dataset_info
        self.infer_ds = infer_ds
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.logger = logging.getLogger(__name__)
        
    def _get_explainer(self, explainer_name: str, params: Dict):
        """Initialize explainer by name."""
        explainer_map = {
            "SmoothGrad": SmoothGrad,
            "SmoothGradSquared": SmoothGrad,  # Assuming variant
            "InputxGrad": InputxGrad,
            "BaseGrad": BaseGrad
        }
        
        explainer_cls = explainer_map.get(explainer_name, BaseGrad)
        if explainer_cls is None:
            raise ValueError(f"Unknown explainer: {explainer_name}")
        
        return explainer_cls(self.model, **params)
    
    def collect_examples(self, num_examples: int = 4) -> List[Dict]:
        """
        Collect explanations from multiple explainers for comparison.
        
        Args:
            num_examples: Number of examples to collect
            
        Returns:
            List of dicts containing batch, predictions, and attributions
        """
        cfg_xai = self.config.cfg_xai
        results_all = []
        examples_collected = 0
        
        self.logger.info(f"Collecting {num_examples} examples for comparison")
        
        for batch_idx, batch in enumerate(self.dataloader):
            if examples_collected >= num_examples:
                break
            
            # Prepare batch
            batch.inputs.tensor = batch.inputs.tensor.to(self.device)
            if batch.outputs is not None:
                batch.outputs.tensor = batch.outputs.tensor.to(self.device)
            batch.forcing.tensor = batch.forcing.tensor.to(self.device)
            
            example_results = {"batch": batch}
            success = True
            
            # Compute explanations for each explainer
            for explainer_name, expl_params in cfg_xai["compare"]["explainers"].items():
                self.logger.debug(f"Computing {explainer_name} for example {examples_collected + 1}")
                
                explainer = self._get_explainer(explainer_name, expl_params)
                
                with torch.autograd.set_grad_enabled(True):
                    batch.inputs.tensor.requires_grad_()
                    batch.forcing.tensor.requires_grad_()
                    
                    try:
                        attrs, pred = explainer.compute_explanations(
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
                            return_output=True
                        )
                        
                        if attrs is None:
                            success = False
                            break
                        
                        example_results[explainer_name] = attrs[0].detach().cpu()
                        example_results["pred"] = pred.tensor.detach().cpu()
                        
                    except Exception as e:
                        self.logger.error(f"Error computing {explainer_name}: {e}")
                        success = False
                        break
            
            if success:
                results_all.append(example_results)
                examples_collected += 1
                self.logger.info(f"Collected example {examples_collected}/{num_examples}")
            
            torch.cuda.empty_cache()
        
        if examples_collected < num_examples:
            self.logger.warning(f"Only collected {examples_collected}/{num_examples} examples")
        
        return results_all


# ============================================================================
# COMPARISON RUNNER
# ============================================================================

class ComparisonRunner:
    """Orchestrates comparison visualization and analysis."""
    
    def __init__(self, model, config, dataloader, dataset_info, infer_ds):
        self.model = model
        self.config = config
        self.dataloader = dataloader
        self.dataset_info = dataset_info
        self.infer_ds = infer_ds
        self.logger = logging.getLogger(__name__)
        
    def run(self, num_examples: int = 4):
        """
        Run complete comparison analysis.
        
        Args:
            num_examples: Number of examples to compare
            
        Returns:
            Dict containing results and analysis
        """
        comparator = ExplainerComparator(
            self.model, self.config, 
            self.dataloader, self.dataset_info, self.infer_ds
        )
        
        # Collect examples
        self.logger.info(f"Starting comparison with {num_examples} examples")
        results = comparator.collect_examples(num_examples)
        
        if not results:
            self.logger.warning("No examples collected for comparison")
            return None
        
        # Perform statistical analysis
        self.logger.info("Performing statistical analysis")
        stats = self._analyze_results(results)
        
        # Generate visualizations
        cfg_xai = self.config.cfg_xai
        save_path = cfg_xai["explain"]["exp_plot"]["fig_path"]
        
        if cfg_xai["explain"]["exp_plot"].get("plot_explanations", True):
            self._generate_visualizations(results, save_path)
        
        # Print summary
        self._print_summary(stats)
        
        return {
            "results": results,
            "statistics": stats
        }
    
    def _analyze_results(self, results: List[Dict]) -> Dict:
        """Perform statistical analysis on collected results."""
        analyzer = ExplanationAnalyzer()
        stats = {}
        
        cfg_xai = self.config.cfg_xai
        input_name = cfg_xai["explain"]["input"]
        explainer_names = list(cfg_xai["compare"]["explainers"].keys())
        
        # Get input index
        batch = results[0]["batch"]
        if cfg_xai["explain"].get("forcing_input", False):
            input_idx = batch.forcing.feature_names_to_idx[input_name]
        else:
            input_idx = batch.inputs.feature_names_to_idx[input_name]
        
        # Collect all attributions
        all_attrs = {name: [] for name in explainer_names}
        
        for example in results:
            for explainer_name in explainer_names:
                attr = example[explainer_name][0].squeeze(1)
                
                if cfg_xai["explain"].get("forcing_input", False):
                    attr = attr[..., input_idx]
                else:
                    attr = attr[0, ..., input_idx]
                
                all_attrs[explainer_name].append(attr)
        
        # Compute statistics per explainer
        for explainer_name in explainer_names:
            attrs = all_attrs[explainer_name]
            
            sparsity_vals = [analyzer.compute_sparsity(a) for a in attrs]
            concentration_vals = [analyzer.compute_concentration(a) for a in attrs]
            max_vals = [a.abs().max().item() for a in attrs]
            mean_abs_vals = [a.abs().mean().item() for a in attrs]
            
            stats[explainer_name] = {
                "sparsity": {
                    "mean": np.mean(sparsity_vals),
                    "std": np.std(sparsity_vals),
                    "values": sparsity_vals
                },
                "concentration": {
                    "mean": np.mean(concentration_vals),
                    "std": np.std(concentration_vals),
                    "values": concentration_vals
                },
                "max_attribution": {
                    "mean": np.mean(max_vals),
                    "std": np.std(max_vals),
                    "values": max_vals
                },
                "mean_abs_attribution": {
                    "mean": np.mean(mean_abs_vals),
                    "std": np.std(mean_abs_vals),
                    "values": mean_abs_vals
                }
            }
        
        # Compute pairwise similarities
        stats["pairwise_similarities"] = self._compute_pairwise_similarities(
            all_attrs, analyzer
        )
        
        return stats
    
    def _compute_pairwise_similarities(self, all_attrs: Dict, 
                                      analyzer: 'ExplanationAnalyzer') -> Dict:
        """Compute pairwise similarities between explainers."""
        explainer_names = list(all_attrs.keys())
        similarities = {}
        
        for i, name1 in enumerate(explainer_names):
            for name2 in explainer_names[i+1:]:
                pair_key = f"{name1}_vs_{name2}"
                
                # Compute similarity for each example
                sim_vals = []
                for attr1, attr2 in zip(all_attrs[name1], all_attrs[name2]):
                    sim = analyzer.compute_similarity(attr1, attr2, method="cosine")
                    sim_vals.append(sim)
                
                similarities[pair_key] = {
                    "mean": np.mean(sim_vals),
                    "std": np.std(sim_vals),
                    "values": sim_vals
                }
        
        return similarities
    
    def _generate_visualizations(self, results: List[Dict], save_path: str):
        """Generate all comparison visualizations."""
      
        cfg_xai = self.config.cfg_xai
        
        self.logger.info("Generating comparison plots")
        
        # Main comparison plot
        plot_explainer_comparison(
            results,
            self.dataset_info,
            cfg_xai,
            save_path=save_path
        )
        
        # # Grid plot
        # plot_explainer_grid(
        #     results,
        #     self.dataset_info,
        #     cfg_xai,
        #     save_path=save_path,
        #     max_examples=min(len(results), 8)
        # )
        
        self.logger.info(f"Visualizations saved to {save_path}")
    
    def _print_summary(self, stats: Dict):
        """Print summary of comparison results."""
        self.logger.info("\n" + "="*80)
        self.logger.info("COMPARISON SUMMARY")
        self.logger.info("="*80)
        
        explainer_names = [k for k in stats.keys() if k != "pairwise_similarities"]
        
        for explainer_name in explainer_names:
            self.logger.info(f"\n{explainer_name}:")
            self.logger.info("-" * 40)
            
            for metric_name, metric_data in stats[explainer_name].items():
                self.logger.info(
                    f"  {metric_name}: {metric_data['mean']:.4f} ± {metric_data['std']:.4f}"
                )
        
        self.logger.info("\n" + "-"*40)
        self.logger.info("Pairwise Similarities (Cosine):")
        self.logger.info("-" * 40)
        
        for pair, sim_data in stats["pairwise_similarities"].items():
            self.logger.info(
                f"  {pair}: {sim_data['mean']:.4f} ± {sim_data['std']:.4f}"
            )
        
        self.logger.info("="*80 + "\n")


















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


def add_target_box(ax, target, projection=ccrs.PlateCarree(), color='black', linewidth=2):
    """Add target region box to axis."""
    if target is None:
        return
    
    lon0, lon1, lat0, lat1 = target
    rect = Rectangle(
        (lon0, lat0), lon1 - lon0, lat1 - lat0,
        linewidth=linewidth, edgecolor=color, facecolor='none',
        transform=projection, zorder=5
    )
    ax.add_patch(rect)


# ============================================================================
# MAIN COMPARISON PLOT
# ============================================================================

def plot_explainer_comparison(results_all: List[Dict], dataset_info, 
                              cfg_xai: Dict, save_path: str):
    """
    Plot side-by-side comparison of different explainers on multiple examples.
    
    Args:
        results_all: List of dicts, each containing:
            - 'batch': The input batch
            - 'pred': Model predictions
            - '<explainer_name>': Attribution maps for each explainer
        dataset_info: Dataset information including stats
        cfg_xai: Configuration dictionary
        save_path: Base path to save figures
    """
    num_examples = len(results_all)
    explainer_names = [k for k in results_all[0].keys() 
                      if k not in ['batch', 'pred']]
    
    extent = cfg_xai["extent"]
    stats = dataset_info.stats
    input_name = cfg_xai["explain"]["input"]
    output_name = cfg_xai["explain"]["output"]
    
    # Determine if forcing input
    batch = results_all[0]["batch"]
    if cfg_xai["explain"].get("forcing_input", False):
        input_idx = batch.forcing.feature_names_to_idx[input_name]
        means_input = torch.asarray(stats[input_name]["mean"])
        std_input = torch.asarray(stats[input_name]["std"])
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
    if n_rows == 1:
        axes = axes.reshape(1, -1)
    
    target = cfg_xai.get("target")
    
    # Plot each example
    for c, example_results in enumerate(results_all):
        batch = example_results["batch"]
        
        # Get input
        if cfg_xai["explain"].get("forcing_input", False):
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
            axes[0, c].set_ylabel("Input (t)", fontsize=11, fontweight='bold')
        axes[0, c].set_title(f"Example {c+1}", fontsize=12, fontweight='bold')
        
        # Row 2: Prediction
        im1 = axes[1, c].imshow(pred, origin="lower", cmap="RdBu_r", extent=extent)
        if c == 0:
            axes[1, c].set_ylabel("Prediction (t+1)", fontsize=11, fontweight='bold')
        
        # Rows 3+: Explainers
        for r, explainer_name in enumerate(explainer_names, start=2):
            attr = example_results[explainer_name][0].squeeze(1)
            
            if cfg_xai["explain"].get("forcing_input", False):
                attr = attr[..., input_idx].numpy()
            else:
                attr = attr[0, ..., input_idx].numpy()
            
            # Normalize attribution
            attr = normalize_attr(attr)
            
            im = axes[r, c].imshow(
                attr, origin="lower", cmap="RdBu_r", 
                extent=extent, vmin=-1, vmax=1
            )
            
            if c == 0:
                axes[r, c].set_ylabel(explainer_name, fontsize=11, fontweight='bold')
            
            if c == n_cols - 1:
                plt.colorbar(im, ax=axes[r, c], orientation="vertical", 
                           fraction=0.046, pad=0.04)
        
        # Add colorbars for first two rows (last column only)
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
            ax.add_feature(cfeature.BORDERS, linewidth = 0.6)

            ax.set_xticks([])
            ax.set_yticks([])
            
            add_target_box(ax, target)
    
    plt.subplots_adjust(wspace=0.05, hspace=0.15)
    plt.suptitle(f"Explainer Comparison: {input_name} → {output_name}", 
                fontsize=16, fontweight='bold', y=0.998)
    
    # Save
    fig_path = os.path.join(save_path, "multiple_examples_comparison")
    os.makedirs(fig_path, exist_ok=True)
    plt.savefig(
        os.path.join(fig_path, f"{input_name}_{output_name}.png"),
        dpi=300, bbox_inches="tight"
    )
    plt.close()


# ============================================================================
# MODEL COMPARISON PLOT
# ============================================================================

def plot_model_comparison(results_all: List[Dict], dataset_info,
                         cfg_xai: Dict, model_names: List[str],
                         save_path: str, num_examples: int = 4):
    """
    Plot comparison of explanations across different models.
    
    Args:
        results_all: List of dicts containing results for each example
        dataset_info: Dataset information
        cfg_xai: Configuration dictionary
        model_names: List of model names
        save_path: Path to save figures
        num_examples: Number of examples to display
    """
    if not results_all:
        print("No results to plot")
        return
    
    num_examples = min(num_examples, len(results_all))
    explainer_names = list(cfg_xai["compare"]["explainers"].keys())
    
    extent = cfg_xai["extent"]
    stats = dataset_info.stats
    input_name = cfg_xai["explain"]["input"]
    output_name = cfg_xai["explain"]["output"]
    
    # Get input index
    batch = results_all[0]["batch"]
    if cfg_xai["explain"].get("forcing_input", False):
        input_idx = batch.forcing.feature_names_to_idx[input_name]
    else:
        input_idx = batch.inputs.feature_names_to_idx[input_name]
    
    means_input = torch.asarray(stats[input_name]["mean"])
    std_input = torch.asarray(stats[input_name]["std"])
    means_output = torch.asarray(stats[output_name]["mean"])
    std_output = torch.asarray(stats[output_name]["std"])
    
    # Calculate rows: 1 input + (1 pred + explainers) per model
    rows_per_model = 1 + len(explainer_names)
    n_rows = 1 + (rows_per_model * len(model_names))
    n_cols = num_examples
    
    fig, axes = plt.subplots(
        n_rows, n_cols,
        figsize=(4.5 * n_cols, 2.5 * n_rows),
        subplot_kw={'projection': ccrs.PlateCarree()}
    )
    
    # Ensure axes is 2D
    if n_cols == 1:
        axes = axes.reshape(-1, 1)
    if n_rows == 1:
        axes = axes.reshape(1, -1)
    
    target = cfg_xai.get("target")
    
    # Plot each example
    for c, example_results in enumerate(results_all[:num_examples]):
        batch = example_results["batch"]
        
        # Get input (same for all models)
        if cfg_xai["explain"].get("forcing_input", False):
            inp = batch.forcing.tensor[0, 0, :, :, input_idx].detach().cpu()
        else:
            inp = batch.inputs.tensor[0, 0, :, :, input_idx].detach().cpu()
        
        inp = inp * std_input + means_input
        
        # Row 0: Input (shared)
        im0 = axes[0, c].imshow(inp, origin="lower", cmap="RdBu_r", extent=extent)
        if c == 0:
            axes[0, c].set_ylabel("Input (t)", fontsize=11, fontweight='bold')
        axes[0, c].set_title(f"Example {c+1}", fontsize=12, fontweight='bold')
        
        if c == n_cols - 1:
            plt.colorbar(im0, ax=axes[0, c], orientation="vertical",
                        fraction=0.046, pad=0.04)
        
        # For each model
        current_row = 1
        for model_idx, model_name in enumerate(model_names):
            # Prediction
            pred_key = f"pred_{model_name}"
            pred = example_results[pred_key][0, 0, :, :, 0]
            pred = pred * std_output + means_output
            
            im_pred = axes[current_row, c].imshow(
                pred, origin="lower", cmap="RdBu_r", extent=extent
            )
            
            if c == 0:
                axes[current_row, c].set_ylabel(
                    f"Pred ({model_name})", 
                    fontsize=10, fontweight='bold'
                )
            
            if c == n_cols - 1:
                plt.colorbar(im_pred, ax=axes[current_row, c],
                           orientation="vertical", fraction=0.046, pad=0.04)
            
            current_row += 1
            
            # Explainers for this model
            for explainer_name in explainer_names:
                attr_key = f"{explainer_name}_{model_name}"
                attr = example_results[attr_key][0].squeeze(1)
                
                if cfg_xai["explain"].get("forcing_input", False):
                    attr = attr[..., input_idx].numpy()
                else:
                    attr = attr[0, ..., input_idx].numpy()
                
                attr = normalize_attr(attr)
                
                im_attr = axes[current_row, c].imshow(
                    attr, origin="lower", cmap="RdBu_r",
                    extent=extent, vmin=-1, vmax=1
                )
                
                if c == 0:
                    axes[current_row, c].set_ylabel(
                        f"{explainer_name}\n({model_name})",
                        fontsize=9, fontweight='bold'
                    )
                
                if c == n_cols - 1:
                    plt.colorbar(im_attr, ax=axes[current_row, c],
                               orientation="vertical", fraction=0.046, pad=0.04)
                
                current_row += 1
    
    # Add features and target box to all axes
    for r in range(n_rows):
        for c in range(n_cols):
            ax = axes[r, c]
            ax.set_extent(extent, crs=ccrs.PlateCarree())
            ax.add_feature(cfeature.COASTLINE, linewidth=0.3)
            ax.add_feature(cfeature.BORDERS, linewidth = 0.6)

            ax.set_xticks([])
            ax.set_yticks([])
            
            add_target_box(ax, target)
    
    plt.subplots_adjust(wspace=0.05, hspace=0.1)
    
    model_names_str = "_vs_".join(model_names)
    plt.suptitle(
        f"Model Comparison: {model_names_str}\n{input_name} → {output_name}",
        fontsize=14, fontweight='bold', y=0.998
    )
    
    # Save
    fig_path = os.path.join(save_path, "model_comparison")
    os.makedirs(fig_path, exist_ok=True)
    plt.savefig(
        os.path.join(fig_path, f"{model_names_str}_{input_name}_{output_name}.png"),
        dpi=300, bbox_inches="tight"
    )
    plt.close()


# ============================================================================
# MULTI-LEVEL ATTRIBUTION PLOT
# ============================================================================

def plot_diff_levels_zoomed(
    explanations: Dict, batch, output, extent: List, method: str,
    fig_path: str, runtime: str, stats: Dict, step: int,
    output_name: str, target: List, forcing: Optional[bool],
    save_figs: bool, input_names: List[str],
    zoom_region: Optional[List] = [2.1-3.2,2.4+3.2,48.6-3,48.9+3]  # [lon_min, lon_max, lat_min, lat_max]
):
    """
    Plot attribution maps for different input channels/levels side-by-side.

    Args:
        explanations:  Dict mapping input_name -> list of attribution tensors (one per step)
        batch:         Input batch
        output:        Model output
        extent:        Spatial extent [lon_min, lon_max, lat_min, lat_max]
        method:        Explainer method name (used for sub-folder)
        fig_path:      Root path to save figure
        runtime:       Runtime string for labeling (e.g. "2022010600")
        stats:         Statistics dict for denormalization
        step:          Time step index (0 = analysis, >0 = forecast)
        output_name:   Name of output variable
        target:        Target region [lon0, lon1, lat0, lat1] or None
        forcing:       Whether using forcing inputs
        save_figs:     Whether to save figure to disk
        input_names:   List of input variable names to plot
        zoom_region:   Optional zoom region [lon_min, lon_max, lat_min, lat_max].
                       If None, no zoom row is added.
    """
    n_inputs = len(input_names)
    has_zoom = zoom_region is not None
    n_rows = 2 if has_zoom else 2

    projection = ccrs.PlateCarree()

    # ------------------------------------------------------------------ #
    # Figure / gridspec
    # ------------------------------------------------------------------ #
    fig_width = 4.8 * n_inputs
    fig_height = 4.0 * n_rows + 0.8          # ~0.8 for suptitle breathing room
    fig = plt.figure(figsize=(fig_width, fig_height))

    height_ratios = [1] * n_rows
    gs = fig.add_gridspec(
        n_rows, n_inputs,
        height_ratios=height_ratios,
        hspace=0.30,
        wspace=0.18,
        left=0.04, right=0.97,
        top=0.94, bottom=0.03,
    )

    TITLE_FS  = 13
    SUPTITLE_FS = 18
    CBAR_FS   = 9
    GL_FS     = 7

    # ------------------------------------------------------------------ #
    # Shared geometry: target box, zoom box
    # ------------------------------------------------------------------ #
    def _make_rect(lon0, lon1, lat0, lat1, color, lw, ls='-'):
        return Rectangle(
            (lon0, lat0), lon1 - lon0, lat1 - lat0,
            linewidth=lw, edgecolor=color, facecolor='none',
            linestyle=ls, transform=projection, zorder=6
        )

    target_kw = dict(color='black', lw=1.8)
    zoom_kw   = dict(color='red',  lw=1.2, ls='--')

    if target is not None:
        lon0, lon1, lat0, lat1 = target[0],target[1],target[2],target[3]

    zlon0, zlon1, zlat0, zlat1 = zoom_region[0],zoom_region[1],zoom_region[2],zoom_region[3]

    # ------------------------------------------------------------------ #
    # Global vmin / vmax across all input channels  (kept as requested)
    # ------------------------------------------------------------------ #
    global_vals = []
    for input_name in input_names:
        idx = (batch.forcing.feature_names_to_idx[input_name] if forcing
               else batch.inputs.feature_names_to_idx[input_name])

        if step == 0:
            raw = (batch.forcing.tensor[0, 0, :, :, idx] if forcing
                   else batch.inputs.tensor[0, 0, :, :, idx])
            raw = denormalize(
                raw.detach().cpu(),
                torch.tensor(stats[input_name]["mean"]),
                torch.tensor(stats[input_name]["std"]),
            ).numpy()
        else:
            raw = output.tensor[0, step - 1, :, :, idx].detach().cpu().numpy()

        global_vals.append(raw)

    all_vals  = np.stack(global_vals)
    vmin_glob = float(np.min(all_vals))
    vmax_glob = float(np.max(all_vals))

    # ------------------------------------------------------------------ #
    # Helper: add gridlines
    # ------------------------------------------------------------------ #
    def _add_gridlines(ax, left_col: bool):
        gl = ax.gridlines(
            draw_labels=left_col,
            linewidth=0.35, linestyle='--',
            color='gray', alpha=0.55,
        )
        gl.top_labels   = False
        gl.right_labels = False
        gl.xlabel_style = {'size': GL_FS}
        gl.ylabel_style = {'size': GL_FS}

    # ------------------------------------------------------------------ #
    # Helper: attach a thin colorbar to an axes
    # ------------------------------------------------------------------ #
    def _add_cbar(fig, ax, im, label: str):
        cax = ax.inset_axes([1.008, 0.08, 0.026, 0.84])
        cb  = fig.colorbar(im, cax=cax)
        cb.set_label(label, fontsize=CBAR_FS)
        cb.ax.tick_params(labelsize=CBAR_FS - 1)
        return cb

    # ------------------------------------------------------------------ #
    # Main loop
    # ------------------------------------------------------------------ #
    time_label = f"t+{step}h" if step > 0 else "t"

    for col, input_name in enumerate(input_names):
        is_left = (col == 0)

        idx = (batch.forcing.feature_names_to_idx[input_name] if forcing
               else batch.inputs.feature_names_to_idx[input_name])

        # ---------- Row 0: input field ----------
        ax_in = fig.add_subplot(gs[0, col], projection=projection)
        ax_in.add_feature(cfeature.COASTLINE, linewidth=0.7)
        ax_in.add_feature(cfeature.BORDERS,   linewidth=0.5)
        ax_in.set_extent(extent, crs=projection)
        # _add_gridlines(ax_in, is_left)

        inp = global_vals[col]           # already computed above
        im_in = ax_in.imshow(
            inp, cmap='RdBu_r', origin='lower',
            extent=extent, transform=projection,
            vmin=vmin_glob, vmax=vmax_glob,
        )
        ax_in.set_title(f"Input ({time_label}) — {input_name}",
                        fontsize=TITLE_FS, pad=6, fontweight='bold')

        if target is not None:
            ax_in.add_patch(_make_rect(lon0, lon1, lat0, lat1, **target_kw))
        # if has_zoom:
        #     ax_in.add_patch(_make_rect(zlon0, zlon1, zlat0, zlat1, **zoom_kw))

        _add_cbar(fig, ax_in, im_in, label=input_name)

        # ---------- Row 1: attribution ----------
        # ax_attr = fig.add_subplot(gs[1, col], projection=projection)
        # ax_attr.add_feature(cfeature.COASTLINE, linewidth=0.7)
        # ax_attr.add_feature(cfeature.BORDERS,   linewidth=0.5)
        # ax_attr.set_extent(extent, crs=projection)
        # _add_gridlines(ax_attr, is_left)

        raw_attr = explanations[input_name].reshape(512, 640).cpu()
        attr     = normalize_attr(raw_attr)

        # im_attr = ax_attr.imshow(
        #     attr, cmap='RdBu_r', origin='lower',
        #     extent=extent, transform=projection,
        #     vmin=-1, vmax=1,
        # )
        # ax_attr.set_title(f"Attribution — {input_name}",
        #                   fontsize=TITLE_FS, pad=6, fontweight='bold')

        # if target is not None:
        #     ax_attr.add_patch(_make_rect(lon0, lon1, lat0, lat1, **target_kw))
        # if has_zoom:
        #     ax_attr.add_patch(_make_rect(zlon0, zlon1, zlat0, zlat1, **zoom_kw))

        # _add_cbar(fig, ax_attr, im_attr, label='Normalized attribution')

        # ---------- Row 2: zoomed attribution (optional) ----------
        ax_zoom = fig.add_subplot(gs[1, col], projection=projection)
        ax_zoom.add_feature(cfeature.COASTLINE, linewidth=0.7)
        ax_zoom.add_feature(cfeature.BORDERS,   linewidth=0.5)
        ax_zoom.set_extent(zoom_region, crs=projection)
        # _add_gridlines(ax_zoom, is_left)

        im_zoom = ax_zoom.imshow(
            attr, cmap='RdBu_r', origin='lower',
            extent=extent, transform=projection,
            vmin=-1, vmax=1,
        )
        ax_zoom.set_title(f"Attribution (Zoomed) — {input_name}",
                            fontsize=TITLE_FS, pad=6, fontweight='bold')

        if target is not None:
            ax_zoom.add_patch(_make_rect(lon0, lon1, lat0, lat1, **target_kw))

        _add_cbar(fig, ax_zoom, im_zoom, label='Normalized attribution')

    # ------------------------------------------------------------------ #
    # Suptitle
    # ------------------------------------------------------------------ #
    # title_str = (
    #     f"Attribution Maps — {method} — {runtime} + {step}h"
    #     if step > 0 else
    #     f"Attribution Maps — {method} — {runtime}"
    # )
    # fig.suptitle(title_str, fontsize=SUPTITLE_FS, fontweight='bold', y=0.997)

    # ------------------------------------------------------------------ #
    # Save / show
    # ------------------------------------------------------------------ #
    if save_figs:
        out_dir = os.path.join(fig_path, method)
        os.makedirs(out_dir, exist_ok=True)
        save_path = os.path.join(out_dir, f"{runtime}_step{step}.png")
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close(fig)
        print(f"Saved → {save_path}")
    else:
        plt.show()

def plot_diff_levels_optionA(
    steps_explanations: dict,   # {step: {input_name: tensor (512,640)}}
    batch, output, extent, method, fig_path, runtime,
    stats, output_name, target, forcing, save_figs, input_names,
    zoom_region=None,
):
    import matplotlib; matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import numpy as np, os, copy, torch
    import cartopy.crs as ccrs
    import cartopy.feature as cfeature
    from matplotlib.patches import Rectangle

    steps      = sorted(steps_explanations.keys())   # [1, 2, …]
    n_inputs   = len(input_names)
    n_rows     = 1 + len(steps)                      # input row + one per step
    projection = ccrs.PlateCarree()
    map_extent = zoom_region if zoom_region is not None else extent

    TITLE_FS   = 11
    SUPTITLE_FS= 14
    CBAR_FS    = 8

    fig, axes = plt.subplots(
        n_rows, n_inputs,
        figsize=(4.0 * n_inputs, 3.2 * n_rows + 0.5),
        subplot_kw=dict(projection=projection),
        gridspec_kw=dict(
            hspace=0.15, wspace=0.08,
            left=0.06, right=0.91,
            top=0.95,  bottom=0.02,
        ),
    )

    # Always 2-D index
    if n_rows == 1:
        axes = axes[np.newaxis, :]
    if n_inputs == 1:
        axes = axes[:, np.newaxis]

    # ── Helpers ──────────────────────────────────────────────────────────────
    def _style(ax):
        ax.add_feature(cfeature.COASTLINE, linewidth=0.5)
        ax.add_feature(cfeature.BORDERS,   linewidth=0.4)
        ax.set_extent(map_extent, crs=projection)
        # ax.gridlines(linewidth=0.2, linestyle='--', color='gray', alpha=0.4)

    def _rect():
        if target is None: return None
        lon0, lon1, lat0, lat1 = target
        return Rectangle((lon0, lat0), lon1-lon0, lat1-lat0,
                          linewidth=0.9, edgecolor='gold', facecolor='none',
                          transform=projection, zorder=6)

    def _cbar(fig, ax, im, label):
        cax = ax.inset_axes([1.03, 0.05, 0.04, 0.90])
        cb  = fig.colorbar(im, cax=cax)
        # cb.set_label(label, fontsize=CBAR_FS, labelpad=2)
        cb.ax.tick_params(labelsize=CBAR_FS - 1)
        return cb
    global_vals = []
    for input_name in input_names:
        idx = batch.inputs.feature_names_to_idx[input_name]

        raw =  batch.inputs.tensor[0, 0, :, :, idx]
        raw = denormalize(
            raw.detach().cpu(),
            torch.tensor(stats[input_name]["mean"]),
            torch.tensor(stats[input_name]["std"]),
        ).numpy()
    
        global_vals.append(raw)

    all_vals  = np.stack(global_vals)
    vmin_glob = float(np.min(all_vals))
    vmax_glob = float(np.max(all_vals))
    final = len(input_names)
    # ── Row 0: input fields ───────────────────────────────────────────────────
    for col, input_name in enumerate(input_names):
        ax = axes[0, col]
        _style(ax)

        idx = batch.inputs.feature_names_to_idx[input_name]
        raw = (batch.inputs.tensor[0, 0, :, :, idx]).detach().cpu()
        raw = denormalize(
            raw,
            torch.tensor(stats[input_name]["mean"]),
            torch.tensor(stats[input_name]["std"]),
        ).numpy()

        # vabs = max(abs(raw.min()), abs(raw.max()))
        im = ax.imshow(raw, cmap='RdBu_r', origin='lower',
                       extent=extent, transform=projection,
                       vmin=vmin_glob, vmax=vmax_glob)

        ax.set_title(input_name, fontsize=TITLE_FS, fontweight='bold', pad=4)

        r = _rect()
        if r: ax.add_patch(r)

        # Colorbar only on last column to save space, or per column — your choice
        if col == final-1: 
            _cbar(fig, ax, im, "Input")

    # Row label for input row
    axes[0, 0].text(-0.18, 0.5, "Input\nt=0",
                    transform=axes[0, 0].transAxes,
                    fontsize=12, fontweight='bold',
                    va='center', ha='center', rotation=90)
    # BEFORE the attribution loop: collect all attribution values across all panels
    # attr_vals = []
    # for step in steps:
    #     for input_name in input_names:
    #         attr_tensor = steps_explanations[step].get(input_name)
    #         if attr_tensor is None:
    #             continue
    #         attr_vals.append(
    #             (attr_tensor.reshape(512, 640).cpu()).numpy()
    #         )

    # attr_stack = np.stack(attr_vals)
    # # Symmetric scale is usually right for RdBu_r (signed data centered at 0)
    # vmin_attr, vmax_attr =float(np.min((attr_stack))), float(np.max((attr_stack)))
    # ── Rows 1…N: attribution per step ───────────────────────────────────────
    for r_idx, step in enumerate(steps):
        row = r_idx + 1
        final = len(input_names)
        attr_vals = []

        for input_name in input_names:
            attr_tensor = steps_explanations[step].get(input_name)
            if attr_tensor is None:
                continue
            a = attr_tensor.reshape(512, 640).cpu().numpy()
            a = a #/a.max()
            # attr_vals.append(
            #     (attr_tensor.reshape(512, 640).cpu()).numpy()
            # )
            attr_vals.append(a)
        attr_stack = np.stack(attr_vals)
        # Symmetric scale is usually right for RdBu_r (signed data centered at 0)
        vmin_attr, vmax_attr =float(np.min((attr_stack))), float(np.max((attr_stack)))
        for col, input_name in enumerate(input_names):
            ax = axes[row, col]
            _style(ax)

            attr_tensor = steps_explanations[step].get(input_name)
            if attr_tensor is None:
                ax.set_visible(False)
                continue

            # attr = normalize_attr(attr_tensor.reshape(512, 640).cpu())
            attr = attr_tensor.reshape(512, 640).cpu()
            attr = attr/vmax_attr
            im = ax.imshow(attr, cmap='RdBu_r', origin='lower',
                           extent=extent, transform=projection,
                           vmin=0, vmax=1)

            r = _rect()
            if r: ax.add_patch(r)
            if col == final-1:
                _cbar(fig, ax, im, "Attribution")

        # Row label
        axes[row, 0].text(-0.18, 0.5, f"t+{step}",
                          transform=axes[row, 0].transAxes,
                          fontsize=12, fontweight='bold',
                          va='center', ha='center', rotation=90)

    # fig.suptitle(f"{method}  —  {runtime}",
    #              fontsize=SUPTITLE_FS, fontweight='bold', y=0.988)

    if save_figs:
        out_dir = os.path.join(fig_path, method)
        os.makedirs(out_dir, exist_ok=True)
        out_path = os.path.join(out_dir, f"{runtime}.pdf")
        plt.savefig(out_path, dpi=300, bbox_inches='tight')
        plt.close(fig)
        print(f"Saved → {out_path}")
    else:
        plt.show()
# ============================================================================
# ATTRIBUTION DIFFERENCE PLOT
# ============================================================================

def plot_attribution_difference(attr1: np.ndarray, attr2: np.ndarray,
                               extent: List, title1: str, title2: str,
                               save_path: Optional[str] = None):
    """
    Plot two attribution maps and their difference.
    
    Args:
        attr1: First attribution map
        attr2: Second attribution map
        extent: Spatial extent
        title1: Title for first map
        title2: Title for second map
        save_path: Optional path to save figure
    """
    # Normalize
    attr1_norm = normalize_attr(attr1)
    attr2_norm = normalize_attr(attr2)
    
    # Compute difference
    diff = attr1_norm - attr2_norm
    
    fig, axes = plt.subplots(
        1, 3, figsize=(18, 5),
        subplot_kw={'projection': ccrs.PlateCarree()}
    )
    
    for ax in axes:
        ax.add_feature(cfeature.COASTLINE, linewidth=0.6)
        ax.add_feature(cfeature.BORDERS, linewidth = 0.6)

        ax.set_extent(extent, crs=ccrs.PlateCarree())
    
    # Plot 1
    im1 = axes[0].imshow(attr1_norm, origin="lower", cmap="RdBu_r",
                        extent=extent, vmin=-1, vmax=1)
    axes[0].set_title(title1, fontsize=12)
    plt.colorbar(im1, ax=axes[0], orientation="vertical", fraction=0.046, pad=0.04)
    
    # Plot 2
    im2 = axes[1].imshow(attr2_norm, origin="lower", cmap="RdBu_r",
                        extent=extent, vmin=-1, vmax=1)
    axes[1].set_title(title2, fontsize=12)
    plt.colorbar(im2, ax=axes[1], orientation="vertical", fraction=0.046, pad=0.04)
    
    # Difference
    im3 = axes[2].imshow(diff, origin="lower", cmap="RdBu_r",
                        extent=extent, vmin=-1, vmax=1)
    axes[2].set_title("Difference (1 - 2)", fontsize=12)
    plt.colorbar(im3, ax=axes[2], orientation="vertical", fraction=0.046, pad=0.04)
    
    plt.suptitle("Attribution Comparison", fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        plt.close()
    else:
        plt.show()


# ============================================================================
# GRID COMPARISON PLOT (Multiple explainers x Multiple examples)
# ============================================================================

def plot_explainer_grid(results_all: List[Dict], dataset_info,
                       cfg_xai: Dict, save_path: str,
                       max_examples: int = 8):
    """
    Create a compact grid showing all explainers for multiple examples.
    Rows = Explainers, Columns = Examples.
    
    Args:
        results_all: List of result dicts
        dataset_info: Dataset info
        cfg_xai: Config dict
        save_path: Save path
        max_examples: Maximum number of examples to show
    """
    num_examples = min(len(results_all), max_examples)
    explainer_names = [k for k in results_all[0].keys() 
                      if k not in ['batch', 'pred']]
    
    extent = cfg_xai["extent"]
    target = cfg_xai.get("target")
    
    n_rows = len(explainer_names)
    n_cols = num_examples
    
    fig, axes = plt.subplots(
        n_rows, n_cols,
        figsize=(3 * n_cols, 2.5 * n_rows),
        subplot_kw={'projection': ccrs.PlateCarree()}
    )
    
    # Ensure 2D
    if n_rows == 1:
        axes = axes.reshape(1, -1)
    if n_cols == 1:
        axes = axes.reshape(-1, 1)
    
    input_name = cfg_xai["explain"]["input"]
    batch = results_all[0]["batch"]
    
    if cfg_xai["explain"].get("forcing_input", False):
        input_idx = batch.forcing.feature_names_to_idx[input_name]
    else:
        input_idx = batch.inputs.feature_names_to_idx[input_name]
    
    # Plot grid
    for r, explainer_name in enumerate(explainer_names):
        for c, example_results in enumerate(results_all[:num_examples]):
            ax = axes[r, c]
            
            # Get attribution
            attr = example_results[explainer_name][0].squeeze(1)
            
            if cfg_xai["explain"].get("forcing_input", False):
                attr = attr[..., input_idx].numpy()
            else:
                attr = attr[0, ..., input_idx].numpy()
            
            attr = normalize_attr(attr)
            
            # Plot
            ax.add_feature(cfeature.COASTLINE, linewidth=0.3)
            ax.add_feature(cfeature.BORDERS, linewidth = 0.6)

            ax.set_extent(extent, crs=ccrs.PlateCarree())
            
            im = ax.imshow(attr, origin="lower", cmap="RdBu_r",
                          extent=extent, vmin=-1, vmax=1)
            
            # Labels
            if c == 0:
                ax.set_ylabel(explainer_name, fontsize=10, fontweight='bold')
            if r == 0:
                ax.set_title(f"Ex. {c+1}", fontsize=10)
            
            ax.set_xticks([])
            ax.set_yticks([])
            
            add_target_box(ax, target, linewidth=1)
            
            # Colorbar on last column
            if c == n_cols - 1:
                plt.colorbar(im, ax=ax, orientation="vertical",
                           fraction=0.046, pad=0.04)
    
    plt.suptitle(f"Explainer Grid: {input_name}", fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    fig_path = os.path.join(save_path, "explainer_grid")
    os.makedirs(fig_path, exist_ok=True)
    plt.savefig(
        os.path.join(fig_path, f"{input_name}_grid.png"),
        dpi=300, bbox_inches="tight"
    )
    plt.close()