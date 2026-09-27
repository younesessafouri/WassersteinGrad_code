from py4castxai.explainers.XGrad import XGrad
import torch
from py4castxai.utils_test.infer_utils import predict_step
import ot
from py4castxai.explainers.inputxgrad import InputxGrad
from py4castxai.explainers.basegrad import BaseGrad
import numpy as np
from abc import ABC, abstractmethod
from py4castxai.utils_test.grad_utils import compute_grad
from skimage.transform import resize
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import numpy as np
import torch
import os
from matplotlib.patches import Rectangle    
import copy
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
from matplotlib.colors import LogNorm

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
from matplotlib.colors import LogNorm
import torch.nn.functional as F
import math
                  
import matplotlib.gridspec as gridspec
  

import os
import numpy as np
import torch
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import Rectangle
import matplotlib.ticker as mticker
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from skimage.transform import resize
        
import os
import numpy as np
import torch
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import Rectangle
import matplotlib.gridspec as gridspec
from skimage.transform import resize
import cartopy.crs as ccrs
import cartopy.feature as cfeature

# # ── Typography / style ────────────────────────────────────────────────────────
plt.rcParams.update({
    "font.family":        "serif",
    "font.serif":         ["Times New Roman", "DejaVu Serif"],
    "axes.titlesize":     9,
    "axes.titlepad":      5,
    "axes.labelsize":     8,
    "xtick.labelsize":    7,
    "ytick.labelsize":    7,
    "axes.linewidth":     0.6,
    "xtick.major.width":  0.5,
    "ytick.major.width":  0.5,
    "figure.dpi":         300,
    "savefig.dpi":        300,
    "savefig.bbox":       "tight",
    "savefig.pad_inches": 0.05,
})

# ── Helpers ───────────────────────────────────────────────────────────────────
# ── Helpers ───────────────────────────────────────────────────────────────────

def _add_geo_features(ax):
    ax.add_feature(cfeature.COASTLINE, linewidth=0.5, edgecolor="#333333")
    ax.add_feature(cfeature.BORDERS,   linewidth=0.4, edgecolor="#555555",
                   linestyle="--")

def _colorbar(fig, im, ax, label=""):
    cb = fig.colorbar(im, ax=ax, orientation="vertical",
                      fraction=0.035, pad=0.03, shrink=0.92)
    cb.ax.tick_params(labelsize=6, length=2, width=0.4)
    cb.outline.set_linewidth(0.4)
    if label:
        cb.set_label(label, fontsize=6)
    return cb

def _panel_label(ax, letter):
    """Bold letter label in the top-left corner (NeurIPS convention)."""
    ax.text(0.015, 0.97, letter, transform=ax.transAxes,
            fontsize=9, va="top", ha="left",
            bbox=dict(facecolor="white", edgecolor="none",
                      alpha=0.7, pad=1.5))


# def normalize_expl_expl(attr):
#     """Normalise attribution map to [-1, 1]."""
#     amax = np.abs(attr).max()
#     return attr / (amax + 1e-9)


def apply_geometric_noise(x, max_translation=0.05, max_rotation=5.0):
    """
    Applies random geometric jitter (translation & rotation) to the input x.
    
    Args:
        x: Input tensor of shape (Batch, Time, Lat, Lon) or (Batch, Lat, Lon)
        max_translation: Max shift as a fraction of image size (e.g. 0.05 = 5%)
        max_rotation: Max rotation in degrees
        
    Returns:
        x_perturbed: Geometrically transformed tensor
    """
    # Ensure input is (N, C, H, W) for grid_sample
    # Assuming x is (Batch, Time, Lat, Lon) -> merge Batch/Time into N, treat Lat/Lon as H/W
    original_shape = x.shape
    if x.ndim == 4: # (B, T, H, W)
        B, T, H, W = x.shape
        x = x.view(B * T, 1, H, W)
    elif x.ndim == 3: # (B, H, W)
        B, H, W = x.shape
        x = x.unsqueeze(1) # (B, 1, H, W)
        T = 1
    
    N = x.shape[0]
    device = x.device

    # 1. Generate Random Translation (-max_t to +max_t)
    # Tx, Ty are shifts in normalized coordinates [-1, 1]
    tx = (torch.rand(N, device=device) * 2 - 1) * max_translation
    ty = (torch.rand(N, device=device) * 2 - 1) * max_translation

    # 2. Generate Random Rotation (-max_r to +max_r)
    theta = (torch.rand(N, device=device) * 2 - 1) * (max_rotation * math.pi / 180.0)

    # 3. Construct Affine Matrix [ 2x3 ]
    # | cos(theta)  -sin(theta)   tx |
    # | sin(theta)   cos(theta)   ty |
    cos_t = torch.cos(theta)
    sin_t = torch.sin(theta)
    
    affine_matrices = torch.zeros(N, 2, 3, device=device)
    affine_matrices[:, 0, 0] = cos_t
    affine_matrices[:, 0, 1] = -sin_t
    affine_matrices[:, 0, 2] = tx
    affine_matrices[:, 1, 0] = sin_t
    affine_matrices[:, 1, 1] = cos_t
    affine_matrices[:, 1, 2] = ty

    # 4. Create Grid and Sample
    grid = F.affine_grid(affine_matrices, x.size(), align_corners=False)
    
    # mode='bilinear' for smooth interpolation
    # padding_mode='reflection' or 'zeros' (reflection is usually better for weather maps to avoid hard edges)
    x_perturbed = F.grid_sample(x, grid, mode='bilinear', padding_mode='reflection', align_corners=False)

    # Restore original shape
    if len(original_shape) == 4:
        x_perturbed = x_perturbed.view(B, T, H, W)
    elif len(original_shape) == 3:
        x_perturbed = x_perturbed.squeeze(1)

    return x_perturbed


def normalize_sum(a):
    a = torch.abs(a)  # ensure non-negative
    return a / (a.sum() + 1e-8)

def normalize_expl(a,abs=True):
            
            if isinstance(a, np.ndarray):
                a = torch.from_numpy(a).to(torch.float32)
            if abs:
                return a.abs() #/ (a.abs().max() + 1e-8)
            return a #/ (a.abs().max() + 1e-8)

def normalize_expl_expl(a):
     
    if isinstance(a, np.ndarray):
        a = torch.from_numpy(a).to(torch.float32)

    return a / (a.abs().max() + 1e-8)


class otGrad(XGrad):


    def __init__(self,model,base_explainer="BaseGrad",std_perturbations=0.01,num_perturbations=50):
        super().__init__(model)
        if base_explainer == "InputxGrad":
            self.base_explainer = InputxGrad(model)
        else:
            self.base_explainer = BaseGrad(model)
        
        self.std_perturbation = std_perturbations
        self.num_perturbations = num_perturbations

    def compute_explanations(self,
                             input,batch_idx,checkpoint,cfg_model,cfg_dataset,
                             cfg_xai,dataset_info,infer_ds,list_run_hour,use_old_weights,
                             target,plot_explanations = True,
                             extent=[-12, 16, 37.5, 55.4],
                             fig_path=".figs/attributions",return_output = False):
        #print(input.forcing)
            print( infer_ds.dataset_info.domain_info.grid_limits)
            output_idx = input.outputs.feature_names_to_idx[cfg_xai["explain"]["output"]]
                 
            input_idx = input.inputs.feature_names_to_idx[cfg_xai["explain"]["input"]]
            fig_path_step =fig_path+ f"/input_{cfg_xai['explain']['input']}_output_{cfg_xai['explain']['output']}"
            batch_size = input.inputs.tensor.shape[0]

            idx_samples = [batch_idx * batch_size + b for b in range(batch_size)]
            samples = [infer_ds.sample_list[idx] for idx in idx_samples]
            runtimes = [
                sample.timestamps.datetime.strftime("%Y%m%d%H") for sample in samples
            ]

            
            stats = dataset_info.stats
            
            grads = {i:[] for i in range(input.num_pred_steps)}
            grads_wasser= {i:[] for i in range(input.num_pred_steps)}
            grads_wasser_pos= {i:[] for i in range(input.num_pred_steps)}
            grads_wasser_neg= {i:[] for i in range(input.num_pred_steps)}
            transport_plans= {i:[] for i in range(input.num_pred_steps)}
            output = predict_step(self.model,input,batch_idx,checkpoint,cfg_model,cfg_dataset,cfg_xai,dataset_info,infer_ds,list_run_hour,use_old_weights,compute_grads=False)
            
            
            if output is None:
                if return_output:
                    return None,None
                return
            
       
            backup_tensor = input.inputs.tensor

            input_idx = input.inputs.feature_names_to_idx[cfg_xai["explain"]["input"]]
            input_clean = input.inputs.tensor[...,input_idx]

            vals = input.inputs.tensor[..., input_idx]
            print("running experiment with noise:",self.std_perturbation)
            std_perturbation = self.std_perturbation * (vals.max() - vals.min())            
            clean_gradient,_ = self.base_explainer.compute_explanations(input=input,batch_idx=batch_idx,checkpoint=checkpoint,cfg_model=cfg_model,cfg_dataset=cfg_dataset,
                                                                        cfg_xai=cfg_xai,dataset_info=dataset_info,infer_ds=infer_ds,list_run_hour=list_run_hour,use_old_weights=use_old_weights,
                                                                        target=target,plot_explanations=False,extent=extent,fig_path=fig_path,return_output=True)

            for k in range(self.num_perturbations): 
                
                # input.inputs.tensor =input_clean+ torch.randn_like(input.inputs.tensor)*std_perturbation
                input_perturbed = input_clean.clone().detach() + torch.randn_like(input_clean) * std_perturbation
                # input_perturbed = apply_geometric_noise(input_perturbed, max_translation=2, max_rotation=0)
                input.inputs.tensor = input.inputs.tensor.clone()
                input.inputs.tensor[..., input_idx] = input_perturbed
                gradient,_ = self.base_explainer.compute_explanations(input=input,batch_idx=batch_idx,checkpoint=checkpoint,cfg_model=cfg_model,cfg_dataset=cfg_dataset,
                                                                      cfg_xai=cfg_xai,dataset_info=dataset_info,infer_ds=infer_ds,list_run_hour=list_run_hour,use_old_weights=use_old_weights,target=target,plot_explanations=False,extent=extent,fig_path=fig_path,return_output=True)

                if gradient is None:
                    if return_output:
                        return None,None
                    return
                
                for i in range(len(gradient)):
                    grads[i].append(gradient[i][0,0,...,0])
                    #TODO: test the squared grads
                    grads_wasser[i].append(normalize_sum(gradient[i][0,0,...,0]))
                    grads_wasser_pos[i].append(normalize_sum(torch.clamp(gradient[i][0,0,...,0],min=0)))
                    grads_wasser_neg[i].append(normalize_sum(torch.clamp(gradient[i][0,0,...,0],max=0)))
                    if gradient is not None:
                    # Get the 2D gradient map (assuming shape is [H, W])
                        grad_2d_first = np.abs(gradient[0][0,0,...,0].cpu().numpy())
                        clean_grad_2d_first = np.abs(clean_gradient[0][0,0,...,0].cpu().numpy())
                        threshold_val = np.percentile(grad_2d_first, 90) # Keep top 10%
                        grad_2d = np.where(grad_2d_first > threshold_val, grad_2d_first, 0)
                        clean_threshold_val = np.percentile(clean_grad_2d_first, 90)
                        clean_grad_2d = np.where(clean_grad_2d_first > clean_threshold_val, clean_grad_2d_first, 0)
                    # Inside your loop: only keep the strongest gradients (the actual weather fronts)
                        SCALE = 16
                        H,W=512,640
                        small_grad  = resize(grad_2d,  (H // SCALE, W // SCALE), anti_aliasing=True)
                        small_clean = resize(clean_grad_2d, (H // SCALE, W // SCALE), anti_aliasing=True)
                        
                        h, w = small_grad.shape 
                        a = small_grad.flatten().astype(np.float64);  a /= a.sum()
                        b = small_clean.flatten().astype(np.float64); b /= b.sum()

                        rows, cols = np.meshgrid(np.arange(h), np.arange(w), indexing='ij')
                        coords = np.stack([rows.flatten(), cols.flatten()], axis=1).astype(np.float64)

                        M = ot.dist(coords, coords, metric='euclidean')
                        M /= M.max()

                        transport_plan = ot.emd(a, b, M)          
                        transport_plans[i].append(transport_plan)
                        # visualize_optimal_transport(grad_2d, clean_grad_2d, transport_plan, h, w, scale=SCALE, top_k=300)
                        # visualize_displacement_field(transport_plan, h, w, scale=SCALE, step=4)

                        #TODO: zero out the diagonal
                        np.fill_diagonal(transport_plan,0)
                        self.plot_explanations_with_ot(grads[i][k],input,output,
                                            extent,"Gradient",fig_path_step,
                                            runtime=runtimes[0],stats=stats,transport_plan=transport_plan,grad_2d= gradient[0][0,0,...,0].cpu().numpy(),
                                            clean_grad_2d=clean_gradient[0][0,0,...,0].cpu().numpy(),
                                            step=i,scale=SCALE,
                                            input_name = cfg_xai["explain"]["input"],
                                            output_name = cfg_xai["explain"]["output"],
                                        input_idx=input_idx,output_idx = output_idx,target=cfg_xai["target"],forcing=cfg_xai["explain"]["forcing_input"],save_figs=cfg_xai["explain"]["exp_plot"]["save_figs"])
           
        

            if cfg_xai["explain"]["forcing_input"]:
                input.forcing.tensor = backup_tensor
            else:

                input.inputs.tensor = backup_tensor
            

    # ── Main plotting function ─────────────────────────────────────────────────────

    def plot_explanations_with_ot(
        self, explanations, input, pred, extent, method, fig_path, runtime, stats,
        transport_plan, grad_2d, clean_grad_2d,
        step=0, input_name="aro_tp_0m", output_name="aro_tp_0m",
        input_idx=0, output_idx=0, target=None, forcing=None, save_figs=True,
        scale=8,
    ):
        # ── Colorbar helper ───────────────────────────────────────────────────────
        def _colorbar(fig, im, ax, label="", shrink=0.6, fraction=0.03, pad=0.03):
            cb = fig.colorbar(
                im, ax=ax,
                orientation="vertical",
                fraction=fraction,
                pad=pad,
                shrink=shrink,
            )
            cb.ax.tick_params(labelsize=6, width=0.4, length=2)
            cb.outline.set_linewidth(0.4)
            if label:
                cb.set_label(label, fontsize=6)
            return cb

        # ── Data extraction ───────────────────────────────────────────────────────
        attr = explanations.detach().cpu().numpy()

        if forcing:
            inp = input.forcing.tensor[0, step, :, :, input_idx].detach().cpu()
        else:
            inp = (
                input.inputs.tensor[0, 0, :, :, input_idx] if step == 0
                else pred.tensor[0, step - 1, :, :, input_idx]
            ).detach().cpu()

        pred_data = pred.tensor[0, step, :, :, output_idx].detach().cpu().numpy()
        gt        = input.outputs.tensor[0, step, :, :, output_idx].detach().cpu()

        means_input  = torch.as_tensor(stats[input_name]["mean"])
        std_input    = torch.as_tensor(stats[input_name]["std"])
        means_output = torch.as_tensor(stats[output_name]["mean"])
        std_output   = torch.as_tensor(stats[output_name]["std"])

        inp  = (inp  * std_input  + means_input ).numpy()
        gt   = (gt   * std_output + means_output).numpy()
        attr = normalize_expl_expl(attr)

        # ── OT maps ───────────────────────────────────────────────────────────────
        H_full, W_full = grad_2d.shape
        h, w = H_full // scale, W_full // scale

        mass_sent     = resize(transport_plan.sum(axis=1).reshape(h, w),
                            (H_full, W_full), anti_aliasing=True)
        mass_received = resize(transport_plan.sum(axis=0).reshape(h, w),
                            (H_full, W_full), anti_aliasing=True)

        # ── Figure ────────────────────────────────────────────────────────────────
        fig = plt.figure(figsize=(7, 2.6)) 
        proj = ccrs.PlateCarree()
        kw   = dict(origin="lower", extent=extent, transform=proj)

        gs = gridspec.GridSpec(
            1, 3,
            figure=fig,
            wspace=0.25,
            left=0.04, right=0.96,
            top=0.88, bottom=0.04,
        )

        ax_cg = fig.add_subplot(gs[0, 0], projection=proj)
        ax_pg = fig.add_subplot(gs[0, 1], projection=proj)
        ax_ot = fig.add_subplot(gs[0, 2], projection=proj)

        for ax in [ax_cg, ax_pg, ax_ot]:
            _add_geo_features(ax)
            ax.set_facecolor("white")

        # ── Gradients (shared colorbar) ───────────────────────────────────────────
        grad_abs_max         = max(np.abs(grad_2d).max(), np.abs(clean_grad_2d).max())
        grad_abs_min         = min(np.abs(grad_2d).min(), np.abs(clean_grad_2d).min())

        grad_vmin, grad_vmax = -1,1
        grad_kw = dict(cmap="RdBu_r", vmin=grad_vmin, vmax=grad_vmax, **kw)

        im_cg = ax_cg.imshow(normalize_expl_expl(clean_grad_2d), **grad_kw)
      

        im_pg = ax_pg.imshow(normalize_expl_expl(grad_2d), **grad_kw)
        ax_cg.set_title("")
        ax_cg.text(0.5, 1.08, f" Clean Gradient",
                transform=ax_cg.transAxes, fontsize=9, ha="center", va="bottom",
                linespacing=1.4)

        ax_pg.set_title("")
        ax_pg.text(0.5, 1.08,  f" Perturbed Gradient",
                transform=ax_pg.transAxes, fontsize=9, ha="center", va="bottom",
                linespacing=1.4)


        # single shared colorbar sitting to the right of ax_pg
        cb_grad = fig.colorbar(
            im_pg, ax=[ax_cg, ax_pg],
            orientation="vertical",
            fraction=0.03,
            pad=0.005,
            shrink=0.65,
        )
        cb_grad.ax.tick_params(labelsize=6, width=0.4, length=2)
        cb_grad.outline.set_linewidth(0.4)

        # ── OT mass transport ─────────────────────────────────────────────────────
        def _norm01(x):
            mn, mx = np.nanmin(x), np.nanmax(x)
            return (x - mn) / (mx - mn + 1e-12)

        ax_ot.imshow(_norm01(mass_sent),     cmap="Reds",   alpha=0.5, **kw)
        ax_ot.imshow(_norm01(mass_received), cmap="Greens", alpha=0.5, **kw)
        
        ax_ot.set_title("")
        ax_ot.text(0.5, 1.08, " OT Mass Transport",
                transform=ax_ot.transAxes, fontsize=9, ha="center", va="bottom",
                linespacing=1.4)
        ax_ot.legend(
            handles=[
                mpatches.Patch(facecolor="Red",   alpha=0.75, linewidth=0,
                            label="Sent (perturbed)"),
                mpatches.Patch(facecolor="Green", alpha=0.75, linewidth=0,
                            label="Received (clean)"),
            ],
            loc="lower left", fontsize=6,
            framealpha=0.88, edgecolor="#bbbbbb", fancybox=False,
            handlelength=1.0, handleheight=0.7,
            borderpad=0.5, labelspacing=0.3,
        )

        # ── Target bounding box ───────────────────────────────────────────────────
        if target is not None:
            lon0, lon1, lat0, lat1 = target
            for ax in [ax_cg, ax_pg, ax_ot]:
                ax.add_patch(Rectangle(
                    (lon0, lat0), lon1 - lon0, lat1 - lat0,
                    linewidth=1.0, edgecolor="black", facecolor="none",
                    transform=proj, zorder=5,
                ))

        # ── Save ──────────────────────────────────────────────────────────────────
        plt.savefig("transport_map_step1.pdf", bbox_inches="tight", dpi=300)
        plt.close()