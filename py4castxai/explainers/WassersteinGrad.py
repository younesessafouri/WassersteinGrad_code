from py4castxai.explainers.XGrad import XGrad
import torch
from py4castxai.utils_test.infer_utils import predict_step
import ot
from py4castxai.explainers.inputxgrad import InputxGrad
from py4castxai.explainers.basegrad import BaseGrad
import numpy as np
from abc import ABC, abstractmethod
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
from scipy.ndimage import center_of_mass

def visualize_optimal_transport(grad_2d, clean_grad_2d, transport_plan, h, w, scale=8, top_k=200):
    """
    Visualize the optimal transport plan between two gradient maps.
    """
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    
    # ── 1. Original gradient maps ──────────────────────────────────────────
    im0 = axes[0, 0].imshow(clean_grad_2d, cmap='hot')
    axes[0, 0].set_title('Clean Gradient', fontsize=13)
    plt.colorbar(im0, ax=axes[0, 0])

    im1 = axes[0, 1].imshow(grad_2d, cmap='hot')
    axes[0, 1].set_title('Perturbed Gradient', fontsize=13)
    plt.colorbar(im1, ax=axes[0, 1])

    # ── 2. Difference map ──────────────────────────────────────────────────
    diff = grad_2d - clean_grad_2d
    im2 = axes[0, 2].imshow(diff, cmap='RdBu_r', norm=plt.Normalize(-np.abs(diff).max(), np.abs(diff).max()))
    axes[0, 2].set_title('Difference (Perturbed - Clean)', fontsize=13)
    plt.colorbar(im2, ax=axes[0, 2])

    # ── 3. Transport plan as heatmap ───────────────────────────────────────
    # Sum over destination pixels → shows WHERE mass is being SENT FROM
    mass_sent = transport_plan.sum(axis=1).reshape(h, w)
    # Sum over source pixels → shows WHERE mass is ARRIVING
    mass_received = transport_plan.sum(axis=0).reshape(h, w)

    im3 = axes[1, 0].imshow(mass_sent, cmap='YlOrRd')
    axes[1, 0].set_title('Mass Sent (from perturbed)', fontsize=13)
    plt.colorbar(im3, ax=axes[1, 0])

    im4 = axes[1, 1].imshow(mass_received, cmap='YlGn')
    axes[1, 1].set_title('Mass Received (to clean)', fontsize=13)
    plt.colorbar(im4, ax=axes[1, 1])

    # ── 4. Flow arrows — top-K transport flows ─────────────────────────────
    axes[1, 2].imshow(clean_grad_2d, cmap='gray', alpha=0.5)
    axes[1, 2].imshow(grad_2d, cmap='hot', alpha=0.3)

    # Get top-K largest transport flows
    flat_indices = np.argsort(transport_plan.flatten())[-top_k:]
    src_indices, dst_indices = np.unravel_index(flat_indices, transport_plan.shape)

    for src, dst in zip(src_indices, dst_indices):
        # Convert flat pixel index → (row, col) in downsampled space
        src_row, src_col = src // w, src % w
        dst_row, dst_col = dst // w, dst % w

        # Scale back to original image space
        src_row_s, src_col_s = src_row, src_col   # DON'T scale back, stay in downsampled coords
        dst_row_s, dst_col_s = dst_row, dst_col
        weight = transport_plan[src, dst]
        alpha  = float(np.clip(weight / transport_plan.max() * 5, 0.1, 1.0))

        axes[1, 2].annotate(
            "", 
            xy=(dst_col_s, dst_row_s),        # arrow tip (destination)
            xytext=(src_col_s, src_row_s),     # arrow tail (source)
            arrowprops=dict(
                arrowstyle="->",
                color='cyan',
                alpha=alpha,
                lw=1.5
            )
        )

    axes[1, 2].set_title(f'Top-{top_k} Transport Flows', fontsize=13)
    axes[1, 2].set_xlim(0, w)
    axes[1, 2].set_ylim(h, 0)

    plt.suptitle(f'Optimal Transport Analysis\nWasserstein Distance', 
                 fontsize=15, fontweight='bold')
    plt.tight_layout()
    plt.savefig('optimal_transport_visualization.png', dpi=150, bbox_inches='tight')
    plt.show()
    print("Saved → optimal_transport_visualization.png")


# ── Extra: Net flow map (displacement field) ───────────────────────────────
def visualize_displacement_field(transport_plan, h, w, scale=8, step=5):
    """
    Shows the average displacement vector at each pixel — like an optical flow map.
    """
    rows, cols = np.meshgrid(np.arange(h), np.arange(w), indexing='ij')
    coords = np.stack([rows.flatten(), cols.flatten()], axis=1)

    # Expected destination for each source pixel
    # displacement[i] = sum_j( T[i,j] * (coord_j - coord_i) )
    expected_dst = transport_plan @ coords          # (N, 2)
    src_mass     = transport_plan.sum(axis=1, keepdims=True)  # (N, 1)
    src_mass     = np.where(src_mass == 0, 1e-10, src_mass)

    mean_dst     = expected_dst / src_mass          # (N, 2)
    displacement = (mean_dst - coords).reshape(h, w, 2)

    dy = displacement[..., 0]  # row displacement
    dx = displacement[..., 1]  # col displacement

    fig, ax = plt.subplots(figsize=(10, 8))
    magnitude = np.sqrt(dx**2 + dy**2)
    ax.imshow(magnitude, cmap='plasma')

    # Quiver plot — subsample for readability
    Y, X = np.mgrid[0:h:step, 0:w:step]
    ax.quiver(X * scale, Y * scale,
              dx[::step, ::step], dy[::step, ::step],
              magnitude[::step, ::step],
              cmap='cool', scale=50, alpha=0.8)

    ax.set_title('Gradient Mass Displacement Field\n(where perturbation moved the saliency)', fontsize=13)
    plt.tight_layout()
    # plt.savefig('displacement_field.png', dpi=150, bbox_inches='tight')
    plt.show()
    print("Saved → displacement_field.png")



def normalize_sum(a):
    a = torch.abs(a)  # ensure non-negative
    return a / (a.sum() + 1e-8)

def normalize_expl(a,abs=True):
            
            if isinstance(a, np.ndarray):
                a = torch.from_numpy(a).to(torch.float32)
            if abs:
                return a.abs() #/ (a.abs().max() + 1e-8)
            return a / (a.abs().max() + 1e-8)

def normalize_expl_expl(a):
     
    if isinstance(a, np.ndarray):
        a = torch.from_numpy(a).to(torch.float32)

    return a / (a.abs().max() + 1e-8)
import torch.nn.functional as F
import math

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
class WassersteinSmoothGrad(XGrad):


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
        
        output_idx = input.outputs.feature_names_to_idx[cfg_xai["explain"]["output"]]

         
        grads = {i:[] for i in range(input.num_pred_steps)}
        grads_wasser= {i:[] for i in range(input.num_pred_steps)}

        output = predict_step(self.model,input,batch_idx,checkpoint,cfg_model,cfg_dataset,cfg_xai,dataset_info,infer_ds,list_run_hour,use_old_weights,compute_grads=False)
        
        
        if output is None:
            if return_output:
                return None,None
            return
        
        if cfg_xai["explain"]["forcing_input"]:
            backup_tensor = input.forcing.tensor
            input_idx = input.forcing.feature_names_to_idx[cfg_xai["explain"]["input"]]
            input_clean = input.forcing.tensor[...,input_idx]
            vals = input.forcing.tensor[..., input_idx]
            std_perturbation = self.std_perturbation * (vals.max() - vals.min())   
            
            for _ in range(self.num_perturbations): 
                
                # input.inputs.tensor =input_clean+ torch.randn_like(input.inputs.tensor)*std_perturbation
                input_perturbed = input_clean.clone().detach() + torch.randn_like(input_clean) * std_perturbation
                input.forcing.tensor = input.forcing.tensor.clone()
                input.forcing.tensor[..., input_idx] = input_perturbed
                gradient,_ = self.base_explainer.compute_explanations(input=input,batch_idx=batch_idx,checkpoint=checkpoint,cfg_model=cfg_model,cfg_dataset=cfg_dataset,
                                                                      cfg_xai=cfg_xai,dataset_info=dataset_info,infer_ds=infer_ds,list_run_hour=list_run_hour,use_old_weights=use_old_weights,
                                                                      target=target,plot_explanations=False,extent=extent,fig_path=fig_path,return_output=True)

                if gradient is None:
                    if return_output:
                        return None,None
                    return
                
                for i in range(len(gradient)):
                    grads[i].append(gradient[i])
                
                self.model.zero_grad(set_to_none=True)

                
        else:
            backup_tensor = input.inputs.tensor

            input_idx = input.inputs.feature_names_to_idx[cfg_xai["explain"]["input"]]
            input_clean = input.inputs.tensor[...,input_idx]

            vals = input.inputs.tensor[..., input_idx]
            std_perturbation = self.std_perturbation * (vals.max() - vals.min())            

            for k in range(self.num_perturbations): 
                input_to_perturb = input_clean.clone().detach()
                # input.inputs.tensor =input_clean+ torch.randn_like(input.inputs.tensor)*std_perturbation
                # input_perturbed = apply_geometric_noise(input_to_perturb, max_translation=0.05, max_rotation=5.0)
                input_perturbed = input_clean.clone().detach() + torch.randn_like(input_clean) * std_perturbation
                input.inputs.tensor = input.inputs.tensor.clone()
                input.inputs.tensor[..., input_idx] = input_perturbed
                gradient,_ = self.base_explainer.compute_explanations(input=input,batch_idx=batch_idx,checkpoint=checkpoint,
                                                                      cfg_model=cfg_model,cfg_dataset=cfg_dataset,cfg_xai=cfg_xai,
                                                                      dataset_info=dataset_info,infer_ds=infer_ds,list_run_hour=list_run_hour,use_old_weights=use_old_weights,
                                                                      target=target,plot_explanations=False,extent=extent,fig_path=fig_path,return_output=True)

                if gradient is None:
                    if return_output:
                        return None,None
                    return
                
                for i in range(len(gradient)):
                    grads[i].append(gradient[i][0,0,...,0])
    
                    grads_wasser[i].append(normalize_sum(gradient[i][0,0,...,0]))

        
        grad_list = []
        wasserstein_list = []
    
        if not cfg_xai["explain"]["end_to_end"]:

            for i in range(input.num_pred_steps):
                
                A_maps = torch.stack(grads_wasser[i], dim =0)
         
                # barycentre = barycentre_pos-barycentre_neg 
                barycentre = ot.bregman.convolutional_barycenter2d(A_maps, reg=0.001)
                barycentre = torch.tensor(barycentre/barycentre.max())

                sg_map = (torch.stack(grads[i]).var(dim=0))
                weighted_barycentre =sg_map.reshape(512,640) * barycentre
       
                wasserstein_list.append(weighted_barycentre)
       

        else:
            A_maps = torch.stack(grads_wasser[0], dim =0)
                


            # barycentre = barycentre_pos-barycentre_neg 
            barycentre = ot.bregman.convolutional_barycenter2d(A_maps, reg=0.001)
            barycentre = torch.tensor(barycentre/barycentre.max())
            
            sg_map = (torch.stack(grads[0]).var(dim=0))

            weighted_barycentre =sg_map.reshape(512,640) * barycentre
     
            wasserstein_list.append(weighted_barycentre)

        if cfg_xai["explain"]["forcing_input"]:
            input.forcing.tensor = backup_tensor
        else:

            input.inputs.tensor = backup_tensor
        
        if plot_explanations:

            batch_size = input.inputs.tensor.shape[0]

            idx_samples = [batch_idx * batch_size + b for b in range(batch_size)]
            samples = [infer_ds.sample_list[idx] for idx in idx_samples]
            runtimes = [
                sample.timestamps.datetime.strftime("%Y%m%d%H") for sample in samples
            ]

            
            stats = dataset_info.stats
            # wasser[0] = grad_list[0].squeeze(1)

            for i in range(len(wasserstein_list)):
                output_idx =output.feature_names_to_idx[cfg_xai["explain"]["output"]] 
                if cfg_xai["explain"]["forcing_input"]:
                    input_idx = input.forcing.feature_names_to_idx[cfg_xai["explain"]["input"]]
                else:
                    input_idx = input.inputs.feature_names_to_idx[cfg_xai["explain"]["input"]]
                fig_path_step =fig_path+ f"/{i+1}_step/input_{cfg_xai['explain']['input']}_output_{cfg_xai['explain']['output']}"
                self.plot_explanations(wasserstein_list[i],input,output,
                                        extent,"WassersteinGrad",fig_path_step,
                                        runtime=runtimes[0],stats=stats,
                                        step=i,
                                        input_name = cfg_xai["explain"]["input"],
                                        output_name = cfg_xai["explain"]["output"],
                                        input_idx=input_idx,output_idx = output_idx,target=cfg_xai["target"],forcing=cfg_xai["explain"]["forcing_input"],save_figs=cfg_xai["explain"]["exp_plot"]["save_figs"])


        if return_output:

            return  wasserstein_list, output         
        

        return wasserstein_list
    
    def plot_explanations(self, explanations, input, pred, extent, method,fig_path,runtime,stats,step=0,input_name = "aro_tp_0m",
                          output_name="aro_tp_0m",input_idx = 0,output_idx = 0,target=None,forcing=None,save_figs=True):
        
        # --- Extract data ---
        
              # attr = explanations[0,...,0,0].detach().cpu().numpy()  # [H, W]
        attr = explanations.detach().cpu().numpy()  # [H, W]

        if forcing:
            inp = input.forcing.tensor[0, step, :, :, input_idx].detach().cpu()
        else:
            if step == 0:

                inp = input.inputs.tensor[0, 0, :, :, input_idx].detach().cpu()

            else:
                inp = pred.tensor[0, step-1, :, :, input_idx].detach().cpu()
       


        pred = pred.tensor[0, step, :, :, output_idx].detach().cpu().numpy()
        
        gt  = input.outputs.tensor[0, step, :, :, output_idx].detach().cpu()
        
        
     
        means_input = torch.asarray(stats[input_name]["mean"])
        std_input = torch.asarray(stats[input_name]["std"])

        means_output = torch.asarray(stats[output_name]["mean"])
        std_output = torch.asarray(stats[output_name]["std"])

            
        inp *= std_input
        inp += means_input
        
        gt *=std_output
        gt += means_output

        inp = inp.numpy()
        gt = gt.numpy()

        attr = normalize_expl_expl(attr)

        fig, axes = plt.subplots(
            1, 4,
            figsize=(40, 16),
            subplot_kw={'projection': ccrs.PlateCarree()}
        )
      
        for ax in axes:
            ax.add_feature(cfeature.COASTLINE, linewidth=0.6)
            ax.add_feature(cfeature.BORDERS, linewidth = 0.6)
         

        vmin = min(gt.min(), pred.min())
        vmax = max(gt.max(), pred.max())

        # ---------- 1️ Input (Ground Truth t)
        im0 = axes[0].imshow(inp, origin="lower", cmap="RdBu_r", extent=extent)
        axes[0].set_title("Input (t)")
        cbar0 = fig.colorbar(im0, ax=axes[0], orientation="vertical", fraction=0.046, pad=0.04)
       
    
        # # ---------- 2 Ground Truth (t+1)
     
        im1 = axes[1].imshow(gt, origin="lower", cmap="RdBu_r", extent=extent,vmin=vmin,vmax=vmax)
        axes[1].set_title("Ground Truth (t+1)")
        cbar1 = fig.colorbar(im1, ax=axes[1], orientation="vertical", fraction=0.046, pad=0.04)
      

        # ---------- 3 Prediction (t+1)
        # diff = np.abs(pred - inp)
        im2 = axes[2].imshow(pred, origin="lower", cmap="RdBu_r", extent=extent,vmin=vmin,vmax=vmax)
        axes[2].set_title("Prediction (t+1)")
        # axes[0].add_patch(copy.deepcopy(rect))

        cbar2 = fig.colorbar(im2, ax=axes[2], orientation="vertical", fraction=0.046, pad=0.04)
        # cbar2.set_label("Predicted scale")

        # ---------- 4 Attribution
        # thresh = 0.2
        # abs_max_attr = attr.abs().max()
        # threshold_value = thresh * abs_max_attr
        # threshold = 0.15
        
        # Mask = attr.abs() > threshold
        # attr = attr*Mask     
        # # Mettre à zéro les valeurs dont l'attribution absolue est inférieure au seuil
        # # np.where(condition, valeur_si_vrai, valeur_si_faux)
        # attr = np.where(attr.abs() < threshold_value, 0, attr)
        im3 = axes[3].imshow(
            attr,
            origin="lower",
            cmap="RdBu_r",
            extent=extent,
            transform=ccrs.PlateCarree(),
            vmin=-1, vmax=1
        )

        axes[3].set_title(f"Attribution — {method}")
        cbar3 = fig.colorbar(im3, ax=axes[3], orientation="vertical", fraction=0.046, pad=0.04)
        cbar3.set_label("Attribution intensity (normalized)")
        
        if target is not None:    
            lon0,lon1 = target[0],target[1]
            lat0,lat1 = target[2],target[3]
            rect = Rectangle(
                    (lon0, lat0),
                    lon1 - lon0,
                    lat1 - lat0,
                    linewidth=2,
                    edgecolor='black',
                    facecolor='none',
                    transform=ccrs.PlateCarree(),
                    zorder=5
                )
            axes[0].add_patch(copy.deepcopy(rect))
            axes[1].add_patch(copy.deepcopy(rect))
            axes[2].add_patch(copy.deepcopy(rect))
            axes[3].add_patch(copy.deepcopy(rect))


        fig.suptitle(f"Input: {input_name} Output: {output_name} — Input / Target / Prediction / Attribution", fontsize=15)
        plt.tight_layout()
      
        fig_path = fig_path+"/"+method
        os.makedirs(fig_path, exist_ok=True)
        if save_figs:
            plt.savefig(fig_path+"/"+runtime, dpi=300, bbox_inches="tight")
            plt.close(fig)
        else:
            plt.show()


class WassersteinGrad(XGrad):


    def __init__(self,model,base_explainer="BaseGrad",std_perturbations=0.1,num_perturbations=50,reg=0.001):
        super().__init__(model)
        if base_explainer == "InputxGrad":
            self.base_explainer = InputxGrad(model)
        else:
            self.base_explainer = BaseGrad(model)
        
        self.std_perturbation = std_perturbations
        self.num_perturbations = num_perturbations
        self.reg = reg
        print("Lambda sinkhorn = ",self.reg)

    def compute_explanations(self,
                             input,batch_idx,checkpoint,cfg_model,cfg_dataset,
                             cfg_xai,dataset_info,infer_ds,list_run_hour,use_old_weights,
                             target,plot_explanations = True,
                             extent=[-12, 16, 37.5, 55.4],
                             fig_path=".figs/attributions",return_output = False):
        #print(input.forcing)
        
        output_idx = input.outputs.feature_names_to_idx[cfg_xai["explain"]["output"]]

         
        grads = {i:[] for i in range(input.num_pred_steps)}
        grads_wasser= {i:[] for i in range(input.num_pred_steps)}
        grads_wasser_pos= {i:[] for i in range(input.num_pred_steps)}
        grads_wasser_neg= {i:[] for i in range(input.num_pred_steps)}
        # transport_plans= {i:[] for i in range(input.num_pred_steps)}
        output = predict_step(self.model,input,batch_idx,checkpoint,cfg_model,cfg_dataset,cfg_xai,dataset_info,infer_ds,list_run_hour,use_old_weights,compute_grads=False)
        
        
        if output is None:
            if return_output:
                return None,None
            return
   
        if cfg_xai["explain"]["forcing_input"]:
            backup_tensor = input.forcing.tensor
            input_idx = input.forcing.feature_names_to_idx[cfg_xai["explain"]["input"]]
            input_clean = input.forcing.tensor[...,input_idx]
            vals = input.forcing.tensor[..., input_idx]
            std_perturbation = self.std_perturbation * (vals.max() - vals.min())   
            
            for _ in range(self.num_perturbations): 
                
                # input.inputs.tensor =input_clean+ torch.randn_like(input.inputs.tensor)*std_perturbation
                input_perturbed = input_clean.clone().detach() + torch.randn_like(input_clean) * std_perturbation
                input.forcing.tensor = input.forcing.tensor.clone()
                input.forcing.tensor[..., input_idx] = input_perturbed
                gradient,_ = self.base_explainer.compute_explanations(input=input,batch_idx=batch_idx,checkpoint=checkpoint,cfg_model=cfg_model,cfg_dataset=cfg_dataset,
                                                                      cfg_xai=cfg_xai,dataset_info=dataset_info,infer_ds=infer_ds,list_run_hour=list_run_hour,use_old_weights=use_old_weights,
                                                                      target=target,plot_explanations=False,extent=extent,fig_path=fig_path,return_output=True)

                if gradient is None:
                    if return_output:
                        return None,None
                    return
                
                for i in range(len(gradient)):
                    grads[i].append(gradient[i])
                
                self.model.zero_grad(set_to_none=True)

                
        else:
            backup_tensor = input.inputs.tensor

            input_idx = input.inputs.feature_names_to_idx[cfg_xai["explain"]["input"]]
            input_clean = input.inputs.tensor[...,input_idx]

            vals = input.inputs.tensor[..., input_idx]
            std_perturbation = self.std_perturbation * (vals.max() - vals.min())            
            # clean_gradient,_ = self.base_explainer.compute_explanations(input=input,batch_idx=batch_idx,checkpoint=checkpoint,cfg_model=cfg_model,cfg_dataset=cfg_dataset,
            #                                                             cfg_xai=cfg_xai,dataset_info=dataset_info,infer_ds=infer_ds,list_run_hour=list_run_hour,use_old_weights=use_old_weights,
            #                                                             target=target,plot_explanations=False,extent=extent,fig_path=fig_path,return_output=True)

            for k in range(self.num_perturbations): 
                
                # input.inputs.tensor =input_clean+ torch.randn_like(input.inputs.tensor)*std_perturbation
                input_perturbed = input_clean.clone().detach() + torch.randn_like(input_clean) * std_perturbation
                input.inputs.tensor = input.inputs.tensor.clone()
                input.inputs.tensor[..., input_idx] = input_perturbed
                gradient,_ = self.base_explainer.compute_explanations(input=input,batch_idx=batch_idx,checkpoint=checkpoint,cfg_model=cfg_model,cfg_dataset=cfg_dataset,cfg_xai=cfg_xai,dataset_info=dataset_info,infer_ds=infer_ds,list_run_hour=list_run_hour,use_old_weights=use_old_weights,target=target,plot_explanations=False,extent=extent,fig_path=fig_path,return_output=True)

                if gradient is None:
                    if return_output:
                        return None,None
                    return
                
                for i in range(len(gradient)):
                    
                    grads_wasser[i].append(normalize_sum(gradient[i].reshape(512,640)))
                    # grads_wasser[i].append((gradient[i].reshape(512,640)))

                    # grads_wasser_pos[i].append(normalize_sum(torch.clamp(gradient[i].reshape(512,640),min=0)))
                    # grads_wasser_neg[i].append(normalize_sum(torch.clamp(gradient[i].reshape(512,640),max=0)))
                    
        wasserstein_list = []
        if not cfg_xai["explain"]["end_to_end"]:

            for i in range(input.num_pred_steps):
                
                A_maps = torch.stack(grads_wasser[i], dim =0)
                
                # #positive 
                # A_maps_pos = torch.stack(grads_wasser_pos[i], dim =0)
                # A_maps_neg = torch.stack(grads_wasser_neg[i], dim =0)

                # barycentre_pos= ot.bregman.convolutional_barycenter2d(A_maps_pos, reg=0.001)
                # barycentre_pos = barycentre_pos/barycentre_pos.max()
                # barycentre_neg= ot.bregman.convolutional_barycenter2d(A_maps_neg, reg=0.001)
                # barycentre_neg = barycentre_neg/barycentre_neg.max()

                # barycentre = barycentre_pos-barycentre_neg 

                barycentre = ot.bregman.convolutional_barycenter2d(A_maps, reg=self.reg)
                # barycentre = ot.unbalanced.barycenter_unbalanced_sinkhorn(A_maps,reg=self.reg)
                barycentre = torch.tensor(barycentre/barycentre.max())
        
                # weighted_barycentre =clean_gradient[i].reshape(512,640) * barycentre
                #weighted_barycentre =input.inputs.tensor[0,0,...,input_idx] * barycentre

                # weighted_barycentre = torch.clamp(clean_gradient[i][0,0,...,0],min=0)*barycentre_pos +  torch.clamp(clean_gradient[i][0,0,...,0],max=0)*barycentre_neg


                # weighted_barycentre = normalize_expl(weighted_barycentre)
                # print(weighted_barycentre.shape)
                wasserstein_list.append(barycentre)
                # grad_list.append(sg_map)
                # grad_list.append(torch.stack(grads[i]).mean(dim=0).unsqueeze(-1))

        else:
            A_maps = torch.stack(grads_wasser[0], dim =0)
                
            # #positive 
            # A_maps_pos = torch.stack(grads_wasser_pos[i], dim =0)
            # A_maps_neg = torch.stack(grads_wasser_neg[i], dim =0)

            # barycentre_pos= ot.bregman.convolutional_barycenter2d(A_maps_pos, reg=0.01)
            # barycentre_pos = barycentre_pos#/barycentre_pos.max()
            # barycentre_neg= ot.bregman.convolutional_barycenter2d(A_maps_neg, reg=0.01)
            # barycentre_neg = barycentre_neg#/barycentre_neg.max()

            # barycentre = barycentre_pos-barycentre_neg
            # print("hddhdhddhdhd") 
            barycentre = ot.bregman.convolutional_barycenter2d(A_maps, reg=self.reg)
            # barycentre = ot.unbalanced.barycenter_unbalanced_sinkhorn(A=A_maps,reg=self.reg,verbose=True)

            # barycentre = torch.tensor(barycentre/barycentre.max())
            # barycentre = barycentre**(1/0.5)
            
            #barycentre = barycentre/barycentre.max()
            wasserstein_list.append(barycentre)

    

        if cfg_xai["explain"]["forcing_input"]:
            input.forcing.tensor = backup_tensor
        else:

            input.inputs.tensor = backup_tensor
        
        if plot_explanations:

            batch_size = input.inputs.tensor.shape[0]

            idx_samples = [batch_idx * batch_size + b for b in range(batch_size)]
            samples = [infer_ds.sample_list[idx] for idx in idx_samples]
            runtimes = [
                sample.timestamps.datetime.strftime("%Y%m%d%H") for sample in samples
            ]

            output_idx =output.feature_names_to_idx[cfg_xai["explain"]["output"]] 
            if cfg_xai["explain"]["forcing_input"]:
                    input_idx = input.forcing.feature_names_to_idx[cfg_xai["explain"]["input"]]
            else:
                    input_idx = input.inputs.feature_names_to_idx[cfg_xai["explain"]["input"]]
            stats = dataset_info.stats
            # wasser[0] = grad_list[0].squeeze(1)
            if cfg_xai["explain"]["end_to_end"]:
                    step = input.num_pred_steps
                    fig_path_step =fig_path+ f"/end_to_end/{step}_step/input_{cfg_xai['explain']['input']}_output_{cfg_xai['explain']['output']}"
                    self.plot_explanations(wasserstein_list[0],input,output,
                                        extent,"WassersteinGrad",fig_path_step,
                                        runtime=runtimes[0],stats=stats,
                                        step=step-1,
                                        input_name = cfg_xai["explain"]["input"],
                                        output_name = cfg_xai["explain"]["output"],
                                        input_idx=input_idx,output_idx = output_idx,target=cfg_xai["target"],forcing=cfg_xai["explain"]["forcing_input"],save_figs=cfg_xai["explain"]["exp_plot"]["save_figs"])
                
            else:
                for i in range(len(wasserstein_list)):
                
                    fig_path_step =fig_path+ f"/{i+1}_step/input_{cfg_xai['explain']['input']}_output_{cfg_xai['explain']['output']}"
                    self.plot_explanations(wasserstein_list[i],input,output,
                                            extent,"WassersteinGrad",fig_path_step,
                                            runtime=runtimes[0],stats=stats,
                                            step=i,
                                            input_name = cfg_xai["explain"]["input"],
                                            output_name = cfg_xai["explain"]["output"],
                                            input_idx=input_idx,output_idx = output_idx,target=cfg_xai["target"],forcing=cfg_xai["explain"]["forcing_input"],save_figs=cfg_xai["explain"]["exp_plot"]["save_figs"])
                    

        if return_output:

            return  wasserstein_list, output         
        

        return wasserstein_list
    
    def plot_explanations(self, explanations, input, pred, extent, method,fig_path,runtime,stats,step=0,input_name = "aro_tp_0m",
                          output_name="aro_tp_0m",input_idx = 0,output_idx = 0,target=None,forcing=None,save_figs=True):
        
        # --- Extract data ---
        
              # attr = explanations[0,...,0,0].detach().cpu().numpy()  # [H, W]
        attr = explanations.detach().cpu().numpy()  # [H, W]

        if forcing:
            inp = input.forcing.tensor[0, step, :, :, input_idx].detach().cpu()
        else:
            if step == 0:

                inp = input.inputs.tensor[0, 0, :, :, input_idx].detach().cpu()

            else:
                inp = pred.tensor[0, step-1, :, :, input_idx].detach().cpu()
       


        pred = pred.tensor[0, step, :, :, output_idx].detach().cpu().numpy()
        
        gt  = input.outputs.tensor[0, step, :, :, output_idx].detach().cpu()
        
        
     
        means_input = torch.asarray(stats[input_name]["mean"])
        std_input = torch.asarray(stats[input_name]["std"])

        means_output = torch.asarray(stats[output_name]["mean"])
        std_output = torch.asarray(stats[output_name]["std"])

            
        inp *= std_input
        inp += means_input
        
        gt *=std_output
        gt += means_output

        inp = inp.numpy()
        gt = gt.numpy()

        # --- Normalize attribution for visualization only ---
        attr = normalize_expl_expl(attr)

       
        vmin = min(gt.min(), pred.min())
        vmax = max(gt.max(), pred.max())
        # from scipy.ndimage import laplace,gaussian_filter
        # z_250 = gaussian_filter(inp,sigma=20)
        # laplace_z = laplace(z_250)
        # mask = np.percentile(laplace_z,40)
        # laplace_z = np.where(laplace_z<mask,laplace_z,0)
              # ── 4. Figure ────────────────────────────────────────────────
        fig, axes = plt.subplots(
            1, 4,
            figsize=(40, 10),
            subplot_kw={"projection": ccrs.PlateCarree()},
        )
        fig.subplots_adjust(wspace=0.08, top=0.88)

        for ax in axes:
            ax.add_feature(cfeature.COASTLINE, linewidth=0.6)
            ax.add_feature(cfeature.BORDERS,   linewidth=0.6)

        panel_cfg = [
            (axes[0],inp,  "RdBu_r", None, None, f"Input  —  {input_name}  (t)"),
            (axes[1], gt,   "RdBu_r", vmin, vmax, f"Ground truth  —  {output_name}  (t+{step+1})"),
            (axes[2], pred, "RdBu_r", vmin, vmax, f"Prediction  (t+{step+1})"),
            (axes[3], attr,     "RdBu_r", -1,   1,    f"Attribution  —  {method}"), # if return_output:

        #     return  wasserstein_list, output         
        
        ]

        for ax, data, cmap, lo, hi, title in panel_cfg:
            kw = dict(origin="lower", cmap=cmap, extent=extent,
                      transform=ccrs.PlateCarree())
            if lo is not None:
                kw.update(vmin=lo, vmax=hi)
            im = ax.imshow(data, **kw)
            ax.set_title(title, fontsize=11, pad=6)
            cb = fig.colorbar(im, ax=ax, orientation="vertical",
                              fraction=0.046, pad=0.04)
            if ax is axes[3]:
                cb.set_label("Normalised attribution", fontsize=9)

        # ── 5. Optional target bounding box ─────────────────────────
        if target is not None:
            lon0, lon1, lat0, lat1 = target
            for ax in axes:
                ax.add_patch(Rectangle(
                    (lon0, lat0), lon1 - lon0, lat1 - lat0,
                    linewidth=2, edgecolor="black", facecolor="none",
                    transform=ccrs.PlateCarree(), zorder=5,
                ))

        # ── 6. Title, save / show ────────────────────────────────────
        fig.suptitle(
            f"Input: {input_name}   →   Output: {output_name}   |   {method}",
            fontsize=13, y=0.97,
        )

        out_dir = os.path.join(fig_path, method)
        os.makedirs(out_dir, exist_ok=True)

        if save_figs:
            plt.savefig(os.path.join(out_dir, runtime), dpi=300, bbox_inches="tight")
            plt.close(fig)
        else:
            plt.show()


    def plot_explanations_with_ot(self, explanations, input, pred, extent, method, fig_path, runtime, stats,
                               transport_plan, grad_2d, clean_grad_2d,  # ← add these
                               step=0, input_name="aro_tp_0m", output_name="aro_tp_0m",
                               input_idx=0, output_idx=0, target=None, forcing=None, save_figs=True,
                               scale=8):

        # --- Your existing data extraction (unchanged) ---
        attr = explanations.detach().cpu().numpy()
        if forcing:
            inp = input.forcing.tensor[0, step, :, :, input_idx].detach().cpu()
        else:
            if step == 0:
                inp = input.inputs.tensor[0, 0, :, :, input_idx].detach().cpu()
            else:
                inp = pred.tensor[0, step-1, :, :, input_idx].detach().cpu()
        pred_data = pred.tensor[0, step, :, :, output_idx].detach().cpu().numpy()
        gt        = input.outputs.tensor[0, step, :, :, output_idx].detach().cpu()

        means_input  = torch.asarray(stats[input_name]["mean"])
        std_input    = torch.asarray(stats[input_name]["std"])
        means_output = torch.asarray(stats[output_name]["mean"])
        std_output   = torch.asarray(stats[output_name]["std"])

        inp = (inp * std_input + means_input).numpy()
        gt  = (gt  * std_output + means_output).numpy()
        attr = normalize_expl_expl(attr)

        vmin = min(gt.min(), pred_data.min())
        vmax = max(gt.max(), pred_data.max())

        # --- Compute OT maps ---
        H_full, W_full = grad_2d.shape
        h, w = H_full // scale, W_full // scale

        # Mass sent / received (in original image space via resize)
        mass_sent     = resize(transport_plan.sum(axis=1).reshape(h, w), (H_full, W_full), anti_aliasing=True)
        mass_received = resize(transport_plan.sum(axis=0).reshape(h, w), (H_full, W_full), anti_aliasing=True)

        # Displacement field
        rows, cols   = np.meshgrid(np.arange(h), np.arange(w), indexing='ij')
        coords       = np.stack([rows.flatten(), cols.flatten()], axis=1).astype(np.float64)
        src_mass     = transport_plan.sum(axis=1, keepdims=True)
        src_mass     = np.where(src_mass == 0, 1e-10, src_mass)
        mean_dst     = (transport_plan @ coords) / src_mass
        displacement = (mean_dst - coords).reshape(h, w, 2)
        dy, dx       = displacement[..., 0], displacement[..., 1]
        magnitude    = np.sqrt(dx**2 + dy**2)

        # Mask to salient regions only
        mass_map  = transport_plan.sum(axis=1).reshape(h, w)
        mask      = mass_map > np.percentile(mass_map, 85)

        # --- Extent helpers ---
        lon0_e, lon1_e, lat0_e, lat1_e = extent

        def pixel_to_geo(row, col, h, w):
            """Convert downsampled pixel (row,col) → (lon, lat)"""
            lon = lon0_e + (col / w) * (lon1_e - lon0_e)
            lat = lat0_e + (row / h) * (lat1_e - lat0_e)
            return lon, lat

        # ── Figure: 2 rows ───────────────────────────────────────────────────────
        fig = plt.figure(figsize=(40, 22))

        proj = ccrs.PlateCarree()

        # Row 1: your original 4 panels
        ax0 = fig.add_subplot(2, 4, 1, projection=proj)
        ax1 = fig.add_subplot(2, 4, 2, projection=proj)
        ax2 = fig.add_subplot(2, 4, 3, projection=proj)
        ax3 = fig.add_subplot(2, 4, 4, projection=proj)

        # Row 2: OT panels
        ax4 = fig.add_subplot(2, 4, 5, projection=proj)   # clean gradient
        ax5 = fig.add_subplot(2, 4, 6, projection=proj)   # perturbed gradient
        ax6 = fig.add_subplot(2, 4, 7, projection=proj)   # mass sent vs received
        ax7 = fig.add_subplot(2, 4, 8, projection=proj)   # displacement field

        all_axes = [ax0, ax1, ax2, ax3, ax4, ax5, ax6, ax7]
        for ax in all_axes:
            ax.add_feature(cfeature.COASTLINE, linewidth=0.6)
            ax.add_feature(cfeature.BORDERS,   linewidth=0.6)

        kw_base = dict(origin="lower", extent=extent, transform=proj)

        # ── Row 1: original panels ───────────────────────────────────────────────
        for ax, data, cmap, lo, hi, title in [
            (ax0, inp,       "RdBu_r", None, None,  f"Input — {input_name} (t)"),
            (ax1, gt,        "RdBu_r", vmin, vmax,  f"Ground truth — {output_name} (t+{step+1})"),
            (ax2, pred_data, "RdBu_r", vmin, vmax,  f"Prediction (t+{step+1})"),
            (ax3, attr,      "RdBu_r", -1,   1,     f"Attribution — {method}"),
        ]:
            kw = {**kw_base, "cmap": cmap}
            if lo is not None:
                kw.update(vmin=lo, vmax=hi)
            im = ax.imshow(data, **kw)
            ax.set_title(title, fontsize=11, pad=6)
            cb = fig.colorbar(im, ax=ax, orientation="vertical", fraction=0.046, pad=0.04)

        # ── Row 2: OT panels ─────────────────────────────────────────────────────
            barycentre = barycentre/barycentre.max()

        # Panel 5 — Clean gradient
        im4 = ax4.imshow(clean_grad_2d, cmap="hot", **kw_base)
        ax4.set_title("Clean Gradient (saliency)", fontsize=11, pad=6)
        fig.colorbar(im4, ax=ax4, orientation="vertical", fraction=0.046, pad=0.04)

        # Panel 6 — Perturbed gradient
        im5 = ax5.imshow(grad_2d, cmap="hot", **kw_base)
        ax5.set_title("Perturbed Gradient (saliency)", fontsize=11, pad=6)
        fig.colorbar(im5, ax=ax5, orientation="vertical", fraction=0.046, pad=0.04)

        # Panel 7 — Mass sent (red) + received (green) overlaid
        ax6.imshow(mass_sent,     cmap="Reds",  alpha=0.6, **kw_base)
        im6 = ax6.imshow(mass_received, cmap="Greens", alpha=0.6, **kw_base)
        ax6.set_title("OT Mass Sent (red) / Received (green)", fontsize=11, pad=6)
        red_patch   = mpatches.Patch(color='red',   alpha=0.6, label='Mass sent (perturbed)')
        green_patch = mpatches.Patch(color='green', alpha=0.6, label='Mass received (clean)')
        ax6.legend(handles=[red_patch, green_patch], loc='lower left', fontsize=8)

        # Panel 8 — Displacement field with quiver in geo coordinates
        mag_full = resize(magnitude, (H_full, W_full), anti_aliasing=True)
        ax7.imshow(mag_full, cmap="plasma", **kw_base)
        ax7.set_title("OT Displacement Field\n(perturbation shifted saliency)", fontsize=11, pad=6)

        # Quiver in geographic coordinates
        step_q = 4  # quiver subsampling in downsampled space
        for r in range(0, h, step_q):
            for c in range(0, w, step_q):
                if not mask[r, c]:
                    continue
                lon_src, lat_src = pixel_to_geo(r,              c,              h, w)
                lon_dst, lat_dst = pixel_to_geo(r + dy[r, c],  c + dx[r, c],  h, w)
                ax7.annotate("",
                    xy=(lon_dst, lat_dst), xytext=(lon_src, lat_src),
                    xycoords=proj._as_mpl_transform(ax7),
                    textcoords=proj._as_mpl_transform(ax7),
                    arrowprops=dict(arrowstyle="->", color="cyan",
                                    lw=1.5, alpha=0.85),
                    annotation_clip=True,
                )

        # ── Target bounding box on all panels ───────────────────────────────────
        if target is not None:
            lon0, lon1, lat0, lat1 = target
            for ax in all_axes:
                ax.add_patch(Rectangle(
                    (lon0, lat0), lon1 - lon0, lat1 - lat0,
                    linewidth=2, edgecolor="black", facecolor="none",
                    transform=proj, zorder=5,
                ))

        fig.suptitle(
            f"Input: {input_name}  →  Output: {output_name}  |  {method}  +  Optimal Transport",
            fontsize=14, y=0.98,
        )
        fig.subplots_adjust(wspace=0.08, hspace=0.15)

        out_dir = os.path.join(fig_path, method)
        os.makedirs(out_dir, exist_ok=True)
        if save_figs:
            plt.savefig(os.path.join(out_dir, runtime), dpi=300, bbox_inches="tight")
            plt.close(fig)
        else:
            plt.show()
    def compute_different_explanations(self, input, batch_idx, checkpoint, cfg_model,
                                   cfg_dataset, cfg_xai, dataset_info, infer_ds,
                                   list_run_hour, use_old_weights, target,
                                   plot_explanations=True, extent=[-12, 16, 37.5, 55.4],
                                   return_output=False, fig_path=".figs/attributions"):
        """
        Compute WassersteinGrad attributions for multiple input variables.
        """

        # Get clean prediction output first
        output = predict_step(self.model, input, batch_idx, checkpoint, cfg_model,
                            cfg_dataset, cfg_xai, dataset_info, infer_ds,
                            list_run_hour, use_old_weights, compute_grads=False)

        if output is None:
            if return_output:
                return None, None
            return None

        # Initialize storage for all input variables
        grad_multiple = {explain_input: [] for explain_input in cfg_xai["multiple_explain"]}

        # Backup original input tensor
        backup_tensor = input.inputs.tensor.clone()

        # Process each input variable
        for explain_input in cfg_xai["multiple_explain"]:
            cfg_xai["explain"]["input"] = explain_input

            # Storage for normalized gradient maps across perturbations for this input
            grads_wasser = {step: [] for step in range(input.num_pred_steps)}
            grads_wasser_pos= {i:[] for i in range(input.num_pred_steps)}
            grads_wasser_neg= {i:[] for i in range(input.num_pred_steps)}
            input_idx = input.inputs.feature_names_to_idx[explain_input]
            input_clean = backup_tensor[..., input_idx]

            # Calculate perturbation scale
            vals = backup_tensor[..., input_idx]
            std_perturbation = self.std_perturbation * (vals.max() - vals.min())

            # Collect gradients over multiple perturbations
            for _ in range(self.num_perturbations):
                # Create perturbed input
                input.inputs.tensor = backup_tensor.clone()
                input_perturbed = input_clean.clone().detach() + torch.randn_like(input_clean) * std_perturbation
                input.inputs.tensor[..., input_idx] = input_perturbed

                # Compute gradient for this perturbation
                gradient, _ = self.base_explainer.compute_explanations(
                    input=input, batch_idx=batch_idx, checkpoint=checkpoint,
                    cfg_model=cfg_model, cfg_dataset=cfg_dataset, cfg_xai=cfg_xai,
                    dataset_info=dataset_info, infer_ds=infer_ds,
                    list_run_hour=list_run_hour, use_old_weights=use_old_weights,
                    target=target, plot_explanations=False, extent=extent,
                    fig_path=fig_path, return_output=True
                )

                if gradient is None:
                    input.inputs.tensor = backup_tensor
                    if return_output:
                        return None, None
                    return None

                # Accumulate normalized gradient maps for each prediction step
                for step_idx in range(len(gradient)):
                    grads_wasser[step_idx].append(
                        normalize_sum(gradient[step_idx].reshape(512, 640))
                    )
                    # grads_wasser_pos[step_idx].append(normalize_sum(torch.clamp(gradient[step_idx].reshape(512,640),min=0)))
                    # grads_wasser_neg[step_idx].append(normalize_sum(torch.clamp(gradient[step_idx].reshape(512,640),max=0)))

            # Aggregate via Wasserstein barycenter for each prediction step
            wasserstein_list_for_input = []
            A_maps = torch.stack(grads_wasser[0], dim =0)
            # A_maps_neg = torch.stack(grads_wasser_neg[0], dim =0)

            # A_maps_pos = torch.stack(grads_wasser_pos[0], dim =0)
            # barycentre_pos= ot.bregman.convolutional_barycenter2d(A_maps_pos, reg=0.001)
            # barycentre_pos = barycentre_pos/barycentre_pos.max()
            # barycentre_neg= ot.bregman.convolutional_barycenter2d(A_maps_neg, reg=0.001)
            # barycentre_neg = -barycentre_neg/barycentre_neg.max()
            
            barycentre= ot.bregman.convolutional_barycenter2d(A_maps, reg=0.001)

            # barycentre = barycentre_pos-barycentre_neg
            # barycentre = torch.tensor(barycentre / barycentre.max())
            wasserstein_list_for_input.append(barycentre)

            grad_multiple[explain_input] = wasserstein_list_for_input
            torch.cuda.empty_cache()

        # Restore original input
        input.inputs.tensor = backup_tensor

        if return_output:
            return grad_multiple, output
        return grad_multiple


import os
import glob
import numpy as np
import torch
import ot
import concurrent.futures

from py4castxai.explainers.XGrad import XGrad
from py4castxai.explainers.inputxgrad import InputxGrad
from py4castxai.explainers.basegrad import BaseGrad
from py4castxai.utils_test.infer_utils import predict_step


# def normalize_sum(tensor):
#     """Normalizes a tensor into a probability distribution summing to 1."""
#     tensor_abs = torch.abs(tensor)
#     return tensor_abs / (torch.sum(tensor_abs) + 1e-12)


class EDAWassersteinGrad(XGrad):
    """
    EDA-guided WassersteinGrad explainer.
    Injects flow-dependent physical perturbations across ALL input variables using PEARO,
    and aggregates the resulting gradient distributions using a Convolutional Wasserstein Barycenter
    to preserve spatial geometry without smearing.
    """

    def __init__(self, 
                 model, 
                 base_explainer="BaseGrad", 
                 eda_dir=None, 
                 eda_perturbations=None, 
                 scale_factor=1.0, 
                 num_members=17,
                 reg=0.001):
        super().__init__(model)
        if base_explainer == "InputxGrad":
            self.base_explainer = InputxGrad(model)
        else:
            self.base_explainer = BaseGrad(model)
        
        self.eda_dir = eda_dir
        self.eda_perturbations = eda_perturbations
        self.scale_factor = scale_factor
        self.num_members = num_members
        self.reg = reg
        print(f"Lambda Sinkhorn (reg) = {self.reg}")

    def _map_feature_to_filename(self, py4cast_name):
        special_maps = {
            "aro_t2m": "t2m",       "aro_t2m_2m": "t2m",
            "aro_r2": "hu2m",       "aro_r2_2m": "hu2m",
            "aro_tp": "rr",         "aro_tp_0m": "rr",
            "aro_u10": "u10m",      "aro_u10_10m": "u10m",
            "aro_v10": "v10m",      "aro_v10_10m": "v10m"
        }
        if py4cast_name in special_maps:
            return special_maps[py4cast_name]
        return py4cast_name.replace("aro_", "").replace("_", "").replace("hpa", "")


    def _load_pearo_eda_from_disk(self, eda_dir, timestamp_str, feature_names_to_idx,
                              target_tensor_shape, device, subdomain=[100, 612, 240, 880]):
        """
        Loads PEARO .npy files containing all 17 members.
        Crops to subdomain and permutes dimensions to match PyTorch shape.
        Caches the assembled tensor to disk for fast reloads.
        """
        B, T, Y_dim, X_dim, C = target_tensor_shape

        # --- Cache lookup ---
        # Key the cache on everything that changes the output: timestamp, shape, subdomain,
        # and the exact set + order of variables.
        cache_dir = os.path.join(eda_dir, ".eda_cache")
        cache_key = f"{timestamp_str}_{B}x{T}x{Y_dim}x{X_dim}x{C}_{subdomain}"
        cache_dir = os.path.join(cache_dir, f"{cache_key}.pt")
        os.makedirs(cache_dir, exist_ok=True)
        cache_path = os.path.join(cache_dir, f"{cache_key}.pt")

        if os.path.exists(cache_path):
            # weights_only=True is safe here (plain tensor) and faster/safer to load
            return torch.load(cache_path, map_location=device, weights_only=True)
        # --- Cache miss: build from raw .npy files ---
        full_eda_members = torch.zeros(
            (self.num_members, B, T, Y_dim, X_dim, C), dtype=torch.float32
        )  # build on CPU, transfer once at the end

        if subdomain is not None:
            y_min, y_max, x_min, x_max = subdomain

        for var_name, var_idx in feature_names_to_idx.items():
            pearo_file_var = self._map_feature_to_filename(var_name)
            file_path = os.path.join(eda_dir, f"{timestamp_str}_{pearo_file_var}_17.npy")

            if os.path.exists(file_path):
                # mmap so only the cropped slice is actually read from disk
                data_np = np.load(file_path, mmap_mode="r")

                if subdomain is not None:
                    data_np = data_np[y_min:y_max, x_min:x_max, :, :]

                # (Lat, Lon, Time, Members) -> (Members, Time, Lat, Lon)
                data_np = np.ascontiguousarray(np.transpose(data_np, (3, 2, 0, 1)))
                data_tensor = torch.from_numpy(data_np)

                time_steps_in_file = data_tensor.shape[1]
                if time_steps_in_file == T:
                    full_eda_members[:, :, :, :, :, var_idx] = data_tensor.unsqueeze(1)
                elif time_steps_in_file == 1:
                    full_eda_members[:, :, -1:, :, :, var_idx] = data_tensor.unsqueeze(1)
            else:
                print(f"Warning: EDA file {file_path} not found. Channel will be zeroes.")

        # --- Write cache (save CPU tensor; move to device on return) ---
        torch.save(full_eda_members, cache_path)

        return full_eda_members.to(device)
    def compute_explanations(self,
                             input, batch_idx, checkpoint, cfg_model, cfg_dataset,
                             cfg_xai, dataset_info, infer_ds, list_run_hour, use_old_weights,
                             target, plot_explanations=True,
                             extent=[-12, 16, 37.5, 55.4],
                             fig_path=".figs/attributions", return_output=False,
                             eda_dir=None, eda_perturbations=None):
        
        output_idx = input.outputs.feature_names_to_idx[cfg_xai["explain"]["output"]]
        grads_wasser = {i: [] for i in range(input.num_pred_steps)}

        output = predict_step(
            self.model, input, batch_idx, checkpoint, cfg_model, cfg_dataset, 
            cfg_xai, dataset_info, infer_ds, list_run_hour, use_old_weights, compute_grads=False
        )
        
        if output is None:
            if return_output: return None, None
            return

        is_forcing = cfg_xai["explain"]["forcing_input"]
        target_obj = input.forcing if is_forcing else input.inputs
        backup_tensor = target_obj.tensor.clone()
        device = backup_tensor.device
        
        input_var_name = cfg_xai["explain"]["input"]
        target_var_idx = target_obj.feature_names_to_idx[input_var_name]

        # Foolproof Subdomain Search
        def _find_subdomain(d):
            if isinstance(d, dict):
                if "subdomain" in d: return d["subdomain"]
                for k, v in d.items():
                    res = _find_subdomain(v)
                    if res is not None: return res
            return None
            
        grid_subdomain = _find_subdomain(cfg_dataset)

        active_eda_dir = eda_dir if eda_dir is not None else self.eda_dir
        active_eda_tensor = eda_perturbations if eda_perturbations is not None else self.eda_perturbations

        if active_eda_tensor is not None:
            raw_eda_states = active_eda_tensor.to(device)
            if grid_subdomain is not None and raw_eda_states.shape[3:5] != backup_tensor.shape[2:4]:
                 y_min, y_max, x_min, x_max = grid_subdomain
                 raw_eda_states = raw_eda_states[:, :, :, y_min:y_max, x_min:x_max, :]
                 
        elif active_eda_dir is not None:
            batch_size = input.inputs.tensor.shape[0]
            idx_samples = [batch_idx * batch_size + b for b in range(batch_size)]
            sample = infer_ds.sample_list[idx_samples[0]]
            timestamp_str = sample.timestamps.datetime.strftime("%Y-%m-%dT%H:%M:%SZ")
            
            raw_eda_states = self._load_pearo_eda_from_disk(
                eda_dir=active_eda_dir,
                timestamp_str=timestamp_str,
                feature_names_to_idx=target_obj.feature_names_to_idx,
                target_tensor_shape=backup_tensor.shape,
                device=device
            )
        else:
            raise ValueError("Either `eda_dir` path or `eda_perturbations` tensor must be provided.")

        # Compute Zero-Mean Physical Deviations
        eda_mean = raw_eda_states.mean(dim=0, keepdim=True)
        zero_mean_perturbations = raw_eda_states - eda_mean
        num_perturbations = zero_mean_perturbations.shape[0]

        # --- Perturbation Loop ---
        for perturbation_idx in range(num_perturbations): 
            epsilon_k = zero_mean_perturbations[perturbation_idx] * self.scale_factor
            input_perturbed = backup_tensor.clone().detach() + epsilon_k
            
            if is_forcing:
                input.forcing.tensor = input_perturbed
            else:
                input.inputs.tensor = input_perturbed

            gradient, _ = self.base_explainer.compute_explanations(
                input=input, batch_idx=batch_idx, checkpoint=checkpoint,
                cfg_model=cfg_model, cfg_dataset=cfg_dataset, cfg_xai=cfg_xai,
                dataset_info=dataset_info, infer_ds=infer_ds, list_run_hour=list_run_hour,
                use_old_weights=use_old_weights, target=target, plot_explanations=False,
                extent=extent, fig_path=fig_path, return_output=True
            )

            if gradient is None:
                if return_output: return None, None
                return

            for i in range(len(gradient)):
                # Extract the gradient for the target variable channel
                grad_step = gradient[i]
                if grad_step.shape[-1] > 1 and grad_step.shape[-1] == backup_tensor.shape[-1]:
                    grad_target_var = grad_step[..., target_var_idx]
                else:
                    grad_target_var = grad_step.squeeze(-1)
                
                # Normalize to a probability distribution over the 2D grid
                # Dynamic spatial extraction replaces hardcoded (512, 640)
                H, W = grad_target_var.shape[-2:]
                grad_norm = normalize_sum(grad_target_var.reshape(H, W))
                
                grads_wasser[i].append(grad_norm)

            self.model.zero_grad(set_to_none=True)

        # Restore clean inputs
        if is_forcing:
            input.forcing.tensor = backup_tensor
        else:
            input.inputs.tensor = backup_tensor

        # --- Wasserstein Aggregation ---
        wasserstein_list = []
        num_steps = 1 if cfg_xai["explain"]["end_to_end"] else input.num_pred_steps

        for i in range(num_steps):
            # Stack the normalized gradient distributions. Shape: (num_perturbations, H, W)
            A_maps = torch.stack(grads_wasser[i], dim=0)
            
            # Compute the Convolutional Wasserstein Barycenter
            # POT converts PyTorch tensors to NumPy arrays internally if configured, 
            # but explicit conversion ensures stability.
            # A_maps_np = A_maps.cpu().numpy()
            barycentre = ot.bregman.convolutional_barycenter2d(A_maps, reg=self.reg)
            
            # Normalize the resulting barycenter back to [0, 1] for visualization
            barycentre = barycentre.to(device=device, dtype=torch.float32)
            # barycentre = barycentre / (barycentre.max() + 1e-12)
            
            wasserstein_list.append(barycentre)

        # --- Visualization ---
        if plot_explanations:
            batch_size = input.inputs.tensor.shape[0]
            idx_samples = [batch_idx * batch_size + b for b in range(batch_size)]
            samples = [infer_ds.sample_list[idx] for idx in idx_samples]
            runtimes = [sample.timestamps.datetime.strftime("%Y%m%d%H") for sample in samples]
            stats = dataset_info.stats

            if cfg_xai["explain"]["end_to_end"]:
                step = input.num_pred_steps
                fig_path_step = fig_path + f"/end_to_end/{step}_step/input_{cfg_xai['explain']['input']}_output_{cfg_xai['explain']['output']}"
                self.plot_explanations(
                    wasserstein_list[0], input, output, extent, "EDAWassersteinGrad", fig_path_step,
                    runtime=runtimes[0], stats=stats, step=step-1,
                    input_name=input_var_name, output_name=cfg_xai["explain"]["output"],
                    input_idx=target_var_idx, output_idx=output_idx, target=cfg_xai["target"],
                    forcing=is_forcing, save_figs=cfg_xai["explain"]["exp_plot"]["save_figs"]
                )
            else:
                for i in range(len(wasserstein_list)):
                    fig_path_step = fig_path + f"/{i+1}_step/input_{cfg_xai['explain']['input']}_output_{cfg_xai['explain']['output']}"
                    self.plot_explanations(
                        wasserstein_list[i], input, output, extent, "EDAWassersteinGrad", fig_path_step,
                        runtime=runtimes[0], stats=stats, step=i,
                        input_name=input_var_name, output_name=cfg_xai["explain"]["output"],
                        input_idx=target_var_idx, output_idx=output_idx, target=cfg_xai["target"],
                        forcing=is_forcing, save_figs=cfg_xai["explain"]["exp_plot"]["save_figs"]
                    )

        if return_output:
            return wasserstein_list, output         
        
        return wasserstein_list
    

    def plot_explanations(self, explanations, input, pred, extent, method,fig_path,runtime,stats,step=0,input_name = "aro_tp_0m",
                          output_name="aro_tp_0m",input_idx = 0,output_idx = 0,target=None,forcing=None,save_figs=True):
        
        # --- Extract data ---
        
              # attr = explanations[0,...,0,0].detach().cpu().numpy()  # [H, W]
        attr = explanations.detach().cpu().numpy()  # [H, W]

        if forcing:
            inp = input.forcing.tensor[0, step, :, :, input_idx].detach().cpu()
        else:
            if step == 0:

                inp = input.inputs.tensor[0, 0, :, :, input_idx].detach().cpu()

            else:
                inp = pred.tensor[0, step-1, :, :, input_idx].detach().cpu()
       


        pred = pred.tensor[0, step, :, :, output_idx].detach().cpu().numpy()
        
        gt  = input.outputs.tensor[0, step, :, :, output_idx].detach().cpu()
        
        
     
        means_input = torch.asarray(stats[input_name]["mean"])
        std_input = torch.asarray(stats[input_name]["std"])

        means_output = torch.asarray(stats[output_name]["mean"])
        std_output = torch.asarray(stats[output_name]["std"])

            
        inp *= std_input
        inp += means_input
        
        gt *=std_output
        gt += means_output

        inp = inp.numpy()
        gt = gt.numpy()

        # --- Normalize attribution for visualization only ---
        attr = normalize_expl_expl(attr)

       
        vmin = min(gt.min(), pred.min())
        vmax = max(gt.max(), pred.max())
        # from scipy.ndimage import laplace,gaussian_filter
        # z_250 = gaussian_filter(inp,sigma=20)
        # laplace_z = laplace(z_250)
        # mask = np.percentile(laplace_z,40)
        # laplace_z = np.where(laplace_z<mask,laplace_z,0)
              # ── 4. Figure ────────────────────────────────────────────────
        fig, axes = plt.subplots(
            1, 4,
            figsize=(40, 10),
            subplot_kw={"projection": ccrs.PlateCarree()},
        )
        fig.subplots_adjust(wspace=0.08, top=0.88)

        for ax in axes:
            ax.add_feature(cfeature.COASTLINE, linewidth=0.6)
            ax.add_feature(cfeature.BORDERS,   linewidth=0.6)

        panel_cfg = [
            (axes[0],inp,  "RdBu_r", None, None, f"Input  —  {input_name}  (t)"),
            (axes[1], gt,   "RdBu_r", vmin, vmax, f"Ground truth  —  {output_name}  (t+{step+1})"),
            (axes[2], pred, "RdBu_r", vmin, vmax, f"Prediction  (t+{step+1})"),
            (axes[3], attr,     "RdBu_r", -1,   1,    f"Attribution  —  {method}"), # if return_output:

        #     return  wasserstein_list, output         
        
        ]

        for ax, data, cmap, lo, hi, title in panel_cfg:
            kw = dict(origin="lower", cmap=cmap, extent=extent,
                      transform=ccrs.PlateCarree())
            if lo is not None:
                kw.update(vmin=lo, vmax=hi)
            im = ax.imshow(data, **kw)
            ax.set_title(title, fontsize=11, pad=6)
            cb = fig.colorbar(im, ax=ax, orientation="vertical",
                              fraction=0.046, pad=0.04)
            if ax is axes[3]:
                cb.set_label("Normalised attribution", fontsize=9)

        # ── 5. Optional target bounding box ─────────────────────────
        if target is not None:
            lon0, lon1, lat0, lat1 = target
            for ax in axes:
                ax.add_patch(Rectangle(
                    (lon0, lat0), lon1 - lon0, lat1 - lat0,
                    linewidth=2, edgecolor="black", facecolor="none",
                    transform=ccrs.PlateCarree(), zorder=5,
                ))

        # ── 6. Title, save / show ────────────────────────────────────
        fig.suptitle(
            f"Input: {input_name}   →   Output: {output_name}   |   {method}",
            fontsize=13, y=0.97,
        )

        out_dir = os.path.join(fig_path, method)
        os.makedirs(out_dir, exist_ok=True)

        if save_figs:
            plt.savefig(os.path.join(out_dir, runtime), dpi=300, bbox_inches="tight")
            plt.close(fig)
        else:
            plt.show()

class WassersteinGradMask(XGrad):


    def __init__(self,model,base_explainer="BaseGrad",std_perturbations=0.01,num_perturbations=50,reg=0.001):
        super().__init__(model)
        if base_explainer == "InputxGrad":
            self.base_explainer = InputxGrad(model)
        else:
            self.base_explainer = BaseGrad(model)
        
        self.std_perturbation = std_perturbations
        self.num_perturbations = num_perturbations
        self.reg = reg
        print("Lambda sinkhorn = ",self.reg)


    def compute_explanations(self,
                             input,batch_idx,checkpoint,cfg_model,cfg_dataset,
                             cfg_xai,dataset_info,infer_ds,list_run_hour,use_old_weights,
                             target,plot_explanations = True,
                             extent=[-12, 16, 37.5, 55.4],
                             fig_path=".figs/attributions",return_output = False):
        #print(input.forcing)
        
        output_idx = input.outputs.feature_names_to_idx[cfg_xai["explain"]["output"]]

         
        grads = {i:[] for i in range(input.num_pred_steps)}
        grads_wasser= {i:[] for i in range(input.num_pred_steps)}
        grads_wasser_pos= {i:[] for i in range(input.num_pred_steps)}
        grads_wasser_neg= {i:[] for i in range(input.num_pred_steps)}
        transport_plans= {i:[] for i in range(input.num_pred_steps)}
        output = predict_step(self.model,input,batch_idx,checkpoint,cfg_model,cfg_dataset,cfg_xai,dataset_info,infer_ds,list_run_hour,use_old_weights,compute_grads=False)
        
        
        # if output is None:
        #     if return_output:
        #         return None,None
        #     return
        
        if cfg_xai["explain"]["forcing_input"]:
            backup_tensor = input.forcing.tensor
            input_idx = input.forcing.feature_names_to_idx[cfg_xai["explain"]["input"]]
            input_clean = input.forcing.tensor[...,input_idx]
            vals = input.forcing.tensor[..., input_idx]
            std_perturbation = self.std_perturbation * (vals.max() - vals.min())   
            
            for _ in range(self.num_perturbations): 
                
                # input.inputs.tensor =input_clean+ torch.randn_like(input.inputs.tensor)*std_perturbation
                input_perturbed = input_clean.clone().detach() + torch.randn_like(input_clean) * std_perturbation
                input.forcing.tensor = input.forcing.tensor.clone()
                input.forcing.tensor[..., input_idx] = input_perturbed
                gradient,_ = self.base_explainer.compute_explanations(input=input,batch_idx=batch_idx,checkpoint=checkpoint,cfg_model=cfg_model,cfg_dataset=cfg_dataset,
                                                                      cfg_xai=cfg_xai,dataset_info=dataset_info,infer_ds=infer_ds,list_run_hour=list_run_hour,use_old_weights=use_old_weights,
                                                                      target=target,plot_explanations=False,extent=extent,fig_path=fig_path,return_output=True)

                if gradient is None:
                    if return_output:
                        return None,None
                    return
                
                for i in range(len(gradient)):
                    grads[i].append(gradient[i])
                
                self.model.zero_grad(set_to_none=True)

                
        else:
            backup_tensor = input.inputs.tensor

            input_idx = input.inputs.feature_names_to_idx[cfg_xai["explain"]["input"]]
            input_clean = input.inputs.tensor[...,input_idx]

            vals = input.inputs.tensor[..., input_idx]
            std_perturbation = self.std_perturbation * (vals.max() - vals.min())            
            clean_gradient,_ = self.base_explainer.compute_explanations(input=input,batch_idx=batch_idx,checkpoint=checkpoint,cfg_model=cfg_model,cfg_dataset=cfg_dataset,
                                                                        cfg_xai=cfg_xai,dataset_info=dataset_info,infer_ds=infer_ds,list_run_hour=list_run_hour,use_old_weights=use_old_weights,
                                                                        target=target,plot_explanations=False,extent=extent,fig_path=fig_path,return_output=True)

            for k in range(self.num_perturbations): 
                
                # input.inputs.tensor =input_clean+ torch.randn_like(input.inputs.tensor)*std_perturbation
                input_perturbed = input_clean.clone().detach() + torch.randn_like(input_clean) * std_perturbation
                input.inputs.tensor = input.inputs.tensor.clone()
                input.inputs.tensor[..., input_idx] = input_perturbed
                gradient,_ = self.base_explainer.compute_explanations(input=input,batch_idx=batch_idx,checkpoint=checkpoint,cfg_model=cfg_model,cfg_dataset=cfg_dataset,cfg_xai=cfg_xai,dataset_info=dataset_info,infer_ds=infer_ds,list_run_hour=list_run_hour,use_old_weights=use_old_weights,target=target,plot_explanations=False,extent=extent,fig_path=fig_path,return_output=True)

                if gradient is None:
                    if return_output:
                        return None,None
                    return
                
                for i in range(len(gradient)):
              
                    grads_wasser[i].append(normalize_sum(gradient[i].reshape(512,640)))

                    
        grad_list = []
        wasserstein_list = []
        if not cfg_xai["explain"]["end_to_end"]:

            for i in range(input.num_pred_steps):
                
                A_maps = torch.stack(grads_wasser[i], dim =0)
                
            
                barycentre = ot.bregman.convolutional_barycenter2d(A_maps, reg=self.reg)
                # weighted_barycentre = clean_gradient[0].reshape(512,640) * barycentre

        
                wasserstein_list.append(weighted_barycentre)
            

        else:
            A_maps = torch.stack(grads_wasser[0], dim =0)
                
            barycentre = ot.bregman.convolutional_barycenter2d(A_maps, reg=self.reg)
            # barycentre = barycentre/barycentre.max()
            weighted_barycentre = clean_gradient[0].reshape(512,640) * barycentre
            
            wasserstein_list.append(weighted_barycentre)

        if cfg_xai["explain"]["forcing_input"]:
                input.forcing.tensor = backup_tensor
        else:

                input.inputs.tensor = backup_tensor
        if return_output:

            return  wasserstein_list, output         
        

        return wasserstein_list
    def compute_different_explanations(self, input, batch_idx, checkpoint, cfg_model,
                                   cfg_dataset, cfg_xai, dataset_info, infer_ds,
                                   list_run_hour, use_old_weights, target,
                                   plot_explanations=True, extent=[-12, 16, 37.5, 55.4],
                                   return_output=False, fig_path=".figs/attributions"):
        """
        Compute WassersteinGrad attributions for multiple input variables.
        """

        # Get clean prediction output first
        output = predict_step(self.model, input, batch_idx, checkpoint, cfg_model,
                            cfg_dataset, cfg_xai, dataset_info, infer_ds,
                            list_run_hour, use_old_weights, compute_grads=False)

        if output is None:
            if return_output:
                return None, None
            return None

        # Initialize storage for all input variables
        grad_multiple = {explain_input: [] for explain_input in cfg_xai["multiple_explain"]}

        # Backup original input tensor
        backup_tensor = input.inputs.tensor.clone()

        # Process each input variable
        for explain_input in cfg_xai["multiple_explain"]:
            cfg_xai["explain"]["input"] = explain_input
            clean_gradient,_ = self.base_explainer.compute_explanations(input=input,batch_idx=batch_idx,checkpoint=checkpoint,cfg_model=cfg_model,cfg_dataset=cfg_dataset,
                                                                        cfg_xai=cfg_xai,dataset_info=dataset_info,infer_ds=infer_ds,list_run_hour=list_run_hour,use_old_weights=use_old_weights,
                                                                        target=target,plot_explanations=False,extent=extent,fig_path=fig_path,return_output=True)

            # Storage for normalized gradient maps across perturbations for this input
            grads_wasser = {step: [] for step in range(input.num_pred_steps)}
            grads_wasser_pos= {i:[] for i in range(input.num_pred_steps)}
            grads_wasser_neg= {i:[] for i in range(input.num_pred_steps)}
            input_idx = input.inputs.feature_names_to_idx[explain_input]
            input_clean = backup_tensor[..., input_idx]

            # Calculate perturbation scale
            vals = backup_tensor[..., input_idx]
            std_perturbation = self.std_perturbation * (vals.max() - vals.min())
            
            # Collect gradients over multiple perturbations
            for _ in range(self.num_perturbations):
                # Create perturbed input
                input.inputs.tensor = backup_tensor.clone()
                input_perturbed = input_clean.clone().detach() + torch.randn_like(input_clean) * std_perturbation
                input.inputs.tensor[..., input_idx] = input_perturbed

                # Compute gradient for this perturbation
                gradient, _ = self.base_explainer.compute_explanations(
                    input=input, batch_idx=batch_idx, checkpoint=checkpoint,
                    cfg_model=cfg_model, cfg_dataset=cfg_dataset, cfg_xai=cfg_xai,
                    dataset_info=dataset_info, infer_ds=infer_ds,
                    list_run_hour=list_run_hour, use_old_weights=use_old_weights,
                    target=target, plot_explanations=False, extent=extent,
                    fig_path=fig_path, return_output=True
                )

                if gradient is None:
                    input.inputs.tensor = backup_tensor
                    if return_output:
                        return None, None
                    return None

                # Accumulate normalized gradient maps for each prediction step
                for step_idx in range(len(gradient)):
                    grads_wasser[step_idx].append(
                        normalize_sum(gradient[step_idx].reshape(512, 640))
                    )
                    # grads_wasser_pos[step_idx].append(normalize_sum(torch.clamp(gradient[step_idx].reshape(512,640),min=0)))
                    # grads_wasser_neg[step_idx].append(normalize_sum(torch.clamp(gradient[step_idx].reshape(512,640),max=0)))

            # Aggregate via Wasserstein barycenter for each prediction step
            wasserstein_list_for_input = []
            A_maps = torch.stack(grads_wasser[0], dim =0)
            # A_maps_neg = torch.stack(grads_wasser_neg[0], dim =0)

            # A_maps_pos = torch.stack(grads_wasser_pos[0], dim =0)
            # barycentre_pos= ot.bregman.convolutional_barycenter2d(A_maps_pos, reg=0.001)
            # barycentre_pos = barycentre_pos/barycentre_pos.max()
            # barycentre_neg= ot.bregman.convolutional_barycenter2d(A_maps_neg, reg=0.001)
            # barycentre_neg = -barycentre_neg/barycentre_neg.max()
            
            barycentre= ot.bregman.convolutional_barycenter2d(A_maps, reg=0.001)

            # barycentre = barycentre_pos-barycentre_neg
            # barycentre = torch.tensor(barycentre / barycentre.max())
            wasserstein_list_for_input.append(barycentre*clean_gradient[0].reshape(512,640))

            grad_multiple[explain_input] = wasserstein_list_for_input
            torch.cuda.empty_cache()

        # Restore original input
        input.inputs.tensor = backup_tensor

        if return_output:
            return grad_multiple, output
        return grad_multiple
