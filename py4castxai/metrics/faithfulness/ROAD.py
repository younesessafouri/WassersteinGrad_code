
# from py4castxai.explainers import SmoothGrad, BaseGrad,InputxGrad,BaseGradTV
# import numpy as np
# import torch
# from py4castxai.utils_test.infer_utils import predict_step
# from scipy.sparse import lil_matrix, csc_matrix
# from scipy.sparse.linalg import spsolve
# import torch.nn.functional as F
# import matplotlib.pyplot as plt
# import os
# import copy
# from py4castxai.explainers import explainers_registry
# from typing import Dict, List, Optional, Tuple, Any
# import matplotlib.pyplot as plt
# import cartopy.crs as ccrs
# import cartopy.feature as cfeature


# def _normalize(tensor, stats, name):
#     mean = torch.as_tensor(stats[name]["mean"], dtype=tensor.dtype)
#     std  = torch.as_tensor(stats[name]["std"],  dtype=tensor.dtype)
#     return (tensor - mean) / std  # stays as tensor throughout

# def normalize_expl(a,abs=True):
            
#             if isinstance(a, np.ndarray):
#                 a = torch.from_numpy(a).to(torch.float32)
#             if abs:
#                 return a.abs() / (a.abs().max() + 1e-8)
#             return a / (a.abs().max() + 1e-8)

# def get_random_mask(explain_flat, top_k):
#     B, N = explain_flat.shape
#     idx = torch.stack([
#         torch.randperm(N, device=explain_flat.device)[:top_k]
#         for _ in range(B)
#     ])
#     return idx


# def linear_imputation(x, mask,noise_std=0.1):
#     """ Apply linear interpolation + Gaussian noise to perturb masked regions. Used to simulate 'feature removal' in ROAD. """ 
#     kernel = torch.ones(1, 1, 3, 3, device=x.device) / 9.0 
#     if x.ndim == 4: # [B, F, H, W] 
#         kernel = kernel.expand(x.shape[1], 1, 3, 3)
#         neighbor_mean = F.conv2d(x, kernel, padding=1, groups=x.shape[1]) 
#     else:
#         neighbor_mean = F.conv2d(x.unsqueeze(1), kernel, padding=1).squeeze(1) 
#     # # Combine original + interpolated with noise 
#     noise = noise_std * torch.randn_like(x) 
#     x_perturbed = x * (1 - mask) + (neighbor_mean + noise) * mask
#     return x_perturbed


# class ROAD:
#     """
#     ROAD (Remove And Debias) — Faithfulness Evaluation Metric

#     Measures how model predictions degrade when the most important
#     features (as indicated by the attribution map) are progressively
#     removed from the input.

    
#     Parameters
#     ----------
#     model : torch.nn.Module
#         Trained model to evaluate.
#     percentages : dict
#         Range of percentages {'start': int, 'stop': int, 'step': int}
#         indicating how many top features to remove.
#     explainer : str
#         Explanation method ('SmoothGrad', 'InputxGrad', or 'BaseGrad').
#     std_noise : float
#         Standard deviation of Gaussian noise added during imputation.
#     **expl_params : dict
#         Additional parameters for the chosen explainer.
#     """

#     def __init__(self, model, percentages,explainer="BaseGrad", std_noise = 0.1,**expl_params): 
#         self.model = model
#         self.std_noise = std_noise
#         # self.percentages = percentages

#         # self.percentages = list(range(percentages["start"], percentages["stop"], percentages["step"]))

#         self.percentages = [i for i in range(1,10)] + list(range(10,50,4))
#         self.explainer = self.get_explainer(explainer_name=explainer,params=expl_params)

#         self.explainer_name = explainer
#         self.max_function = np.nanmax


#     def get_explainer(self, explainer_name: str, params: Dict):
#             """Initialize explainer by name."""
#             explainer_cls = explainers_registry.get(explainer_name)
#             self.explainer_name = explainer_name
#             if explainer_cls is None:
#                 raise ValueError(f"Unknown explainer: {explainer_name}")
            
#             return explainer_cls(self.model, **params)
#     def evaluate(self, dataloader, checkpoint,
#                  cfg_model, cfg_dataset,cfg_xai, dataset_info,
#                  infer_ds, list_run_hour, use_old_weights, target,plot_explanations,extent,fig_path):

#         """
#         Evaluate ROAD faithfulness metric on the provided dataloader.

#         Steps:
#         1. Compute base prediction f(x)
#         2. Compute explanation map a(x)
#         3. For each percentage p:
#            - Remove top-p% of features using linear imputation
#            - Compute degraded prediction f(x_pert)
#            - Record degradation MSE(f(x) - f(x_pert))
#         4. Compute mean/std degradation per p and AUC of ROAD curve.

#         Returns
#         -------
#         mean_curve : dict[int, float]
#             Mean degradation across batches for each percentage.
#         std_curve : dict[int, float]
#             Standard deviation across batches for each percentage.
#         auc : float
#             Normalized area under degradation curve (faithfulness score).
#         """
#         if  cfg_xai["explain"]["end_to_end"]:
#             all_scores = {0:{p : [] for p in self.percentages}}
#             auc_list = {0:[]}
#         else:
#             auc_list = {i:[] for i in range(int(cfg_dataset["num_pred_steps_val_test"])) }
#             all_scores = {i:{p : [] for p in self.percentages} for i in range(int(cfg_dataset["num_pred_steps_val_test"])) }
        
#         device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#         os.makedirs(cfg_xai["eval"]["eval_plot"]["fig_path"]+"/faithfulness",exist_ok=True)
#         # import numpy as np
#         # import matplotlib.pyplot as plt
        
#         # all_precip_means = []

#         # for batch_idx, batch in enumerate(dataloader):
#         #     output_idx = batch.outputs.feature_names_to_idx[cfg_xai["explain"]["output"]]
#         #     extent = cfg_xai["extent"]
                    
#         #     lon0,lon1 = target[0], target[1]
#         #     lat0,lat1 = target[2], target[3]
#         #     H, W = batch.inputs.tensor.shape[2:4]
#         #     lat_min, lat_max = extent[2], extent[3]
#         #     lon_min, lon_max = extent[0], extent[1]

#         #     # Compute grid spacing
#         #     lat_step = (lat_max - lat_min) / H
#         #     lon_step = (lon_max - lon_min) / W
        
        
#         #     i_min = int((lat0-lat_min)/lat_step)
#         #     i_max =  int((lat1-lat_min)/lat_step)
#         #     j_min = int ((lon0-lon_min)/lon_step)
#         #     j_max = int((lon1-lon_min)/lon_step)
#         #     # ROI mean precipitation for this sample
#         #     roi_mean = batch.outputs.tensor[
#         #         :, :, i_min:i_max, j_min:j_max, output_idx
#         #     ].mean().item()
            
#         #     all_precip_means.append(roi_mean)

#         # all_precip_means = np.array(all_precip_means)

#         # # Plot the distribution
#         # fig, axes = plt.subplots(1, 2, figsize=(12, 4))

#         # axes[0].hist(all_precip_means, bins=100)
#         # axes[0].set_xlabel("Mean ROI precipitation")
#         # axes[0].set_ylabel("Sample count")
#         # axes[0].set_title("Full distribution")

#         # axes[1].hist(all_precip_means, bins=100)
#         # axes[1].set_yscale("log")
#         # axes[1].set_xlabel("Mean ROI precipitation")
#         # axes[1].set_title("Log scale — reveals the tail")

#         # print(f"Total samples:     {len(all_precip_means)}")
#         # print(f"Min:               {all_precip_means.min():.6f}")
#         # print(f"Max:               {all_precip_means.max():.6f}")
#         # print(f"Mean:              {all_precip_means.mean():.6f}")
#         # print(f"Median:            {np.median(all_precip_means):.6f}")
#         # print(f"75th percentile:   {np.percentile(all_precip_means, 75):.6f}")
#         # print(f"90th percentile:   {np.percentile(all_precip_means, 90):.6f}")
#         # print(f"95th percentile:   {np.percentile(all_precip_means, 95):.6f}")

#         # plt.tight_layout()
#         # plt.show()
#         with torch.autograd.set_grad_enabled(True):
            
#             for batch_idx, batch in enumerate(dataloader):
#                 extent = cfg_xai["extent"]
                        
#                 lon0,lon1 = target[0], target[1]
#                 lat0,lat1 = target[2], target[3]
#                 H, W = batch.inputs.tensor.shape[2:4]
#                 lat_min, lat_max = extent[2], extent[3]
#                 lon_min, lon_max = extent[0], extent[1]

#                 # Compute grid spacing
#                 lat_step = (lat_max - lat_min) / H
#                 lon_step = (lon_max - lon_min) / W
            
            
#                 i_min = int((lat0-lat_min)/lat_step)
#                 i_max =  int((lat1-lat_min)/lat_step)
#                 j_min = int ((lon0-lon_min)/lon_step)
#                 j_max = int((lon1-lon_min)/lon_step)
#                 output_idx = batch.outputs.feature_names_to_idx[cfg_xai["explain"]["output"]]
#                 if batch.outputs.tensor[:,:,i_min:i_max,j_min:j_max,output_idx].mean() > 0.15:
#                     batch.inputs.tensor = batch.inputs.tensor.to(device)
#                     batch.forcing.tensor = batch.forcing.tensor.to(device)
#                     if batch.outputs is not None:
#                         batch.outputs.tensor = batch.outputs.tensor.to(device)
#                     batch.inputs.tensor.requires_grad_()

                    
#                     explain_clean,output = self.explainer.compute_explanations(
#                         batch, batch_idx, checkpoint,
#                         cfg_model, cfg_dataset,cfg_xai, dataset_info,
#                         infer_ds, list_run_hour, use_old_weights, target,plot_explanations=plot_explanations,extent=extent,fig_path=fig_path,return_output=True
#                     )
#                     if explain_clean is None:
#                         continue
#                     # print("rainy event:",batch_idx)
#                     idx_samples = [batch_idx * 1 + b for b in range(1)]
#                     samples = [infer_ds.sample_list[idx] for idx in idx_samples]
#                     runtimes = [
#                                     sample.timestamps.datetime.strftime("%Y%m%d%H") for sample in samples
#                                 ]
#                     if cfg_xai["explain"]["forcing_input"]:
#                         input_clean = batch.forcing.tensor.clone()
#                         input_idx = batch.forcing.feature_names_to_idx[cfg_xai["explain"]["input"]]
#                         vals = batch.forcing.tensor[..., input_idx]

#                         noise_std = self.std_noise * (vals.max() - vals.min())
#                     else:
#                         input_clean = batch.inputs.tensor.clone()
#                         input_idx = batch.inputs.feature_names_to_idx[cfg_xai["explain"]["input"]]
#                         vals = batch.inputs.tensor[..., input_idx]

#                         noise_std = self.std_noise * (vals.max() - vals.min())


#                     output_idx = batch.outputs.feature_names_to_idx[cfg_xai["explain"]["output"]]

#                     if self.explainer_name in {"GradCam", "WassersteinGrad","WassersteinSmoothGrad"}:
#                             for i in range(len(explain_clean)):
#                                 explain_clean_t = normalize_expl(explain_clean[i])
#                                 B=1
#                                 # print(explain_clean_t.shape)
#                                 H,W = explain_clean_t.shape
                            
#                                 #TODO: case where we have multiple previous timesteps
#                                 if cfg_xai["explain"]["forcing_input"]:
#                                     x = batch.forcing.tensor[:,i,...,input_idx].unsqueeze(-1)
                                
#                                 else:
#                                     if i ==0:
#                                         x = batch.inputs.tensor[:,0,...,input_idx].unsqueeze(-1)
#                                     else:
#                                         print("nooooooooooooooooooooooooooooo")
#                                         x = output.tensor[:,i-1,...,input_idx].unsqueeze(-1)


#                                 explain_clean_t= explain_clean_t.reshape(B, -1)
#                                 all_scores_test = []
#                                 # ---  ROAD loop: progressively remove top-p% ---
#                                 for p in self.percentages:

#                                     per = p
#                                     top_k =int(per /100* H*W)
                                
#                                     _, idx = torch.topk(explain_clean_t, top_k, dim=1)
#                                     # idx = get_random_mask(explain_clean_t, top_k)
#                                     mask = torch.zeros_like(explain_clean_t)
#                                     mask.scatter_(1, idx, 1.0)

#                                     mask = mask.view(B,H, W,1)

                                
                                    
#                                     # Apply linear perturbation
#                                     x_pert = linear_imputation(x, mask.to(device),noise_std=noise_std).unsqueeze(1).squeeze(-1)
                                
                 
#                                     if cfg_xai["explain"]["forcing_input"]:
#                                         new_inputs = batch.forcing.tensor.clone()
#                                         new_inputs[...,input_idx] = x_pert
#                                         batch.forcing.tensor = new_inputs


#                                     else:
#                                         new_inputs = batch.inputs.tensor.clone()
#                                         new_inputs[...,input_idx] = x_pert
#                                         batch.inputs.tensor = new_inputs
                                
#                                     # batch.num_pred_steps = 1
                                
#                                     with torch.no_grad():
#                                         pred_pert = predict_step(self.model, batch, batch_idx, checkpoint,
#                                             cfg_model, cfg_dataset,cfg_xai, dataset_info,
#                                             infer_ds, list_run_hour, use_old_weights)
                                    

#                                     batch.num_pred_steps = int(cfg_dataset["num_pred_steps_val_test"])
                                    
#                                     if cfg_xai["explain"]["forcing_input"]:
#                                         batch.forcing.tensor = input_clean

#                                     else:
#                                         batch.inputs.tensor = input_clean

#                                     # Compare output degradation
#                                     # score = torch.mean((batch.outputs.tensor[:,batch.num_pred_steps-1,...,output_idx] - pred_pert.tensor[:,batch.num_pred_steps-1,...,output_idx]) ** 2)
#                                     # clean_mse = torch.mean((batch.outputs.tensor[:, batch.num_pred_steps-1, ..., output_idx]
#                                     #                         - output.tensor[:, batch.num_pred_steps-1, ..., output_idx]) ** 2)
#                                     # pert_mse  = torch.mean((batch.outputs.tensor[:, batch.num_pred_steps-1, ..., output_idx]
#                                     #                     - pred_pert.tensor[:, batch.num_pred_steps-1, ..., output_idx]) ** 2)
                                    
#                                     # score = torch.abs(pert_mse - clean_mse)  # faithfulness = forecast gets worse
#                                     # score = score / clean_mse
#                                     extent = cfg_xai["extent"]
                        
#                                     lon0,lon1 = target[0], target[1]
#                                     lat0,lat1 = target[2], target[3]
#                                     H, W = batch.inputs.tensor.shape[2:4]
#                                     lat_min, lat_max = extent[2], extent[3]
#                                     lon_min, lon_max = extent[0], extent[1]

#                                     # Compute grid spacing
#                                     lat_step = (lat_max - lat_min) / H
#                                     lon_step = (lon_max - lon_min) / W
                                
                                
#                                     i_min = int((lat0-lat_min)/lat_step)
#                                     i_max =  int((lat1-lat_min)/lat_step)
#                                     j_min = int ((lon0-lon_min)/lon_step)
#                                     j_max = int((lon1-lon_min)/lon_step)
#                                     out_t = _normalize(output.tensor[:, batch.num_pred_steps-1, i_min:i_max,j_min:j_max, output_idx].detach().cpu(),stats=dataset_info.stats,name="aro_tp_0m")
#                                     out_p = _normalize(pred_pert.tensor[:, batch.num_pred_steps-1, i_min:i_max,j_min:j_max, output_idx].detach().cpu(),stats=dataset_info.stats,name="aro_tp_0m")
#                                     score = torch.mean((torch.abs(out_t-out_p)))#/torch.mean(out_t**2)
                                
#                                     # print(score)
#                                     score /= torch.mean(torch.abs(out_t))
                                    
#                                     if score> 0.008:
#                                         all_scores[i][p].append(0)
#                                         all_scores_test.append(0) 
                                    
#                                     else:        
#                                         all_scores[i][p].append(1)
#                                         all_scores_test.append(1)                   
#                                     # all_scores[i][p].append(score.cpu().detach().numpy())
#                                     # all_scores_test.append(score.cpu().detach().numpy())    
                                    
#                                 y = np.array(all_scores_test)          # % of features removed
#                                 x = np.array(list(self.percentages))        # degradation (MSE)
#                                 auc_val = np.trapz(y, x) / (x.max() - x.min()) 
#                                 # print(np.trapz(y, x) / (x.max() - x.min()))    

#                                 # if auc_val >0.01 :
#                                 #     continue
#                                 print("rainy_event: ",runtimes[0])


#                                 auc_list[i].append( np.trapz(y, x) / (x.max() - x.min())  )      
#                     else:
                    
#                             for i in range(len(explain_clean)):
#                                 explain_clean_t = normalize_expl(explain_clean[i][...,0]).squeeze(1).cpu()
#                                 if explain_clean_t.dim() == 3:
#                                     B,H,W = explain_clean_t.shape
#                                 elif explain_clean_t.dim() == 4:
#                                     B,H,W,_ = explain_clean_t.shape
#                                 else:
#                                     B,_,H,W,_ = explain_clean_t.shape

#                                 #TODO: case where we have multiple previous timesteps
#                                 if cfg_xai["explain"]["forcing_input"]:
#                                     x = batch.forcing.tensor[:,i,...,input_idx].unsqueeze(-1)
                                
#                                 else:
#                                     if i ==0:
#                                         x = batch.inputs.tensor[:,0,...,input_idx].unsqueeze(-1)
#                                     else:
#                                         x = output.tensor[:,i-1,...,input_idx].unsqueeze(-1)

#                                 if explain_clean_t.dim()==3:

#                                     explain_clean_t= explain_clean_t.reshape(B, -1)
#                                 else: 
#                                     explain_clean_t= explain_clean_t[...,0].reshape(B, -1)

#                                 all_scores_test = []
#                                 # ---  ROAD loop: progressively remove top-p% ---
#                                 for p in self.percentages:

#                                     per = p
#                                     top_k =int(per /100* H*W)
                                
                                
#                                     _, idx = torch.topk(explain_clean_t, top_k, dim=1)
#                                     mask = torch.zeros_like(explain_clean_t)
#                                     mask.scatter_(1, idx, 1.0)

#                                     mask = mask.view(B,H, W,1)

#                                     x_pert = linear_imputation(x, mask.to(device),noise_std=noise_std).unsqueeze(1).squeeze(-1)
                        
                                
#                                     if cfg_xai["explain"]["forcing_input"]:
#                                         new_inputs = batch.forcing.tensor.clone()
#                                         new_inputs[...,input_idx] = x_pert
#                                         batch.forcing.tensor = new_inputs


#                                     else:
#                                         new_inputs = batch.inputs.tensor.clone()
#                                         new_inputs[...,input_idx] = x_pert
#                                         batch.inputs.tensor = new_inputs
                                
#                                     # batch.num_pred_steps = 1
                                
#                                     with torch.no_grad():
#                                         pred_pert = predict_step(self.model, batch, batch_idx, checkpoint,
#                                             cfg_model, cfg_dataset,cfg_xai, dataset_info,
#                                             infer_ds, list_run_hour, use_old_weights)
                                    

#                                     batch.num_pred_steps = int(cfg_dataset["num_pred_steps_val_test"])
                                    
#                                     if cfg_xai["explain"]["forcing_input"]:
#                                         batch.forcing.tensor = input_clean

#                                     else:
#                                         batch.inputs.tensor = input_clean

#                                     # Compare output degradation
                                    
#                                     lon0,lon1 = target[0], target[1]
#                                     lat0,lat1 = target[2], target[3]
#                                     H, W = batch.inputs.tensor.shape[2:4]
#                                     lat_min, lat_max = extent[2], extent[3]
#                                     lon_min, lon_max = extent[0], extent[1]

#                                     # Compute grid spacing
#                                     lat_step = (lat_max - lat_min) / H
#                                     lon_step = (lon_max - lon_min) / W
                                
                                
#                                     i_min = int((lat0-lat_min)/lat_step)
#                                     i_max =  int((lat1-lat_min)/lat_step)
#                                     j_min = int ((lon0-lon_min)/lon_step)
#                                     j_max = int((lon1-lon_min)/lon_step)
#                                     out_t = _normalize(output.tensor[:, batch.num_pred_steps-1, i_min:i_max,j_min:j_max, output_idx].detach().cpu(),stats=dataset_info.stats,name="aro_tp_0m")
#                                     out_p = _normalize(pred_pert.tensor[:, batch.num_pred_steps-1, i_min:i_max,j_min:j_max, output_idx].detach().cpu(),stats=dataset_info.stats,name="aro_tp_0m")
#                                     score = torch.mean((torch.abs(out_t-out_p)))#/torch.mean(out_t**2)
                                
#                                     # print(score)
#                                     score /= torch.mean(torch.abs(out_t))
                                    
#                                     if score> 0.008:
#                                         all_scores[i][p].append(0)
#                                         all_scores_test.append(0) 
                                    
#                                     else:        
#                                         all_scores[i][p].append(1)
#                                         all_scores_test.append(1)                   
#                                     # all_scores[i][p].append(score.cpu().detach().numpy())
#                                     # all_scores_test.append(score.cpu().detach().numpy())    
                                    
#                                 y = np.array(all_scores_test)          # % of features removed
#                                 x = np.array(list(self.percentages))        # degradation (MSE)
#                                 auc_val = np.trapz(y, x) / (x.max() - x.min()) 
#                                 # print(np.trapz(y, x) / (x.max() - x.min()))    

#                                 # if auc_val >0.01 :
#                                 #     continue
#                                 print("rainy_event: ",runtimes[0])


#                                 auc_list[i].append( np.trapz(y, x) / (x.max() - x.min())  )   

            

                   
#         #all_scores[0][p]
#         road_values =[]
#         for p in self.percentages:
#             avg_p = np.nanmean(all_scores[0][p])
#             road_values.append(avg_p)

#         x_p = np.array(self.percentages)
#         y_p = np.array(road_values)
#         print("The road score: ",np.trapz(y_p, x_p) / (x_p.max() - x_p.min()) )

#         # return mean_curve,std_curve,auc_list
#         all_scores_mean = []
#         all_scores_std = []
#         for i in all_scores:
#             all_scores_mean.append(np.nanmean(auc_list[i]))
#             all_scores_std.append(np.nanstd(auc_list[i]))

#         return auc_list,all_scores_mean,all_scores_std

# Copyright 2024 - ROAD Metric adapted for Meteorological Attribution
# Adapted from Google PIC metric (https://arxiv.org/abs/1906.02825)
# Key adaptation: PIC-style normalisation between blurred and clean prediction
# to remove precipitation intensity confounding.

import numpy as np
import torch
import torch.nn.functional as F
import os
from typing import Dict, List, Optional, Tuple, NamedTuple
from scipy.ndimage import gaussian_filter

from py4castxai.explainers import explainers_registry
from py4castxai.utils_test.infer_utils import predict_step


# ---------------------------------------------------------------------------
# Result containers
# ---------------------------------------------------------------------------

class ROADEventResult(NamedTuple):
    runtime:      str
    auc:          float
    curve_x:      np.ndarray
    curve_y:      np.ndarray
    pred_clean:   float
    # pred_blurred: float
    precip_mean:  float
    # gap:          float        # added: useful diagnostic


class ROADAggregateResult(NamedTuple):
    mean_auc:      float
    std_auc:       float
    sem_auc:       float
    curve_x:       np.ndarray
    curve_y_mean:  np.ndarray
    curve_y_std:   np.ndarray
    n_events:      int
    event_results: List[ROADEventResult]


# ---------------------------------------------------------------------------
# Utility functions
# ---------------------------------------------------------------------------

def normalize_expl(a: torch.Tensor, abs_val: bool = True) -> torch.Tensor:
    """Normalise attribution map to [0, 1]."""
    if isinstance(a, np.ndarray):
        a = torch.from_numpy(a).to(torch.float32)
    if abs_val:
        return a.abs() / (a.abs().max() + 1e-8)
    return a / (a.abs().max() + 1e-8)


def spatial_blur(x: torch.Tensor, sigma: float = 10.0) -> torch.Tensor:
    """
    Gaussian blur over spatial field — PIC lower bound baseline.
    Args:
        x:     [B, H, W, 1]
        sigma: blur radius in pixels (~25km at 2.5km resolution for sigma=10)
    """
    x_np = x.detach().cpu().squeeze().numpy()
    blurred_np = gaussian_filter(x_np, sigma=sigma)
    return torch.from_numpy(blurred_np).to(x.dtype).to(x.device).reshape(x.shape)


def linear_imputation(
    x: torch.Tensor,
    mask: torch.Tensor,
    noise_std: float = 0.01,
    seed: Optional[int] = None       # Fix Bug 3: explicit seed control
) -> torch.Tensor:
    """
    Replace masked pixels with local neighbourhood mean + small noise.

    Args:
        x:         [B, H, W, 1]
        mask:      [B, H, W, 1], 1 = pixels to replace
        noise_std: Gaussian noise std (small to stay in-distribution)
        seed:      if provided, seeds torch RNG for reproducibility
    """
    if seed is not None:
        torch.manual_seed(seed)

    kernel = torch.ones(1, 1, 3, 3, device=x.device) / 9.0

    if x.ndim == 4 and x.shape[-1] == 1:
        x_perm = x.permute(0, 3, 1, 2)
        neighbor_mean = F.conv2d(x_perm, kernel, padding=1)
        neighbor_mean = neighbor_mean.permute(0, 2, 3, 1)
    else:
        neighbor_mean = F.conv2d(
            x.unsqueeze(1), kernel, padding=1
        ).squeeze(1)

    noise = noise_std * torch.randn_like(x)
    return x * (1 - mask) + (neighbor_mean + noise) * mask


def compute_roi_indices(
    target: List[float],
    extent: List[float],
    H: int,
    W: int
) -> Tuple[int, int, int, int]:
    """Geographic bounding box → pixel indices."""
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

    return i_min, i_max, j_min, j_max


def get_roi_mean(
    tensor: torch.Tensor,
    i_min: int, i_max: int,
    j_min: int, j_max: int,
    t_idx: int,
    feat_idx: int
) -> float:
    """Mean model output over Paris ROI."""
    
    return tensor[
        :, t_idx, i_min:i_max, j_min:j_max, feat_idx
    ].mean().item()


# ---------------------------------------------------------------------------
# ROAD class
# ---------------------------------------------------------------------------

class ROAD:
    """
    ROAD (Remove And Debias) — Binary Faithfulness Metric
    Adapted for meteorological regression.

    Scoring:
        For each masking percentage p, we compare:
          degradation_salient = |pred_clean - pred_pert_salient|
          degradation_random  = mean over N_random random masks of
                                |pred_clean - pred_pert_random|

        binary_score(p) = 1 if degradation_salient > degradation_random
                          0 otherwise

        AUC = trapz(binary_scores, percentages) / range

    Interpretation:
        AUC = 1.0  → always beats random → perfectly faithful
        AUC = 0.5  → chance level (random baseline)
        AUC = 0.0  → worse than random

    Validity filters:
        1. precip_threshold: skip dry events
        2. gap_threshold:    skip events where jet stream
                             doesn't affect Paris precipitation
    """

    def __init__(
        self,
        model: torch.nn.Module,
        explainer: str = "BaseGrad",
        precip_threshold: float = 0.10,
        gap_threshold: float = 0.0005,
        blur_sigma: float = 10.0,
        std_noise: float = 0.2,
        n_random: int = 5,
        percentages: Optional[List[int]] = None,
        seed: int = 42,
        **expl_params
    ):
        self.model = model
        self.precip_threshold = 5
        self.gap_threshold = gap_threshold
        self.blur_sigma = blur_sigma
        self.noise_std = std_noise
        self.n_random = n_random
        self.seed = seed

        # self.percentages = (
        #      [i / 10 for i in range(1, 10)]
        #        + list(range(1, 10))
        #        + list(range(10, 30, 2))
        # )
        self.percentages =list(range(1, 16,1))
        # → [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 12, 14]
        self.explainer_name = explainer
        self.explainer = self._init_explainer(explainer, expl_params)

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _init_explainer(self, name: str, params: Dict):
        cls = explainers_registry.get(name)
        if cls is None:
            raise ValueError(
                f"Unknown explainer: {name}. "
                f"Available: {list(explainers_registry.keys())}"
            )
        return cls(self.model, **params)

    def _get_runtime(self, batch_idx: int, infer_ds) -> str:
        sample = infer_ds.sample_list[batch_idx]
        return sample.timestamps.datetime.strftime("%Y%m%d%H")

    def _move_batch_to_device(self, batch, device: torch.device):
        batch.inputs.tensor  = batch.inputs.tensor.to(device)
        batch.forcing.tensor = batch.forcing.tensor.to(device)
        if batch.outputs is not None:
            batch.outputs.tensor = batch.outputs.tensor.to(device)
        return batch

    def _run_mask(
        self,
        idx: torch.Tensor,
        x: torch.Tensor,
        batch,
        input_clean: torch.Tensor,
        input_idx: int,
        output_idx: int,
        i_min: int, i_max: int,
        j_min: int, j_max: int,
        t_step: int,
        H: int, W: int,
        noise_std: float,
        batch_idx: int,
        checkpoint,
        cfg_model, cfg_dataset, cfg_xai,
        dataset_info, infer_ds,
        list_run_hour, use_old_weights,
        device: torch.device,
        forcing_input: bool,
        seed: Optional[int] = None    # Fix Bug 3: pass seed explicitly
    ) -> float:
        """
        Apply pixel mask, run model, return ROI prediction.

        Parameters
        ----------
        idx  : [1, top_k] flat pixel indices to mask
        seed : RNG seed for imputation noise (None = unseeded)
        """
        B = 1
        mask = torch.zeros(B, H * W, device=x.device)
        mask.scatter_(1, idx.to(x.device), 1.0)
        mask = mask.view(B, H, W, 1)

        # Fix Bug 3: pass seed to control noise independently
        x_pert = linear_imputation(
            x, mask.to(device),
            noise_std=noise_std,
            seed=seed
        ).unsqueeze(1).squeeze(-1)

        if forcing_input:
            new_inputs = batch.forcing.tensor.clone()
            new_inputs[..., input_idx] = x_pert
            batch.forcing.tensor = new_inputs
        else:
            new_inputs = batch.inputs.tensor.clone()
            new_inputs[..., input_idx] = x_pert
            batch.inputs.tensor = new_inputs

        with torch.no_grad():
            pred_pert = predict_step(
                self.model, batch, batch_idx, checkpoint,
                cfg_model, cfg_dataset, cfg_xai, dataset_info,
                infer_ds, list_run_hour, use_old_weights
            )

        if forcing_input:
            batch.forcing.tensor = input_clean
        else:
            batch.inputs.tensor = input_clean

        return get_roi_mean(
            pred_pert.tensor,
            i_min, i_max, j_min, j_max,
            t_idx=t_step, feat_idx=output_idx
        )

    def _compute_blurred_prediction(
        self,
        batch,
        input_clean: torch.Tensor,
        input_idx: int,
        output_idx: int,
        i_min: int, i_max: int,
        j_min: int, j_max: int,
        t_step: int,
        batch_idx: int,
        checkpoint,
        cfg_model, cfg_dataset, cfg_xai,
        dataset_info, infer_ds,
        list_run_hour, use_old_weights,
        device: torch.device,
        forcing_input: bool
    ) -> float:
        """
        PIC lower bound: model prediction when jet stream
        spatial structure is destroyed by Gaussian blur.
        """
        x = (
            batch.forcing.tensor[:, 0, ..., input_idx].unsqueeze(-1)
            if forcing_input
            else batch.inputs.tensor[:, 0, ..., input_idx].unsqueeze(-1)
        )

        x_blurred = spatial_blur(x, sigma=self.blur_sigma).to(device)
        x_blurred = x_blurred.unsqueeze(1).squeeze(-1)

        if forcing_input:
            new_inputs = batch.forcing.tensor.clone()
            new_inputs[..., input_idx] = x_blurred
            batch.forcing.tensor = new_inputs
        else:
            new_inputs = batch.inputs.tensor.clone()
            new_inputs[..., input_idx] = x_blurred
            batch.inputs.tensor = new_inputs

        with torch.no_grad():
            pred_blurred = predict_step(
                self.model, batch, batch_idx, checkpoint,
                cfg_model, cfg_dataset, cfg_xai, dataset_info,
                infer_ds, list_run_hour, use_old_weights
            )

        if forcing_input:
            batch.forcing.tensor = input_clean
        else:
            batch.inputs.tensor = input_clean

        return get_roi_mean(
            pred_blurred.tensor,
            i_min, i_max, j_min, j_max,
            t_step, output_idx
        )

    # ------------------------------------------------------------------
    # Main evaluation
    # ------------------------------------------------------------------

    def evaluate(
        self,
        dataloader,
        checkpoint,
        cfg_model,
        cfg_dataset,
        cfg_xai,
        dataset_info,
        infer_ds,
        list_run_hour,
        use_old_weights,
        target,
        plot_explanations,
        extent,
        fig_path,precip_thresh=None
    ) -> ROADAggregateResult:
        """
        Evaluate binary ROAD faithfulness across all valid events.

        For each valid event:
          1. Filter: precip > threshold AND gap > gap_threshold
          2. For each masking percentage p:
             a. Mask top-p% salient features → degradation_salient
             b. Average N_random random masks → degradation_random
             c. binary(p) = 1 if salient > random else 0
          3. AUC = trapz(binary_curve) / range

        Higher AUC = more faithful (beats random more often).
        Random baseline ≈ 0.5 by construction.
        """
        # Global seed for reproducibility
        torch.manual_seed(self.seed)
        np.random.seed(self.seed)

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        os.makedirs(
            cfg_xai["eval"]["eval_plot"]["fig_path"] + "/faithfulness",
            exist_ok=True
        )


        if precip_thresh:
            self.precip_threshold = precip_thresh
            print("ejzpoijhdphpihd",self.precip_threshold)
        forcing_input = cfg_xai["explain"]["forcing_input"]
        n_pred_steps  = int(cfg_dataset["num_pred_steps_val_test"])
        all_event_results: List[ROADEventResult] = []
        print("steps are:", n_pred_steps)
        with torch.autograd.set_grad_enabled(True):

            for batch_idx, batch in enumerate(dataloader):

                # ── ROI indices ───────────────────────────────────────────
                H, W = batch.inputs.tensor.shape[2:4]
                i_min, i_max, j_min, j_max = compute_roi_indices(
                    target, extent, H, W
                )
                output_idx = batch.outputs.feature_names_to_idx[
                    cfg_xai["explain"]["output"]
                ]

                # ── Filter 1: precipitation ───────────────────────────────
                precip_mean = get_roi_mean(
                    batch.outputs.tensor,
                    i_min, i_max, j_min, j_max,
                    t_idx=0, feat_idx=output_idx
                )
                if precip_mean <= self.precip_threshold:
                    continue

                # ── Device + gradient setup ───────────────────────────────
                batch = self._move_batch_to_device(batch, device)
                batch.inputs.tensor.requires_grad_()
                runtime = self._get_runtime(batch_idx, infer_ds)

                # ── Attribution map ───────────────────────────────────────
                explain_clean, output_clean = self.explainer.compute_explanations(
                    batch, batch_idx, checkpoint,
                    cfg_model, cfg_dataset, cfg_xai, dataset_info,
                    infer_ds, list_run_hour, use_old_weights,
                    target,
                    plot_explanations=plot_explanations,
                    extent=extent,
                    fig_path=fig_path,
                    return_output=True
                )
                if explain_clean is None:
                    continue

                # ── Input bookkeeping ─────────────────────────────────────
                if forcing_input:
                    input_clean = batch.forcing.tensor.clone()
                    input_idx   = batch.forcing.feature_names_to_idx[
                        cfg_xai["explain"]["input"]
                    ]
                    vals = batch.forcing.tensor[..., input_idx]
                else:
                    input_clean = batch.inputs.tensor.clone()
                    input_idx   = batch.inputs.feature_names_to_idx[
                        cfg_xai["explain"]["input"]
                    ]
                    vals = batch.inputs.tensor[..., input_idx]

                noise_std = self.noise_std * (vals.max() - vals.min()).item()
                t_step    = n_pred_steps - 1

                # ── Clean prediction (PIC upper bound) ────────────────────
                pred_clean_roi = get_roi_mean(
                    output_clean.tensor,
                    i_min, i_max, j_min, j_max,
                    t_idx=t_step, feat_idx=output_idx
                )

                # ── Attribution normalisation ─────────────────────────────
                explain_t = explain_clean[0]
                explain_t = normalize_expl(
                    explain_t if explain_t.dim() == 2
                    else explain_t[..., 0].squeeze()
                )
                explain_flat = explain_t.reshape(1, -1)

                x = (
                    batch.forcing.tensor[:, 0, ..., input_idx].unsqueeze(-1)
                    if forcing_input
                    else batch.inputs.tensor[:, 0, ..., input_idx].unsqueeze(-1)
                )

                # ── Masking loop ──────────────────────────────────────────
                curve_y      = []
                valid_event  = True

                for p_idx, p in enumerate(self.percentages):

                    top_k = int(p / 100 * H * W)
                    if top_k == 0:
                        curve_y.append(0)
                        continue

                    # Salient mask
                    _, idx_salient = torch.topk(explain_flat, top_k, dim=1)

                    # Fix Bug 3: unique seed per (event, percentage, salient)
                    salient_seed = self.seed + batch_idx * 1000 + p_idx

                    pred_salient = self._run_mask(
                        idx=idx_salient,
                        x=x,
                        batch=batch,
                        input_clean=input_clean,
                        input_idx=input_idx,
                        output_idx=output_idx,
                        i_min=i_min, i_max=i_max,
                        j_min=j_min, j_max=j_max,
                        t_step=t_step,
                        H=H, W=W,
                        noise_std=noise_std,
                        batch_idx=batch_idx,
                        checkpoint=checkpoint,
                        cfg_model=cfg_model,
                        cfg_dataset=cfg_dataset,
                        cfg_xai=cfg_xai,
                        dataset_info=dataset_info,
                        infer_ds=infer_ds,
                        list_run_hour=list_run_hour,
                        use_old_weights=use_old_weights,
                        device=device,
                        forcing_input=forcing_input,
                        seed=salient_seed
                    )

                    degradation_salient = abs(pred_clean_roi - pred_salient)

                    # Random baseline: N_random masks, different seed each
                    random_degradations = []
                    for r in range(self.n_random):

                        # Fix Bug 3: unique seed per (event, percentage, random_idx)
                        random_seed = self.seed + batch_idx * 1000 + p_idx * 100 + r

                        torch.manual_seed(random_seed)
                        idx_random = torch.stack([
                            torch.randperm(H * W)[:top_k]
                        ])

                        pred_r = self._run_mask(
                            idx=idx_random,
                            x=x,
                            batch=batch,
                            input_clean=input_clean,
                            input_idx=input_idx,
                            output_idx=output_idx,
                            i_min=i_min, i_max=i_max,
                            j_min=j_min, j_max=j_max,
                            t_step=t_step,
                            H=H, W=W,
                            noise_std=noise_std,
                            batch_idx=batch_idx,
                            checkpoint=checkpoint,
                            cfg_model=cfg_model,
                            cfg_dataset=cfg_dataset,
                            cfg_xai=cfg_xai,
                            dataset_info=dataset_info,
                            infer_ds=infer_ds,
                            list_run_hour=list_run_hour,
                            use_old_weights=use_old_weights,
                            device=device,
                            forcing_input=forcing_input,
                            seed=random_seed + 999  # different from mask seed
                        )
                        random_degradations.append(
                            abs(pred_clean_roi - pred_r)
                        )

                    degradation_random = float(np.mean(random_degradations))

                    # Binary comparison — same scale (both raw MAE)
                    binary = 1 if degradation_salient > degradation_random else 0
                    curve_y.append(binary)

                    # print(
                    #     f"  p={p:5.1f}% | "
                    #     f"salient={degradation_salient:.5f} | "
                    #     f"random={degradation_random:.5f} | "
                    #     f"binary={binary}"
                    # )

                # ── AUC ───────────────────────────────────────────────────
                if not valid_event or len(curve_y) != len(self.percentages):
                    continue

                curve_x = np.array(self.percentages, dtype=float)
                curve_y = np.array(curve_y,          dtype=float)
                auc = (
                    np.trapezoid(curve_y, curve_x)
                    / (curve_x.max() - curve_x.min())
                )

                all_event_results.append(ROADEventResult(
                    runtime=runtime,
                    auc=float(auc),
                    curve_x=curve_x,
                    curve_y=curve_y,
                    pred_clean=pred_clean_roi,
                    # pred_blurred=pred_blurred_roi,
                    precip_mean=precip_mean,
                    # gap=gap
                ))

                print(
                    f"[ROAD] {runtime} | "
                    f"precip={precip_mean:.3f} | "
                    # f"gap={gap:.4f} | "
                    f"AUC={auc:.4f}"
                )

        return self._aggregate(all_event_results)

    # ------------------------------------------------------------------
    # Aggregation
    # ------------------------------------------------------------------

    def _aggregate(
        self,
        event_results: List[ROADEventResult]
    ) -> ROADAggregateResult:
        """
        Aggregate per-event results.
        Reports SEM for paper table (not STD).
        STD reflects meteorological regime variability — real, not noise.
        """
        if not event_results:
            raise ValueError(
                "No valid events. Check precip_threshold and gap_threshold."
            )

        aucs   = np.array([r.auc for r in event_results])
        curves = np.stack([r.curve_y for r in event_results], axis=0)

        n        = len(aucs)
        mean_auc = float(np.mean(aucs))
        std_auc  = float(np.std(aucs))
        sem_auc  = float(std_auc / np.sqrt(n))

        print(f"\n[ROAD] Aggregated over {n} valid events")
        print(f"  AUC mean : {mean_auc:.4f}")
        print(f"  AUC std  : {std_auc:.4f}  (event-to-event variability)")
        print(f"  AUC SEM  : {sem_auc:.4f}  (report in paper table)")
        print(f"  Random baseline ≈ 0.5 by construction")

        return ROADAggregateResult(
            mean_auc=mean_auc,
            std_auc=std_auc,
            sem_auc=sem_auc,
            curve_x=event_results[0].curve_x,
            curve_y_mean=np.mean(curves, axis=0),
            curve_y_std=np.std(curves,  axis=0),
            n_events=n,
            event_results=event_results
        )

    # ------------------------------------------------------------------
    # Stratified analysis
    # ------------------------------------------------------------------

    def stratified_auc(
        self,
        result: ROADAggregateResult,
        low_threshold:  float = 0.10,
        high_threshold: float = 0.40
    ) -> Dict:
        """
        Break AUC down by precipitation intensity.
        All methods must be evaluated on the same event set
        before comparing stratified results.
        """
        events = result.event_results
        weak   = [r for r in events
                  if low_threshold < r.precip_mean <= high_threshold]
        strong = [r for r in events if r.precip_mean > high_threshold]

        def _summarise(subset, label):
            if not subset:
                print(f"[stratified] No events in '{label}'")
                return {"mean": None, "sem": None, "n": 0}
            aucs = np.array([r.auc for r in subset])
            return {
                "mean": float(np.mean(aucs)),
                "sem":  float(np.std(aucs) / np.sqrt(len(aucs))),
                "n":    len(aucs)
            }

        return {
            "weak":   _summarise(weak,   "weak"),
            "strong": _summarise(strong, "strong"),
            "all":    _summarise(events, "all")
        }
    

import os
import torch
import numpy as np
from typing import Optional, List, Dict

# Assumes external imports are handled in your main script:
# from your_module import explainers_registry, linear_imputation, predict_step, ...
# from your_module import get_roi_mean, spatial_blur, compute_roi_indices, normalize_expl
# from your_module import ROADEventResult, ROADAggregateResult


#  class ROAD:
#     """
#     ROAD-MSE (Remove And Debias - Mean Squared Error)
#     Adapted for meteorological regression.

#     Scoring:
#         For each masking percentage p, we compute the Mean Squared Error (MSE):
#           mse_salient = (pred_clean - pred_pert_salient)^2
#           mse_random  = mean over N_random random masks of
#                         (pred_clean - pred_pert_random)^2
                        
#           mse_gap(p)  = mse_salient - mse_random

#         AUC = trapz(mse_gaps, percentages) / range

#     Interpretation:
#         AUC > 0.0  → Salient mask causes worse error than random mask (Faithful)
#         AUC = 0.0  → Chance level (explainer is as good as random)
#         AUC < 0.0  → Worse than random

#     Validity filters:
#         1. precip_threshold: skip dry events
#         2. gap_threshold:    skip events where jet stream
#                              doesn't affect Paris precipitation
#     """

#     def __init__(
#         self,
#         model: torch.nn.Module,
#         explainer: str = "BaseGrad",
#         precip_threshold: float = 0.10,
#         gap_threshold: float = 0.0005,
#         blur_sigma: float = 10.0,
#         std_noise: float = 0.2,
#         n_random: int = 5,
#         percentages: Optional[List[int]] = None,
#         seed: int = 42,
#         **expl_params
#     ):
#         self.model = model
#         self.precip_threshold = 5
#         self.gap_threshold = gap_threshold
#         self.blur_sigma = blur_sigma
#         self.noise_std = std_noise
#         self.n_random = n_random
#         self.seed = seed

#         # → [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 12, 14]
#         self.percentages = list(range(1, 10)) + list(range(10, 16, 1))
        
#         self.explainer_name = explainer
#         self.explainer = self._init_explainer(explainer, expl_params)

#     # ------------------------------------------------------------------
#     # Private helpers
#     # ------------------------------------------------------------------

#     def _init_explainer(self, name: str, params: Dict):
#         cls = explainers_registry.get(name)
#         if cls is None:
#             raise ValueError(
#                 f"Unknown explainer: {name}. "
#                 f"Available: {list(explainers_registry.keys())}"
#             )
#         return cls(self.model, **params)

#     def _get_runtime(self, batch_idx: int, infer_ds) -> str:
#         sample = infer_ds.sample_list[batch_idx]
#         return sample.timestamps.datetime.strftime("%Y%m%d%H")

#     def _move_batch_to_device(self, batch, device: torch.device):
#         batch.inputs.tensor  = batch.inputs.tensor.to(device)
#         batch.forcing.tensor = batch.forcing.tensor.to(device)
#         if batch.outputs is not None:
#             batch.outputs.tensor = batch.outputs.tensor.to(device)
#         return batch

#     def _run_mask(
#         self,
#         idx: torch.Tensor,
#         x: torch.Tensor,
#         batch,
#         input_clean: torch.Tensor,
#         input_idx: int,
#         output_idx: int,
#         i_min: int, i_max: int,
#         j_min: int, j_max: int,
#         t_step: int,
#         H: int, W: int,
#         noise_std: float,
#         batch_idx: int,
#         checkpoint,
#         cfg_model, cfg_dataset, cfg_xai,
#         dataset_info, infer_ds,
#         list_run_hour, use_old_weights,
#         device: torch.device,
#         forcing_input: bool,
#         seed: Optional[int] = None
#     ) -> float:
#         """
#         Apply pixel mask, run model, return ROI prediction.
#         """
#         B = 1
#         mask = torch.zeros(B, H * W, device=x.device)
#         mask.scatter_(1, idx.to(x.device), 1.0)
#         mask = mask.view(B, H, W, 1)

#         x_pert = linear_imputation(
#             x, mask.to(device),
#             noise_std=noise_std,
#             seed=seed
#         ).unsqueeze(1).squeeze(-1)

#         if forcing_input:
#             new_inputs = batch.forcing.tensor.clone()
#             new_inputs[..., input_idx] = x_pert
#             batch.forcing.tensor = new_inputs
#         else:
#             new_inputs = batch.inputs.tensor.clone()
#             new_inputs[..., input_idx] = x_pert
#             batch.inputs.tensor = new_inputs

#         with torch.no_grad():
#             pred_pert = predict_step(
#                 self.model, batch, batch_idx, checkpoint,
#                 cfg_model, cfg_dataset, cfg_xai, dataset_info,
#                 infer_ds, list_run_hour, use_old_weights
#             )

#         if forcing_input:
#             batch.forcing.tensor = input_clean
#         else:
#             batch.inputs.tensor = input_clean

#         return get_roi_mean(
#             pred_pert.tensor,
#             i_min, i_max, j_min, j_max,
#             t_idx=t_step, feat_idx=output_idx
#         )

#     def _compute_blurred_prediction(
#         self,
#         batch,
#         input_clean: torch.Tensor,
#         input_idx: int,
#         output_idx: int,
#         i_min: int, i_max: int,
#         j_min: int, j_max: int,
#         t_step: int,
#         batch_idx: int,
#         checkpoint,
#         cfg_model, cfg_dataset, cfg_xai,
#         dataset_info, infer_ds,
#         list_run_hour, use_old_weights,
#         device: torch.device,
#         forcing_input: bool
#     ) -> float:
#         """
#         PIC lower bound: model prediction when jet stream
#         spatial structure is destroyed by Gaussian blur.
#         """
#         x = (
#             batch.forcing.tensor[:, 0, ..., input_idx].unsqueeze(-1)
#             if forcing_input
#             else batch.inputs.tensor[:, 0, ..., input_idx].unsqueeze(-1)
#         )

#         x_blurred = spatial_blur(x, sigma=self.blur_sigma).to(device)
#         x_blurred = x_blurred.unsqueeze(1).squeeze(-1)

#         if forcing_input:
#             new_inputs = batch.forcing.tensor.clone()
#             new_inputs[..., input_idx] = x_blurred
#             batch.forcing.tensor = new_inputs
#         else:
#             new_inputs = batch.inputs.tensor.clone()
#             new_inputs[..., input_idx] = x_blurred
#             batch.inputs.tensor = new_inputs

#         with torch.no_grad():
#             pred_blurred = predict_step(
#                 self.model, batch, batch_idx, checkpoint,
#                 cfg_model, cfg_dataset, cfg_xai, dataset_info,
#                 infer_ds, list_run_hour, use_old_weights
#             )

#         if forcing_input:
#             batch.forcing.tensor = input_clean
#         else:
#             batch.inputs.tensor = input_clean

#         return get_roi_mean(
#             pred_blurred.tensor,
#             i_min, i_max, j_min, j_max,
#             t_step, output_idx
#         )

#     # ------------------------------------------------------------------
#     # Main evaluation
#     # ------------------------------------------------------------------

#     def evaluate(
#         self,
#         dataloader,
#         checkpoint,
#         cfg_model,
#         cfg_dataset,
#         cfg_xai,
#         dataset_info,
#         infer_ds,
#         list_run_hour,
#         use_old_weights,
#         target,
#         plot_explanations,
#         extent,
#         fig_path,
#         precip_thresh=None
#     ) -> ROADAggregateResult:
#         """
#         Evaluate ROAD-MSE faithfulness across all valid events.

#         For each valid event:
#           1. Filter: precip > threshold
#           2. For each masking percentage p:
#              a. Mask top-p% salient features → mse_salient
#              b. Average N_random random masks → mse_random
#              c. gap(p) = mse_salient - mse_random
#           3. AUC = trapz(gap_curve) / range

#         Higher AUC = more faithful.
#         Random baseline is 0.0.
#         """
#         # Global seed for reproducibility
#         torch.manual_seed(self.seed)
#         np.random.seed(self.seed)

#         device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#         os.makedirs(
#             cfg_xai["eval"]["eval_plot"]["fig_path"] + "/faithfulness",
#             exist_ok=True
#         )

#         if precip_thresh:
#             self.precip_threshold = precip_thresh
#             print("Precipitation threshold override:", self.precip_threshold)
            
#         forcing_input = cfg_xai["explain"]["forcing_input"]
#         n_pred_steps  = int(cfg_dataset["num_pred_steps_val_test"])
#         all_event_results: List[ROADEventResult] = []
#         print("steps are:", n_pred_steps)
        
#         with torch.autograd.set_grad_enabled(True):

#             for batch_idx, batch in enumerate(dataloader):

#                 # ── ROI indices ───────────────────────────────────────────
#                 H, W = batch.inputs.tensor.shape[2:4]
#                 i_min, i_max, j_min, j_max = compute_roi_indices(
#                     target, extent, H, W
#                 )
#                 output_idx = batch.outputs.feature_names_to_idx[
#                     cfg_xai["explain"]["output"]
#                 ]

#                 # ── Filter 1: precipitation ───────────────────────────────
#                 precip_mean = get_roi_mean(
#                     batch.outputs.tensor,
#                     i_min, i_max, j_min, j_max,
#                     t_idx=0, feat_idx=output_idx
#                 )
#                 if precip_mean <= self.precip_threshold:
#                     continue

#                 # ── Device + gradient setup ───────────────────────────────
#                 batch = self._move_batch_to_device(batch, device)
#                 batch.inputs.tensor.requires_grad_()
#                 runtime = self._get_runtime(batch_idx, infer_ds)

#                 # ── Attribution map ───────────────────────────────────────
#                 explain_clean, output_clean = self.explainer.compute_explanations(
#                     batch, batch_idx, checkpoint,
#                     cfg_model, cfg_dataset, cfg_xai, dataset_info,
#                     infer_ds, list_run_hour, use_old_weights,
#                     target,
#                     plot_explanations=plot_explanations,
#                     extent=extent,
#                     fig_path=fig_path,
#                     return_output=True
#                 )
#                 if explain_clean is None:
#                     continue

#                 # ── Input bookkeeping ─────────────────────────────────────
#                 if forcing_input:
#                     input_clean = batch.forcing.tensor.clone()
#                     input_idx   = batch.forcing.feature_names_to_idx[
#                         cfg_xai["explain"]["input"]
#                     ]
#                     vals = batch.forcing.tensor[..., input_idx]
#                 else:
#                     input_clean = batch.inputs.tensor.clone()
#                     input_idx   = batch.inputs.feature_names_to_idx[
#                         cfg_xai["explain"]["input"]
#                     ]
#                     vals = batch.inputs.tensor[..., input_idx]

#                 noise_std = self.noise_std * (vals.max() - vals.min()).item()
#                 t_step    = n_pred_steps - 1

#                 # ── Clean prediction (PIC upper bound) ────────────────────
#                 pred_clean_roi = get_roi_mean(
#                     output_clean.tensor,
#                     i_min, i_max, j_min, j_max,
#                     t_idx=t_step, feat_idx=output_idx
#                 )

#                 # ── Attribution normalisation ─────────────────────────────
#                 explain_t = explain_clean[0]
#                 explain_t = normalize_expl(
#                     explain_t if explain_t.dim() == 2
#                     else explain_t[..., 0].squeeze()
#                 )
#                 explain_flat = explain_t.reshape(1, -1)

#                 x = (
#                     batch.forcing.tensor[:, 0, ..., input_idx].unsqueeze(-1)
#                     if forcing_input
#                     else batch.inputs.tensor[:, 0, ..., input_idx].unsqueeze(-1)
#                 )

#                 # ── Masking loop (ROAD-MSE) ───────────────────────────────
#                 curve_y_gap     = []
#                 valid_event     = True

#                 for p_idx, p in enumerate(self.percentages):

#                     top_k = int(p / 100 * H * W)
#                     if top_k == 0:
#                         curve_y_gap.append(0.0)
#                         continue

#                     # Salient mask
#                     _, idx_salient = torch.topk(explain_flat, top_k, dim=1)

#                     salient_seed = self.seed + batch_idx * 1000 + p_idx

#                     pred_salient = self._run_mask(
#                         idx=idx_salient,
#                         x=x,
#                         batch=batch,
#                         input_clean=input_clean,
#                         input_idx=input_idx,
#                         output_idx=output_idx,
#                         i_min=i_min, i_max=i_max,
#                         j_min=j_min, j_max=j_max,
#                         t_step=t_step,
#                         H=H, W=W,
#                         noise_std=noise_std,
#                         batch_idx=batch_idx,
#                         checkpoint=checkpoint,
#                         cfg_model=cfg_model,
#                         cfg_dataset=cfg_dataset,
#                         cfg_xai=cfg_xai,
#                         dataset_info=dataset_info,
#                         infer_ds=infer_ds,
#                         list_run_hour=list_run_hour,
#                         use_old_weights=use_old_weights,
#                         device=device,
#                         forcing_input=forcing_input,
#                         seed=salient_seed
#                     )

#                     mse_salient = (pred_clean_roi - pred_salient) ** 2

#                     # Random baseline: N_random masks, different seed each
#                     random_mses = []
#                     for r in range(self.n_random):
#                         random_seed = self.seed + batch_idx * 1000 + p_idx * 100 + r
#                         torch.manual_seed(random_seed)
#                         idx_random = torch.stack([
#                             torch.randperm(H * W)[:top_k]
#                         ])

#                         pred_r = self._run_mask(
#                             idx=idx_random,
#                             x=x,
#                             batch=batch,
#                             input_clean=input_clean,
#                             input_idx=input_idx,
#                             output_idx=output_idx,
#                             i_min=i_min, i_max=i_max,
#                             j_min=j_min, j_max=j_max,
#                             t_step=t_step,
#                             H=H, W=W,
#                             noise_std=noise_std,
#                             batch_idx=batch_idx,
#                             checkpoint=checkpoint,
#                             cfg_model=cfg_model,
#                             cfg_dataset=cfg_dataset,
#                             cfg_xai=cfg_xai,
#                             dataset_info=dataset_info,
#                             infer_ds=infer_ds,
#                             list_run_hour=list_run_hour,
#                             use_old_weights=use_old_weights,
#                             device=device,
#                             forcing_input=forcing_input,
#                             seed=random_seed + 999 
#                         )
#                         random_mses.append((pred_clean_roi - pred_r) ** 2)

#                     mse_random = float(np.mean(random_mses))

#                     # Calculate the MSE Gap
#                     mse_gap = mse_salient - mse_random
#                     curve_y_gap.append(mse_gap)

#                 # ── AUC ───────────────────────────────────────────────────
#                 if not valid_event or len(curve_y_gap) != len(self.percentages):
#                     continue

#                 curve_x = np.array(self.percentages, dtype=float)
#                 curve_y = np.array(curve_y_gap,      dtype=float)
                
#                 # Unbounded AUC of the MSE gap
#                 auc = (
#                     np.trapezoid(curve_y, curve_x)
#                     / (curve_x.max() - curve_x.min())
#                 )

#                 # Pass curve_y as the gap curve so existing plot/aggregate logic works natively
#                 all_event_results.append(ROADEventResult(
#                     runtime=runtime,
#                     auc=float(auc),
#                     curve_x=curve_x,
#                     curve_y=curve_y,
#                     pred_clean=pred_clean_roi,
#                     precip_mean=precip_mean,
#                 ))

#                 print(
#                     f"[ROAD-MSE] {runtime} | "
#                     f"precip={precip_mean:.3f} | "
#                     f"Gap AUC={auc:.4f}"
#                 )

#         return self._aggregate(all_event_results)

#     # ------------------------------------------------------------------
#     # Aggregation
#     # ------------------------------------------------------------------

#     def _aggregate(
#         self,
#         event_results: List[ROADEventResult]
#     ) -> ROADAggregateResult:
#         """
#         Aggregate per-event results.
#         Reports SEM for paper table.
#         """
#         if not event_results:
#             raise ValueError(
#                 "No valid events. Check precip_threshold and gap_threshold."
#             )

#         aucs   = np.array([r.auc for r in event_results])
#         curves = np.stack([r.curve_y for r in event_results], axis=0)

#         n        = len(aucs)
#         mean_auc = float(np.mean(aucs))
#         std_auc  = float(np.std(aucs))
#         sem_auc  = float(std_auc / np.sqrt(n))

#         print(f"\n[ROAD-MSE] Aggregated over {n} valid events")
#         print(f"  Gap AUC mean : {mean_auc:.4f}  (Higher is better, 0.0 is chance)")
#         print(f"  Gap AUC std  : {std_auc:.4f}  (event-to-event variability)")
#         print(f"  Gap AUC SEM  : {sem_auc:.4f}  (report in paper table)")

#         return ROADAggregateResult(
#             mean_auc=mean_auc,
#             std_auc=std_auc,
#             sem_auc=sem_auc,
#             curve_x=event_results[0].curve_x,
#             curve_y_mean=np.mean(curves, axis=0),
#             curve_y_std=np.std(curves,  axis=0),
#             n_events=n,
#             event_results=event_results
#         )

#     # ------------------------------------------------------------------
#     # Stratified analysis
#     # ------------------------------------------------------------------

#     def stratified_auc(
#         self,
#         result: ROADAggregateResult,
#         low_threshold:  float = 0.10,
#         high_threshold: float = 0.40
#     ) -> Dict:
#         """
#         Break AUC down by precipitation intensity.
#         """
#         events = result.event_results
#         weak   = [r for r in events
#                   if low_threshold < r.precip_mean <= high_threshold]
#         strong = [r for r in events if r.precip_mean > high_threshold]

#         def _summarise(subset, label):
#             if not subset:
#                 print(f"[stratified] No events in '{label}'")
#                 return {"mean": None, "sem": None, "n": 0}
#             aucs = np.array([r.auc for r in subset])
#             return {
#                 "mean": float(np.mean(aucs)),
#                 "sem":  float(np.std(aucs) / np.sqrt(len(aucs))),
#                 "n":    len(aucs)
#             }

#         return {
#             "weak":   _summarise(weak,   "weak"),
#             "strong": _summarise(strong, "strong"),
#             "all":    _summarise(events, "all")
#         }



############################ ROAD WITH COSINE DISTANCE ##################

class ROADCosine:
    """
    Cosine ROAD — Structural Faithfulness Metric.
    
    Measures whether masking the salient region
    causes a larger STRUCTURAL change in the 
    attribution map than masking random pixels.
    
    For each masking percentage p:
      1. Compute clean attribution A(x)
      2. Compute attribution under salient masking A(x_sal)
      3. Compute attribution under K random maskings A(x_rand_k)
      4. binary(p) = 1 if d_cos(A(x), A(x_sal)) > 
                          mean_k d_cos(A(x), A(x_rand_k))
    
    This measures structural faithfulness:
    "Does masking salient pixels change the 
     attribution geometry more than random masking?"
    
    Interpretation:
      AUC = 1.0 → attribution always structurally 
                  changes more under salient masking
                  → geometrically faithful
      AUC = 0.5 → chance level
      AUC = 0.0 → random masking changes structure more
                  → attribution is not geometrically faithful
    
    Key difference from scalar ROAD:
      Scalar ROAD: measures prediction degradation
      Cosine ROAD: measures attribution structure change
      These are complementary: a method can be 
      prediction-faithful but not structure-faithful.
    """

    def __init__(
        self,
        model: torch.nn.Module,
        explainer: str = "BaseGrad",
        precip_threshold: float = 5.0,
        std_noise: float = 0.1,
        n_random: int = 5,
        percentages: Optional[List[int]] = None,
        seed: int = 42,
        **expl_params
    ):
        self.model            = model
        self.precip_threshold = precip_threshold
        self.noise_std        = std_noise
        print("std_noise: ", self.noise_std)

        self.n_random         = n_random
        self.seed             = seed
        self.explainer_name   = explainer
        self.percentages      = list(range(1, 10)) + list(range(10, 21, 1))
    
        self.explainer = self._init_explainer(explainer, expl_params)

    def _init_explainer(self, name: str, params: Dict):
        cls = explainers_registry.get(name)
        if cls is None:
            raise ValueError(f"Unknown explainer: {name}")
        return cls(self.model, **params)

    def _get_runtime(self, batch_idx: int, infer_ds) -> str:
        return (
            infer_ds.sample_list[batch_idx]
            .timestamps.datetime
            .strftime("%Y%m%d%H")
        )

    def _move_batch_to_device(self, batch, device):
        batch.inputs.tensor  = batch.inputs.tensor.to(device)
        batch.forcing.tensor = batch.forcing.tensor.to(device)
        if batch.outputs is not None:
            batch.outputs.tensor = batch.outputs.tensor.to(device)
        return batch

    # Cosine degradation (full output field):
    @staticmethod
    def _cosine_degradation(
        pred_clean: torch.Tensor,
        pred_masked: torch.Tensor,
        output_idx: int,
        t_step: int
    ) -> float:
        """
        Cosine distance between clean and masked 
        MODEL OUTPUT FIELDS (not attributions).
        
        Measures structural change in the full 
        precipitation forecast, not just ROI scalar.
        """
        # Extract full spatial output field
        field_clean  = pred_clean[:, t_step, ..., output_idx]
                    # shape: [B, H, W]
        field_masked = pred_masked[:, t_step, ..., output_idx]
        
        c_flat = field_clean.reshape(-1).float()
        m_flat = field_masked.reshape(-1).float()
        
        norm_c = torch.norm(c_flat) + 1e-8
        norm_m = torch.norm(m_flat) + 1e-8
        
        sim  = torch.dot(c_flat, m_flat) / (norm_c * norm_m)
        dist = 1.0 - sim.item()
        
        return float(np.clip(dist, 0.0, 1.0))
    @staticmethod
    def _cosine_distance(
        a: torch.Tensor,
        b: torch.Tensor
    ) -> float:
        """
        Cosine distance between two attribution maps.
        Both inputs normalised to unit vectors before comparison.
        Returns value in [0, 1]:
          0 = identical structure
          1 = completely different structure
        """
        a_flat = a.reshape(-1).float()
        b_flat = b.reshape(-1).float()
        norm_a = torch.norm(a_flat) + 1e-8
        norm_b = torch.norm(b_flat) + 1e-8
        sim    = torch.dot(a_flat, b_flat) / (norm_a * norm_b)
        return float(np.clip(1.0 - sim.item(), 0.0, 1.0))

    def _apply_mask_and_explain(
        self,
        idx: torch.Tensor,
        x: torch.Tensor,
        batch,
        input_clean: torch.Tensor,
        input_idx: int,
        H: int, W: int,
        noise_std: float,
        batch_idx: int,
        checkpoint,
        cfg_model, cfg_dataset, cfg_xai,
        dataset_info, infer_ds,
        list_run_hour, use_old_weights,
        device: torch.device,
        forcing_input: bool,
        seed: Optional[int] = None,
        target=None,
        extent=None,
        fig_path=None
    ) -> Optional[torch.Tensor]:
        """
        Apply pixel mask to input, recompute attribution map,
        return normalised flat attribution vector.
        
        Returns None if explainer fails.
        """
        B = 1
        mask = torch.zeros(B, H * W, device=x.device)
        mask.scatter_(1, idx.to(x.device), 1.0)
        mask = mask.view(B, H, W, 1)

        x_pert = linear_imputation(
            x, mask.to(device),
            noise_std=noise_std,
            seed=seed
        ).unsqueeze(1).squeeze(-1)

        # Apply masked input
        if forcing_input:
            new_inputs = batch.forcing.tensor.clone()
            new_inputs[..., input_idx] = x_pert
            batch.forcing.tensor = new_inputs
        else:
            new_inputs = batch.inputs.tensor.clone()
            new_inputs[..., input_idx] = x_pert
            batch.inputs.tensor = new_inputs
            batch.inputs.tensor.requires_grad_()

        # Recompute attribution on masked input
        try:
            explain_masked = self.explainer.compute_explanations(
                batch, batch_idx, checkpoint,
                cfg_model, cfg_dataset, cfg_xai,
                dataset_info, infer_ds,
                list_run_hour, use_old_weights,
                target,
                plot_explanations=False,
                extent=extent,
                fig_path=fig_path,
                return_output=False
            )
        finally:
            # Always restore
            if forcing_input:
                batch.forcing.tensor = input_clean
            else:
                batch.inputs.tensor = input_clean

        if explain_masked is None:
            return None

        # Normalise to flat unit vector
        raw = explain_masked[0]
        a = normalize_expl(raw.reshape(512,640))
       

        return a.reshape(1, -1)

    # ------------------------------------------------------------------
    # Main evaluation
    # ------------------------------------------------------------------
    def _run_mask_full(
        self,
        idx,
        x, batch, input_clean,
        input_idx, output_idx,
        i_min, i_max, j_min, j_max,
        t_step, H, W, noise_std,
        batch_idx, checkpoint,
        cfg_model, cfg_dataset, cfg_xai,
        dataset_info, infer_ds,
        list_run_hour, use_old_weights,
        device, forcing_input, seed=None
    ) -> torch.Tensor:
        """
        Same as _run_mask but returns FULL prediction tensor
        instead of ROI scalar mean.
        """
        B = 1
        mask = torch.zeros(B, H * W, device=x.device)
        mask.scatter_(1, idx.to(x.device), 1.0)
        mask = mask.view(B, H, W, 1)

        x_pert = linear_imputation(
            x, mask.to(device),
            noise_std=noise_std,
            seed=seed
        ).unsqueeze(1).squeeze(-1)

        if forcing_input:
            new_inputs = batch.forcing.tensor.clone()
            new_inputs[..., input_idx] = x_pert
            batch.forcing.tensor = new_inputs
        else:
            new_inputs = batch.inputs.tensor.clone()
            new_inputs[..., input_idx] = x_pert
            batch.inputs.tensor = new_inputs

        with torch.no_grad():
            pred_pert = predict_step(
                self.model, batch, batch_idx, checkpoint,
                cfg_model, cfg_dataset, cfg_xai, dataset_info,
                infer_ds, list_run_hour, use_old_weights
            )

        if forcing_input:
            batch.forcing.tensor = input_clean
        else:
            batch.inputs.tensor = input_clean

        # Return FULL tensor for cosine computation
        return pred_pert.tensor[:, t_step, :,:, output_idx]
    def evaluate(
    self,
    dataloader,
    checkpoint,
    cfg_model, cfg_dataset, cfg_xai,
    dataset_info, infer_ds,
    list_run_hour, use_old_weights,
    target, plot_explanations,
    extent, fig_path,
    valid_runtimes=None
) -> ROADAggregateResult:
        """
        Evaluate cosine ROAD across all valid events.
        
        Compares MODEL PREDICTIONS (not attributions) via cosine distance.
        
        For each masking percentage p:
        d_cos(Y_clean, Y_salient) vs d_cos(Y_clean, Y_random)
        
        where Y = full precipitation forecast field [H, W].
        Higher cosine distance = masking caused larger structural
        change in forecast = attribution is more faithful.
        """
        torch.manual_seed(self.seed)
        np.random.seed(self.seed)

        device        = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        forcing_input = cfg_xai["explain"]["forcing_input"]
        n_pred_steps  = int(cfg_dataset["num_pred_steps_val_test"])
        all_event_results = []

        os.makedirs(
            cfg_xai["eval"]["eval_plot"]["fig_path"] + "/faithfulness_cosine",
            exist_ok=True
        )

        with torch.autograd.set_grad_enabled(True):

            for batch_idx, batch in enumerate(dataloader):

                # ── Runtime filter ─────────────────────────────────────
                runtime = self._get_runtime(batch_idx, infer_ds)

                # Fix Bug 2: uncomment valid_runtimes filter
                if valid_runtimes is not None:
                    if runtime not in valid_runtimes:
                        continue

                # ── ROI setup ──────────────────────────────────────────
                H, W = batch.inputs.tensor.shape[2:4]
                i_min, i_max, j_min, j_max = compute_roi_indices(
                    target, extent, H, W
                )
                output_idx = batch.outputs.feature_names_to_idx[
                    cfg_xai["explain"]["output"]
                ]

                # ── Precipitation filter ───────────────────────────────
                precip_mean = get_roi_mean(
                    batch.outputs.tensor,
                    i_min, i_max, j_min, j_max,
                    t_idx=0, feat_idx=output_idx
                )
                if valid_runtimes is None:
                    if precip_mean < self.precip_threshold:
                        continue

                # ── Device setup ───────────────────────────────────────
                batch = self._move_batch_to_device(batch, device)
                batch.inputs.tensor.requires_grad_()

                # ── Clean attribution + clean prediction ───────────────
                # Fix Bug 3: capture output_clean
                explain_clean, output_clean = self.explainer.compute_explanations(
                    batch, batch_idx, checkpoint,
                    cfg_model, cfg_dataset, cfg_xai, dataset_info,
                    infer_ds, list_run_hour, use_old_weights,
                    target,
                    plot_explanations=plot_explanations,
                    extent=extent,
                    fig_path=fig_path,
                    return_output=True
                )
                if explain_clean is None or output_clean is None:
                    continue

                # ── Input bookkeeping ──────────────────────────────────
                if forcing_input:
                    input_clean = batch.forcing.tensor.clone()
                    input_idx   = batch.forcing.feature_names_to_idx[
                        cfg_xai["explain"]["input"]
                    ]
                    vals = batch.forcing.tensor[..., input_idx]
                else:
                    input_clean = batch.inputs.tensor.clone()
                    input_idx   = batch.inputs.feature_names_to_idx[
                        cfg_xai["explain"]["input"]
                    ]
                    vals = batch.inputs.tensor[..., input_idx]

                noise_std = self.noise_std * (vals.max() - vals.min()).item()
                t_step    = n_pred_steps - 1

                # ── Clean forecast field (for cosine comparison) ───────
                # Shape: [H, W] — full spatial precipitation field
                field_clean = output_clean.tensor[
                    0, t_step, :, :, output_idx
                ].detach()

                # ── Attribution map for pixel ranking ─────────────────
                # Fix Bug 1: use H, W not hardcoded 512,640
                raw_clean    = explain_clean[0]
                a_clean      = normalize_expl(raw_clean.reshape(H, W))
                explain_flat = a_clean.reshape(1, -1)

                x = (
                    batch.forcing.tensor[:, 0, ..., input_idx].unsqueeze(-1)
                    if forcing_input
                    else batch.inputs.tensor[:, 0, ..., input_idx].unsqueeze(-1)
                )

                # ── Masking loop ───────────────────────────────────────
                curve_y     = []
                valid_event = True

                for p_idx, p in enumerate(self.percentages):

                    top_k = int(p / 100 * H * W)
                    if top_k == 0:
                        curve_y.append(0)
                        continue

                    # ── Salient mask: run model, get forecast field ────
                    _, idx_salient = torch.topk(
                        explain_flat, top_k, dim=1
                    )
                    salient_seed = self.seed + batch_idx * 1000 + p_idx

                    # Fix Bug 4: use _run_mask_full not _apply_mask_and_explain
                    field_salient = self._run_mask_full(
                        idx=idx_salient,
                        x=x,
                        batch=batch,
                        input_clean=input_clean,
                        input_idx=input_idx,
                        output_idx=output_idx,
                        t_step=t_step,i_min=i_min,i_max=i_max,j_min=j_min,j_max=j_max,
                        H=H, W=W,
                        noise_std=noise_std,
                        batch_idx=batch_idx,
                        checkpoint=checkpoint,
                        cfg_model=cfg_model,
                        cfg_dataset=cfg_dataset,
                        cfg_xai=cfg_xai,
                        dataset_info=dataset_info,
                        infer_ds=infer_ds,
                        list_run_hour=list_run_hour,
                        use_old_weights=use_old_weights,
                        device=device,
                        forcing_input=forcing_input,
                        seed=salient_seed
                    )

                    if field_salient is None:
                        valid_event = False
                        break

                    # Cosine distance between forecast FIELDS (predictions)
                    dist_salient = self._cosine_distance(
                        field_clean.reshape(1, -1),
                        field_salient.reshape(1, -1)
                    )

                    # ── Random masks: run model, get forecast fields ───
                    # dist_randoms = []
                    # for r in range(self.n_random):

                    #     random_seed = (
                    #         self.seed
                    #         + batch_idx * 1000
                    #         + p_idx * 100
                    #         + r
                    #     )
                    #     torch.manual_seed(random_seed)
                    #     idx_random = torch.stack([
                    #         torch.randperm(H * W)[:top_k]
                    #     ])

                    #     field_random = self._run_mask_full(
                    #         idx=idx_random,
                    #         x=x,
                    #         batch=batch,
                    #         input_clean=input_clean,
                    #         input_idx=input_idx,
                    #         output_idx=output_idx,
                    #         t_step=t_step,i_min=i_min,i_max=i_max,j_min=j_min,j_max=j_max,
                    #         H=H, W=W,
                    #         noise_std=noise_std,
                    #         batch_idx=batch_idx,
                    #         checkpoint=checkpoint,
                    #         cfg_model=cfg_model,
                    #         cfg_dataset=cfg_dataset,
                    #         cfg_xai=cfg_xai,
                    #         dataset_info=dataset_info,
                    #         infer_ds=infer_ds,
                    #         list_run_hour=list_run_hour,
                    #         use_old_weights=use_old_weights,
                    #         device=device,
                    #         forcing_input=forcing_input,
                    #         seed=random_seed + 999
                    #     )

                    #     if field_random is None:
                    #         continue

                    #     dist_randoms.append(
                    #         self._cosine_distance(
                    #             field_clean.reshape(1, -1),
                    #             field_random.reshape(1, -1)
                    #         )
                    #     )

                    # if not dist_randoms:
                    #     valid_event = False
                    #     break

                    # dist_random_mean = float(np.mean(dist_randoms))

                    # ── Binary comparison ──────────────────────────────
                    # Salient masking should cause larger structural change
                    # in the forecast field than random masking
                    # binary = 1 if dist_salient > dist_random_mean else 0
                    curve_y.append(dist_salient)

                    # print(
                    #     f"  p={p:5.1f}% | "
                    #     f"d_sal={dist_salient:.5f} | "
                    # )

                # ── AUC ────────────────────────────────────────────────
                if not valid_event or len(curve_y) != len(self.percentages):
                    continue

                curve_x = np.array(self.percentages, dtype=float)
                curve_y = np.array(curve_y,          dtype=float)
                auc = (
                    np.trapz(curve_y, curve_x)
                    / (curve_x.max() - curve_x.min())
                )

                all_event_results.append(ROADEventResult(
                    runtime=runtime,
                    auc=float(auc),
                    curve_x=curve_x,
                    curve_y=curve_y,
                    pred_clean=0.0,
                    precip_mean=precip_mean,
                ))

                print(
                    f"[ROAD-Cosine] {runtime} | "
                    f"precip={precip_mean:.3f} | "
                    f"AUC={auc:.4f}"
                )

        return self._aggregate(all_event_results)

    # ------------------------------------------------------------------
    # Aggregation (identical to scalar ROAD)
    # ------------------------------------------------------------------

    def _aggregate(
        self,
        event_results: List[ROADEventResult]
    ) -> ROADAggregateResult:
        if not event_results:
            raise ValueError("No valid events.")

        aucs   = np.array([r.auc for r in event_results])
        curves = np.stack([r.curve_y for r in event_results], axis=0)

        n        = len(aucs)
        mean_auc = float(np.mean(aucs))
        std_auc  = float(np.std(aucs))
        sem_auc  = float(std_auc / np.sqrt(n))

        print(f"\n[ROAD-Cosine] Aggregated over {n} valid events")
        print(f"  AUC mean : {mean_auc:.4f}")
        print(f"  AUC std  : {std_auc:.4f}")
        print(f"  AUC SEM  : {sem_auc:.4f}  (report in paper table)")

        return ROADAggregateResult(
            mean_auc=mean_auc,
            std_auc=std_auc,
            sem_auc=sem_auc,
            curve_x=event_results[0].curve_x,
            curve_y_mean=np.mean(curves, axis=0),
            curve_y_std=np.std(curves,  axis=0),
            n_events=n,
            event_results=event_results
        )
