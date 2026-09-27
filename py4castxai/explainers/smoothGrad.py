from py4castxai.explainers.XGrad import XGrad
import torch
from py4castxai.utils_test.infer_utils import predict_step

from py4castxai.explainers.inputxgrad import InputxGrad
from py4castxai.explainers.basegrad import BaseGrad
import os
import numpy as np
import torch
def normalize_sum(a):
    a = torch.abs(a)  # ensure non-negative
    return a / (a.sum() + 1e-8)
import torch
import torch.nn.functional as F

def gaussian_blur_baseline(measures, sigma_px, renormalize=True):
    """
    Gaussian-blur control for the Wasserstein barycenter.

    measures : (N, H, W) tensor of per-sample spatial measures, each already
               |G_i| / sum|G_i| (i.e. the SAME mu_i that WGBary aggregates).
    sigma_px : Gaussian std in PIXELS. This is the swept parameter.

    Returns a single (H, W) measure: blur( mean_i mu_i ), renormalized to sum 1,
    so it is directly comparable to WGBary on the same metrics.
    """
    mean_measure = measures.mean(dim=0)                      # (H, W): arithmetic mean of mu_i

    # build a separable Gaussian kernel
    radius = max(1, int(round(3 * sigma_px)))
    coords = torch.arange(-radius, radius + 1, dtype=mean_measure.dtype,
                          device=mean_measure.device)
    k1d = torch.exp(-(coords ** 2) / (2 * sigma_px ** 2))
    k1d = k1d / k1d.sum()

    x = mean_measure[None, None]                             # (1,1,H,W)
    kx = k1d.view(1, 1, 1, -1)
    ky = k1d.view(1, 1, -1, 1)
    x = F.conv2d(x, kx, padding=(0, radius))                 # blur along W
    x = F.conv2d(x, ky, padding=(radius, 0))                 # blur along H
    blurred = x[0, 0]

    if renormalize:
        blurred = blurred / (blurred.sum() + 1e-12)          # keep it a probability measure
    return blurred
class SmoothGrad(XGrad):


    def __init__(self,model,base_explainer="BaseGrad",std_perturbations=0.1,num_perturbations=50):
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
            print("an hna",std_perturbation)
            for perturbation_idx in range(self.num_perturbations): 
                
                # input.inputs.tensor =input_clean+ torch.randn_like(input.inputs.tensor)*std_perturbation
                input_perturbed = input_clean.clone().detach() + torch.randn_like(input_clean) * std_perturbation
                
                input.forcing.tensor = input.forcing.tensor.clone()
                input.forcing.tensor[..., input_idx] = input_perturbed
                gradient,_ = self.base_explainer.compute_explanations(input=input,batch_idx=batch_idx,checkpoint=checkpoint,
                                                                      cfg_model=cfg_model,cfg_dataset=cfg_dataset,cfg_xai=cfg_xai,
                                                                      dataset_info=dataset_info,infer_ds=infer_ds,list_run_hour=list_run_hour,use_old_weights=use_old_weights,target=target,plot_explanations=False,
                                                                      extent=extent,fig_path=fig_path,return_output=True)

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
            stats = dataset_info.stats
            
            std_perturbation = self.std_perturbation * (vals.max() - vals.min())
            for perturbation_idx in range(self.num_perturbations): 
                
                input_perturbed = input_clean.clone().detach() + torch.randn_like(input_clean) * std_perturbation
                
       
                # # Continue with gradient computation
                # if cfg_xai["explain"]["forcing_input"]:
                #     input.forcing.tensor = input.forcing.tensor.clone()
                #     input.forcing.tensor[..., input_idx] = input_perturbed
                # else:
                #     input.inputs.tensor = input.inputs.tensor.clone()
                #     input.inputs.tensor[..., input_idx] = input_perturbed
                

            
                input.inputs.tensor = backup_tensor.clone()
                input.inputs.tensor[..., input_idx] = input_perturbed
                gradient,_ = self.base_explainer.compute_explanations(input=input,batch_idx=batch_idx,checkpoint=checkpoint,cfg_model=cfg_model,cfg_dataset=cfg_dataset,
                                                                      cfg_xai=cfg_xai,dataset_info=dataset_info,infer_ds=infer_ds,list_run_hour=list_run_hour,
                                                                      use_old_weights=use_old_weights,target=target,plot_explanations=False,extent=extent,
                                                                      fig_path=fig_path,return_output=True)

                if gradient is None:
                    if return_output:
                         return None,None
                    return
                
                for i in range(len(gradient)):
                    # grads[i].append(gradient[i].squeeze(-1))
                    grads[i].append(gradient[i].squeeze(-1))
                    # grads[i].append(normalize_sum(gradient[i]).squeeze(-1))
          
          
        grad_list = []
        if not cfg_xai["explain"]["end_to_end"]:
            for i in range(input.num_pred_steps):
                
                grad_list.append(torch.stack(grads[i]).mean(dim=0).unsqueeze(-1))
        
        else: 
            grad_list.append(torch.stack(grads[0]).mean(dim=0).unsqueeze(-1))
        # grad_list = []
        # steps = range(input.num_pred_steps) if not cfg_xai["explain"]["end_to_end"] else [0]
        # for i in steps:
        #     stacked = torch.stack(grads[i]).squeeze(1).squeeze(1)   # (20, 512, 640)
        #     blurred = gaussian_blur_baseline(stacked, sigma_px=8)   # (512, 640)
        #     # restore the leading (batch, step) dims and trailing channel dim to match SmoothGrad's output
        #     blurred = blurred.unsqueeze(0).unsqueeze(0).unsqueeze(-1)   # (1, 1, 512, 640, 1)
        #     grad_list.append(blurred)


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
            grad_list[0] = grad_list[0].squeeze(1)
            if cfg_xai["explain"]["end_to_end"]:
                    step = input.num_pred_steps
                    fig_path_step =fig_path+ f"/end_to_end/{step}_step/input_{cfg_xai['explain']['input']}_output_{cfg_xai['explain']['output']}"
                                        
                    self.plot_explanations(grad_list[0],input,output,
                                            extent,"SmoothGrad",fig_path_step,
                                            runtime=runtimes[0],stats=stats,step=step-1,
                                            input_name = cfg_xai["explain"]["input"],
                                            output_name = cfg_xai["explain"]["output"],
                                        input_idx=input_idx,output_idx = output_idx,target=cfg_xai["target"],forcing=cfg_xai["explain"]["forcing_input"],save_figs=cfg_xai["explain"]["exp_plot"]["save_figs"])
                
            else: 
                for i in range(len(grad_list)):
                    output_idx =output.feature_names_to_idx[cfg_xai["explain"]["output"]] 
                    if cfg_xai["explain"]["forcing_input"]:
                        input_idx = input.forcing.feature_names_to_idx[cfg_xai["explain"]["input"]]
                    else:
                        input_idx = input.inputs.feature_names_to_idx[cfg_xai["explain"]["input"]]
                    fig_path_step =fig_path+ f"/{i+1}_step/input_{cfg_xai['explain']['input']}_output_{cfg_xai['explain']['output']}"
                    self.plot_explanations(grad_list[i],input,output,
                                            extent,"SmoothGrad",fig_path_step,
                                            runtime=runtimes[0],stats=stats,step=i,
                                            input_name = cfg_xai["explain"]["input"],
                                            output_name = cfg_xai["explain"]["output"],
                                            input_idx=input_idx,output_idx = output_idx,target=cfg_xai["target"],forcing=cfg_xai["explain"]["forcing_input"],save_figs=cfg_xai["explain"]["exp_plot"]["save_figs"])


        if return_output:

            return  grad_list, output         
        
        return grad_list



import torch
import scipy.stats

class StatGrad(XGrad):

    def __init__(self, model, base_explainer="BaseGrad", std_perturbations=0.1, num_perturbations=50, confidence_level=0.999):
        super().__init__(model)
        if base_explainer == "InputxGrad":
            self.base_explainer = InputxGrad(model)
        else:
            self.base_explainer = BaseGrad(model)
        
        self.std_perturbation = std_perturbations
        self.num_perturbations = num_perturbations
        self.confidence_level = confidence_level
        
        # Calculate critical t-value based on confidence level and degrees of freedom (N-1)
        # Using two-tailed test
        alpha = 1.0 - self.confidence_level
        degrees_of_freedom = self.num_perturbations - 1
        self.t_crit = scipy.stats.t.ppf(1.0 - alpha / 2.0, degrees_of_freedom)
        print(f"StatGrad initialized with t_crit={self.t_crit:.4f} for confidence={self.confidence_level*100}%")

    def compute_explanations(self,
                             input, batch_idx, checkpoint, cfg_model, cfg_dataset,
                             cfg_xai, dataset_info, infer_ds, list_run_hour, use_old_weights,
                             target, plot_explanations=True,
                             extent=[-12, 16, 37.5, 55.4],
                             fig_path=".figs/attributions", return_output=False):
        
        output_idx = input.outputs.feature_names_to_idx[cfg_xai["explain"]["output"]]
        grads = {i: [] for i in range(input.num_pred_steps)}
     
        output = predict_step(self.model, input, batch_idx, checkpoint, cfg_model, cfg_dataset, cfg_xai, dataset_info, infer_ds, list_run_hour, use_old_weights, compute_grads=False)
        
        if output is None:
            if return_output:
                return None, None
            return

        if cfg_xai["explain"]["forcing_input"]:
            backup_tensor = input.forcing.tensor
            input_idx = input.forcing.feature_names_to_idx[cfg_xai["explain"]["input"]]
            input_clean = input.forcing.tensor[..., input_idx]
            vals = input.forcing.tensor[..., input_idx]
            std_perturbation = self.std_perturbation * (vals.max() - vals.min())   
            
            for perturbation_idx in range(self.num_perturbations): 
                input_perturbed = input_clean.clone().detach() + torch.randn_like(input_clean) * std_perturbation
                
                input.forcing.tensor = input.forcing.tensor.clone()
                input.forcing.tensor[..., input_idx] = input_perturbed
                gradient, _ = self.base_explainer.compute_explanations(input=input, batch_idx=batch_idx, checkpoint=checkpoint,
                                                                      cfg_model=cfg_model, cfg_dataset=cfg_dataset, cfg_xai=cfg_xai,
                                                                      dataset_info=dataset_info, infer_ds=infer_ds, list_run_hour=list_run_hour, use_old_weights=use_old_weights, target=target, plot_explanations=False,
                                                                      extent=extent, fig_path=fig_path, return_output=True)

                if gradient is None:
                    if return_output:
                        return None, None
                    return
                
                for i in range(len(gradient)):
                    grads[i].append(gradient[i])
                
                self.model.zero_grad(set_to_none=True)
                
        else:
            backup_tensor = input.inputs.tensor
            input_idx = input.inputs.feature_names_to_idx[cfg_xai["explain"]["input"]]
            input_clean = input.inputs.tensor[..., input_idx]
            vals = input.inputs.tensor[..., input_idx] 
            
            std_perturbation = self.std_perturbation * (vals.max() - vals.min())
            for perturbation_idx in range(self.num_perturbations): 
                
                input_perturbed = input_clean.clone().detach() + torch.randn_like(input_clean) * std_perturbation
                
                input.inputs.tensor = backup_tensor.clone()
                input.inputs.tensor[..., input_idx] = input_perturbed
                gradient, _ = self.base_explainer.compute_explanations(input=input, batch_idx=batch_idx, checkpoint=checkpoint, cfg_model=cfg_model, cfg_dataset=cfg_dataset,
                                                                      cfg_xai=cfg_xai, dataset_info=dataset_info, infer_ds=infer_ds, list_run_hour=list_run_hour,
                                                                      use_old_weights=use_old_weights, target=target, plot_explanations=False, extent=extent,
                                                                      fig_path=fig_path, return_output=True)

                if gradient is None:
                    if return_output:
                         return None, None
                    return
                
                for i in range(len(gradient)):
                    grads[i].append(gradient[i].squeeze(-1))
          
        grad_list = []
        
        # Determine which steps to process
        steps_to_process = range(input.num_pred_steps) if not cfg_xai["explain"]["end_to_end"] else [0]
        
        for i in steps_to_process:
            # Stack gradients: Shape will be (num_perturbations, ...)
            stacked_grads = torch.stack(grads[i])
            
            # 1. Calculate Sample Mean
            mean_grad = stacked_grads.mean(dim=0)
            
            # 2. Calculate Sample Variance (unbiased, divide by N-1)
            var_grad = stacked_grads.var(dim=0, unbiased=True)
            
            # 3. Calculate Standard Error (SE) 
            # Added a small epsilon (1e-8) to avoid division by zero in zero-variance pixels
            se_grad = torch.sqrt(var_grad / self.num_perturbations) + 1e-8
            
            # 4. Calculate Absolute T-Statistic for every pixel
            t_stat = torch.abs(mean_grad / se_grad)
            
            # 5. Create Binary Mask (1 if T-statistic > t_crit, else 0)
            mask = (t_stat > self.t_crit).float()
            
            # 6. Apply Mask to the Mean Gradient
            stat_grad = mean_grad * mask
            
            # Re-add the channel dimension removed during squeeze(-1)
            grad_list.append(stat_grad.unsqueeze(-1))

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
            grad_list[0] = grad_list[0].squeeze(1)
            if cfg_xai["explain"]["end_to_end"]:
                    step = input.num_pred_steps
                    fig_path_step = fig_path + f"/end_to_end/{step}_step/input_{cfg_xai['explain']['input']}_output_{cfg_xai['explain']['output']}"
                                        
                    self.plot_explanations(grad_list[0], input, output,
                                            extent, "StatGrad", fig_path_step,
                                            runtime=runtimes[0], stats=stats, step=step-1,
                                            input_name=cfg_xai["explain"]["input"],
                                            output_name=cfg_xai["explain"]["output"],
                                            input_idx=input_idx, output_idx=output_idx, target=cfg_xai["target"], forcing=cfg_xai["explain"]["forcing_input"], save_figs=cfg_xai["explain"]["exp_plot"]["save_figs"])
                
            else: 
                for i in range(len(grad_list)):
                    output_idx = output.feature_names_to_idx[cfg_xai["explain"]["output"]] 
                    if cfg_xai["explain"]["forcing_input"]:
                        input_idx = input.forcing.feature_names_to_idx[cfg_xai["explain"]["input"]]
                    else:
                        input_idx = input.inputs.feature_names_to_idx[cfg_xai["explain"]["input"]]
                    fig_path_step = fig_path + f"/{i+1}_step/input_{cfg_xai['explain']['input']}_output_{cfg_xai['explain']['output']}"
                    
                    self.plot_explanations(grad_list[i], input, output,
                                            extent, "StatGrad", fig_path_step,
                                            runtime=runtimes[0], stats=stats, step=i,
                                            input_name=cfg_xai["explain"]["input"],
                                            output_name=cfg_xai["explain"]["output"],
                                            input_idx=input_idx, output_idx=output_idx, target=cfg_xai["target"], forcing=cfg_xai["explain"]["forcing_input"], save_figs=cfg_xai["explain"]["exp_plot"]["save_figs"])

        if return_output:
            return grad_list, output         
        
        return grad_list




class EDASmoothGrad(XGrad):
    """
    EDA-guided SmoothGrad explainer tailored for atmospheric forecasting models.
    Injects flow-dependent physical perturbations across ALL input variables using 
    Météo-France PEARO ensemble files to preserve cross-variable physical balance.
    """

    def __init__(self, 
                 model, 
                 base_explainer="BaseGrad", 
                 eda_dir=None, 
                 eda_perturbations=None, 
                 scale_factor=1.0, 
                 num_members=17,num_perturbations=20):
        """
        Args:
            model: PyTorch forecasting model.
            base_explainer: "BaseGrad" or "InputxGrad".
            eda_dir: Path to directory containing PEARO .npy files.
            eda_perturbations: Pre-loaded PyTorch tensor of EDA perturbations.
            scale_factor: Multiplier to scale perturbation magnitude (default 1.0).
            num_members: Expected number of members in the .npy file (PEARO defaults to 17).
        """
        super().__init__(model)
        if base_explainer == "InputxGrad":
            self.base_explainer = InputxGrad(model)
        else:
            self.base_explainer = BaseGrad(model)
        
        self.eda_dir = eda_dir
        self.eda_perturbations = eda_perturbations
        self.scale_factor = scale_factor
        self.num_members = num_members
        self.num_perturbations = num_perturbations

    def _map_feature_to_filename(self, py4cast_name):
        """
        Translates py4cast config names (e.g., 'aro_v_250hpa', 'aro_t2m_2m') 
        to PEARO filenames (e.g., 'v250', 't2m').
        """
        # Explicit mappings for surface variables 
        # (Handling both the base names and the py4cast appended level suffixes)
        special_maps = {
            "aro_t2m": "t2m",       "aro_t2m_2m": "t2m",
            "aro_r2": "hu2m",       "aro_r2_2m": "hu2m",
            "aro_tp": "rr",         "aro_tp_0m": "rr",
            "aro_u10": "u10m",      "aro_u10_10m": "u10m",
            "aro_v10": "v10m",      "aro_v10_10m": "v10m"
        }
        
        if py4cast_name in special_maps:
            return special_maps[py4cast_name]
            
        # Dynamic mapping for pressure levels (aro_z_850hpa -> z850)
        pearo_name = py4cast_name.replace("aro_", "").replace("_", "").replace("hpa", "")
        return pearo_name



    def _load_pearo_eda_from_disk(
        self,
        eda_dir,
        timestamp_str,
        feature_names_to_idx,
        target_tensor_shape,
        device,
        subdomain=[100, 612, 240, 880],
        eda_analysis_step=1,     # which timestep in the 2-step files matches the model's input frame
        verbose=False,
    ):
            """
            Load PEARO .npy files (all 17 members) for one timestamp, crop to the model
            subdomain, and assemble a tensor shaped like the model input with a leading
            member axis:  (num_members, B, T, Y, X, C).

            Key fixes vs. the old version
            -----------------------------
            * Explicit timestep selection. The old `if Tfile == T / elif Tfile == 1`
            logic silently left a channel at ZERO whenever Tfile (2) != T (e.g. 1),
            so for a 1-step model every 2-step variable went unperturbed. Here we
            SELECT `eda_analysis_step` from multi-step files and place it into every
            model step, and we RAISE if a file cannot supply the needed step.
            * Per-channel diagnostics. After assembly we print each channel's std; any
            channel reading 0.0 is unperturbed and flagged loudly.
            * Missing files raise by default instead of quietly zeroing a channel
            (a zeroed channel breaks cross-variable balance -- the whole point of EDA).

            Parameters
            ----------
            eda_analysis_step : int
                Index into the file's time axis that corresponds to the single frame the
                model ingests (the analysis time). Resolve this ONCE by value-matching an
                EDA ensemble mean against the model's clean input for the same date, then
                hard-code it. For 1-step files (rr) this argument is ignored.
            """
            B, T, Y_dim, X_dim, C = target_tensor_shape

            if verbose:
                print(f"\n[EDA] loading {timestamp_str}")
                print(f"[EDA] target shape (B,T,Y,X,C) = "
                    f"({B},{T},{Y_dim},{X_dim},{C})  members={self.num_members}")
                print(f"[EDA] analysis step for multi-step files = {eda_analysis_step}")

            # ---- cache lookup -----------------------------------------------------
            # Key on everything that changes the result, including the step choice.
            cache_root = os.path.join(eda_dir, ".eda_cache")
            os.makedirs(cache_root, exist_ok=True)
            cache_key = (f"{timestamp_str}_{B}x{T}x{Y_dim}x{X_dim}x{C}"
                        f"_sub{subdomain}_step{eda_analysis_step}")
            cache_path = os.path.join(cache_root, f"{cache_key}.pt")

            if os.path.exists(cache_path):
                if verbose:
                    print(f"[EDA] cache HIT -> {cache_path}")
                return torch.load(cache_path, map_location=device, weights_only=True)
            if verbose:
                print(f"[EDA] cache MISS -> building from .npy")

            # ---- assemble on CPU, transfer once -----------------------------------
            full_eda = torch.zeros(
                (self.num_members, B, T, Y_dim, X_dim, C), dtype=torch.float32
            )

            if subdomain is not None:
                y_min, y_max, x_min, x_max = subdomain

            # track which channels actually got written so we can catch silent zeros
            written = {}          # var_idx -> var_name
            missing = []          # (var_name, path)

            # deterministic order so prints read cleanly
            items = sorted(feature_names_to_idx.items(), key=lambda kv: kv[1])

            for var_name, var_idx in items:
                pearo_var = self._map_feature_to_filename(var_name)
                file_path = os.path.join(eda_dir, f"{timestamp_str}_{pearo_var}_17.npy")

                if not os.path.exists(file_path):
                    missing.append((var_name, file_path))
                    if verbose:
                        print(f"[EDA]   ch{var_idx:2d} {var_name:16s} -> {pearo_var:6s} "
                            f"[MISSING] {os.path.basename(file_path)}")
                    continue

                # mmap so only the cropped slice is read from disk
                arr = np.load(file_path, mmap_mode="r")           # (Y, X, Tfile, M)
                if subdomain is not None:
                    arr = arr[y_min:y_max, x_min:x_max, :, :]      # (512, 640, Tfile, M)

                # (Y, X, Tfile, M) -> (M, Tfile, Y, X)
                arr = np.ascontiguousarray(np.transpose(arr, (3, 2, 0, 1)))
                t_file = arr.shape[1]

                # ---- select the frame(s) that match the model's T ----------------
                if t_file == 1:
                    # rr: single accumulated frame; broadcast to every model step
                    frame = arr[:, 0:1]                            # (M, 1, Y, X)
                    step_used = "single(0)"
                elif t_file >= T:
                    if not (0 <= eda_analysis_step < t_file):
                        raise ValueError(
                            f"{var_name}: eda_analysis_step={eda_analysis_step} out of "
                            f"range for file with {t_file} steps")
                    s = eda_analysis_step
                    frame = arr[:, s:s + 1]                        # (M, 1, Y, X)
                    step_used = f"idx{s}"
                else:
                    raise ValueError(
                        f"{var_name}: EDA file has {t_file} steps but model needs {T} "
                        f"and no single-frame fallback applies")

                member_check = frame.shape[0]
                if member_check != self.num_members:
                    raise ValueError(
                        f"{var_name}: file has {member_check} members, "
                        f"expected {self.num_members}")

                frame_t = torch.from_numpy(frame)                 # (M, 1, Y, X)
                # place into (M, B, T, Y, X) for this channel: broadcast over B and T
                # frame is one timestep -> repeat across all T model steps
                placed = frame_t.unsqueeze(1).expand(self.num_members, B, T, Y_dim, X_dim)
                full_eda[:, :, :, :, :, var_idx] = placed
                written[var_idx] = var_name

                if verbose:
                    fmin = float(frame.min()); fmax = float(frame.max())
                    print(f"[EDA]   ch{var_idx:2d} {var_name:16s} -> {pearo_var:6s} "
                        f"file_t={t_file} use={step_used:9s} "
                        f"range=[{fmin:8.2f},{fmax:8.2f}]")

            # ---- integrity checks -------------------------------------------------
            if missing:
                names = ", ".join(n for n, _ in missing)
                raise FileNotFoundError(
                    f"[EDA] {len(missing)} variable(s) had no EDA file: {names}. "
                    f"A zeroed channel breaks cross-variable balance. "
                    f"Fix the name mapping or the data, or pass a variable subset.")

            unwritten = [var_idx for var_idx in range(C) if var_idx not in written]
            if unwritten:
                # channels in the tensor with no corresponding feature name -- unusual
                print(f"[EDA] WARNING: channels with no EDA source (stay zero): {unwritten}")

            # per-channel std: the definitive "did every channel get data" check
            if verbose:
                print(f"[EDA] per-channel std after assembly (0.0 == UNPERTURBED):")
                per_ch_std = full_eda.std(dim=(0, 1, 2, 3, 4))    # (C,)
                idx_to_name = {v: k for k, v in feature_names_to_idx.items()}
                n_zero = 0
                for ci in range(C):
                    s = float(per_ch_std[ci])
                    flag = "  <-- ZERO!" if s == 0.0 else ""
                    if s == 0.0:
                        n_zero += 1
                    print(f"[EDA]   ch{ci:2d} {idx_to_name.get(ci,'?'):16s} std={s:.4f}{flag}")
                if n_zero:
                    print(f"[EDA] {n_zero} channel(s) are unperturbed -- investigate before use")
                else:
                    print(f"[EDA] all {C} channels perturbed OK")

            # ---- write cache and return -------------------------------------------
            torch.save(full_eda, cache_path)
            if verbose:
                print(f"[EDA] cached -> {cache_path}")
            return full_eda.to(device)

    def compute_explanations(self,
                             input, batch_idx, checkpoint, cfg_model, cfg_dataset,
                             cfg_xai, dataset_info, infer_ds, list_run_hour, use_old_weights,
                             target, plot_explanations=True,
                             extent=[-12, 16, 37.5, 55.4],
                             fig_path=".figs/attributions", return_output=False,
                             eda_dir=None, eda_perturbations=None):
        
        output_idx = input.outputs.feature_names_to_idx[cfg_xai["explain"]["output"]]
        grads = {i: [] for i in range(input.num_pred_steps)}

        # Run forward pass for clean baseline prediction
        output = predict_step(
            self.model, input, batch_idx, checkpoint, cfg_model, cfg_dataset, 
            cfg_xai, dataset_info, infer_ds, list_run_hour, use_old_weights, compute_grads=False
        )
        
        if output is None:
            if return_output:
                return None, None
            return

        is_forcing = cfg_xai["explain"]["forcing_input"]
        target_obj = input.forcing if is_forcing else input.inputs
        
        # Backup FULL clean input tensor across ALL variables
        backup_tensor = target_obj.tensor.clone()
        device = backup_tensor.device
        
        # Identify the single variable we want to extract attributions for
        input_var_name = cfg_xai["explain"]["input"]
        target_var_idx = target_obj.feature_names_to_idx[input_var_name]

        # Extract subdomain coordinates from dataset config
        grid_subdomain = cfg_dataset.get("grid", {}).get("subdomain", None)

        # Retrieve Multi-Variable EDA Member States
        active_eda_dir = eda_dir if eda_dir is not None else self.eda_dir
        active_eda_tensor = eda_perturbations if eda_perturbations is not None else self.eda_perturbations

        if active_eda_tensor is not None:
            raw_eda_states = active_eda_tensor.to(device)
            # Failsafe crop if a full grid tensor was passed manually
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

        #Compute Zero-Mean Physical Deviations across Members (epsilon_k)
        # eda_mean = raw_eda_states.mean(dim=0, keepdim=True)
        # zero_mean_perturbations = raw_eda_states - eda_mean
        # num_perturbations = zero_mean_perturbations.shape[0]

        # 1. Zero-mean physical deviations, normalized to model space in one step
        eda_mean = raw_eda_states.mean(dim=0, keepdim=True)
        base = raw_eda_states - eda_mean                       # (m, B, T, Y, X, C)

        C = backup_tensor.shape[-1]
        channel_stds = torch.ones(C, device=device, dtype=torch.float32)
        for feature_name, var_idx in target_obj.feature_names_to_idx.items():
            channel_stds[var_idx] = torch.asarray(
                dataset_info.stats[feature_name]["std"], device=device)
        assert not torch.any(channel_stds == 1.0), "a channel std stayed at default 1.0"
        channel_stds = channel_stds.view(1, 1, 1, 1, 1, -1)

        base = base / channel_stds                             # normalized deviations

        # 2. Gaussian subspace sampling -- PLAIN N(0,1), no row normalization.
        #    This is what gives Cov(eta) = P exactly (see derivation).
        m = base.shape[0]
        weights = torch.randn(self.num_perturbations, m, device=device)
        weights = weights / np.sqrt(m - 1)          # restores Cov(eta) = P
        perturbations = self.scale_factor * torch.einsum('nm,mbtyxc->nbtyxc', weights, base)
        num_perturbations = perturbations.shape[0]
        
    
        for perturbation_idx in range(num_perturbations): 
            
            # Extract normalized deviation and apply smoothing scale factor
            epsilon_k = perturbations[perturbation_idx] #* self.scale_factor
            
            # Inject noise (Both tensors are now safely in N(0,1) standardized space)
            input_perturbed = backup_tensor.clone().detach() + epsilon_k
            
            if is_forcing:
                input.forcing.tensor = input_perturbed
            else:
                input.inputs.tensor = input_perturbed

            # Compute gradients on the fully balanced, normalized state
            gradient, _ = self.base_explainer.compute_explanations(
                input=input, batch_idx=batch_idx, checkpoint=checkpoint,
                cfg_model=cfg_model, cfg_dataset=cfg_dataset, cfg_xai=cfg_xai,
                dataset_info=dataset_info, infer_ds=infer_ds, list_run_hour=list_run_hour,
                use_old_weights=use_old_weights, target=target, plot_explanations=False,
                extent=extent, fig_path=fig_path, return_output=True
            )

            if gradient is None:
                if return_output:
                    return None, None
                return

            for i in range(len(gradient)):
                # Keep exactly the same squeeze logic as standard SmoothGrad to prevent evaluation shape crashes
                grads[i].append(gradient[i].squeeze(-1))

            self.model.zero_grad(set_to_none=True)

        # Restore original clean input state
        if is_forcing:
            input.forcing.tensor = backup_tensor
        else:
            input.inputs.tensor = backup_tensor

        # Aggregation
        grad_list = []
        num_steps = 1 if cfg_xai["explain"]["end_to_end"] else input.num_pred_steps
        for i in range(num_steps):
            # Mirror standard SmoothGrad aggregation EXACTLY
            grad_mean_full = torch.stack(grads[i]).mean(dim=0).unsqueeze(-1)
            grad_list.append(grad_mean_full)

        # Plotting
        if plot_explanations:
            batch_size = input.inputs.tensor.shape[0]
            idx_samples = [batch_idx * batch_size + b for b in range(batch_size)]
            samples = [infer_ds.sample_list[idx] for idx in idx_samples]
            runtimes = [sample.timestamps.datetime.strftime("%Y%m%d%H") for sample in samples]
            stats = dataset_info.stats

            grad_plot = grad_list[0].squeeze(1) if grad_list[0].ndim > 4 else grad_list[0]

            if cfg_xai["explain"]["end_to_end"]:
                step = input.num_pred_steps
                fig_path_step = fig_path + f"/end_to_end/{step}_step/input_{cfg_xai['explain']['input']}_output_{cfg_xai['explain']['output']}"
                self.plot_explanations(
                    grad_plot, input, output, extent, "EDASmoothGrad", fig_path_step,
                    runtime=runtimes[0], stats=stats, step=step-1,
                    input_name=input_var_name, output_name=cfg_xai["explain"]["output"],
                    input_idx=target_var_idx, output_idx=output_idx, target=cfg_xai["target"],
                    forcing=is_forcing, save_figs=cfg_xai["explain"]["exp_plot"]["save_figs"]
                )
            else:
                for i in range(len(grad_list)):
                    fig_path_step = fig_path + f"/{i+1}_step/input_{cfg_xai['explain']['input']}_output_{cfg_xai['explain']['output']}"
                    self.plot_explanations(
                        grad_list[i], input, output, extent, "EDASmoothGrad", fig_path_step,
                        runtime=runtimes[0], stats=stats, step=i,
                        input_name=input_var_name, output_name=cfg_xai["explain"]["output"],
                        input_idx=target_var_idx, output_idx=output_idx, target=cfg_xai["target"],
                        forcing=is_forcing, save_figs=cfg_xai["explain"]["exp_plot"]["save_figs"]
                    )

        if return_output:
            return grad_list, output

        return grad_list