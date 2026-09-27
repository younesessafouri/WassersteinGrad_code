from py4castxai.explainers.XGrad import XGrad
import torch
from py4castxai.utils_test.infer_utils import predict_step

from py4castxai.explainers.inputxgrad import InputxGrad
from py4castxai.explainers.basegrad import BaseGrad
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

class SmoothGradSquared(XGrad):


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
                gradient,_ = self.base_explainer.compute_explanations(input=input,batch_idx=batch_idx,checkpoint=checkpoint,cfg_model=cfg_model,cfg_dataset=cfg_dataset,cfg_xai=cfg_xai,dataset_info=dataset_info,infer_ds=infer_ds,list_run_hour=list_run_hour,use_old_weights=use_old_weights,target=target,plot_explanations=False,extent=extent,fig_path=fig_path,return_output=True)

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
      
            for _ in range(self.num_perturbations): 
                
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
                    grads[i].append(gradient[i].squeeze(-1))
        
        grad_list = []
        if not cfg_xai["explain"]["end_to_end"]:

            for i in range(input.num_pred_steps):
                
                grad_list.append((torch.stack(grads[i])**2).mean(dim=0).unsqueeze(-1))


        else:
            grad_list.append((torch.stack(grads[0])**2).mean(dim=0).unsqueeze(-1))
  
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
            if cfg_xai["explain"]["forcing_input"]:
                        input_idx = input.forcing.feature_names_to_idx[cfg_xai["explain"]["input"]]
            else:
                        input_idx = input.inputs.feature_names_to_idx[cfg_xai["explain"]["input"]]
                 
            output_idx =output.feature_names_to_idx[cfg_xai["explain"]["output"]] 

            stats = dataset_info.stats
            grad_list[0] = grad_list[0].squeeze(1)
            if cfg_xai["explain"]["end_to_end"]:
                    step = input.num_pred_steps
                    fig_path_step =fig_path+ f"/end_to_end/{step}_step/input_{cfg_xai['explain']['input']}_output_{cfg_xai['explain']['output']}"
                                        
                    self.plot_explanations(grad_list[0],input,output,
                                            extent,"SmoothGradSquared",fig_path_step,
                                            runtime=runtimes[0],stats=stats,step=step-1,
                                            input_name = cfg_xai["explain"]["input"],
                                            output_name = cfg_xai["explain"]["output"],
                                        input_idx=input_idx,output_idx = output_idx,target=cfg_xai["target"],forcing=cfg_xai["explain"]["forcing_input"],save_figs=cfg_xai["explain"]["exp_plot"]["save_figs"])
                
            else: 
                for i in range(len(grad_list)):
                    fig_path_step =fig_path+ f"/{i+1}_step/input_{cfg_xai['explain']['input']}_output_{cfg_xai['explain']['output']}"
                    self.plot_explanations(grad_list[i],input,output,
                                            extent,"SmoothGradSquared",fig_path_step,
                                            runtime=runtimes[0],stats=stats,step=i,
                                            input_name = cfg_xai["explain"]["input"],
                                            output_name = cfg_xai["explain"]["output"],
                                            input_idx=input_idx,output_idx = output_idx,target=cfg_xai["target"],forcing=cfg_xai["explain"]["forcing_input"],save_figs=cfg_xai["explain"]["exp_plot"]["save_figs"])


        if return_output:

            return  grad_list, output         
        
        return grad_list
    
    def compute_different_explanations(self, input, batch_idx, checkpoint, cfg_model, 
                                      cfg_dataset, cfg_xai, dataset_info, infer_ds, 
                                      list_run_hour, use_old_weights, target, 
                                      plot_explanations=True, extent=[-12, 16, 37.5, 55.4], 
                                      return_output=False, fig_path=".figs/attributions"):
        """
        Compute SmoothGrad attributions for multiple input variables.
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

            # Storage for gradients across perturbations for this input
            grads = {step: [] for step in range(input.num_pred_steps)}
            
            input_idx = input.inputs.feature_names_to_idx[explain_input]
            input_clean = backup_tensor[..., input_idx]
            # Calculate perturbation scale
            vals = backup_tensor[..., input_idx]
            std_perturbation = self.std_perturbation * (vals.max() - vals.min())
            # SmoothGrad: average gradients over multiple perturbations
            for _ in range(self.num_perturbations):
                # Create perturbed input
                input.inputs.tensor = backup_tensor.clone()  # Reset to clean state
                input_to_perturb = input_clean.clone().detach()
                # input.inputs.tensor =input_clean+ torch.randn_like(input.inputs.tensor)*std_perturbation
                input_perturbed = apply_geometric_noise(input_to_perturb, max_translation=0.05, max_rotation=5.0)
                # input_perturbed = input_clean.clone().detach() + torch.randn_like(input_clean) * std_perturbation
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
                
                # Accumulate gradients for each prediction step
                for step_idx in range(len(gradient)):
                    grads[step_idx].append(gradient[step_idx].squeeze(-1))
                
                # Clear gradients to free memory
                # self.model.zero_grad(set_to_none=True)
            
            # Average gradients across perturbations for each prediction step
            grad_list_for_input = []
            for step_idx in range(input.num_pred_steps):
                averaged_grad = (torch.stack(grads[step_idx])**2).mean(dim=0).unsqueeze(-1)
     
                grad_list_for_input.append(averaged_grad)
            
            # Store all steps for this input variable
            
            grad_multiple[explain_input] = grad_list_for_input
            torch.cuda.empty_cache()

        # Restore original input
        input.inputs.tensor = backup_tensor
        
        if return_output:
            return grad_multiple, output
        return grad_multiple
    
    def compute_different_explanations_forcing(self, input, batch_idx, checkpoint, cfg_model, 
                                      cfg_dataset, cfg_xai, dataset_info, infer_ds, 
                                      list_run_hour, use_old_weights, target, 
                                      plot_explanations=True, extent=[-12, 16, 37.5, 55.4], 
                                      return_output=False, fig_path=".figs/attributions"):
        """
        Compute SmoothGrad attributions for multiple input variables.
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
        backup_tensor = input.forcing.tensor.clone()
    
        # Process each input variable
        for explain_input in cfg_xai["multiple_explain"]:
            cfg_xai["explain"]["input"] = explain_input

            # Storage for gradients across perturbations for this input
            grads = {step: [] for step in range(input.num_pred_steps)}
            
            input_idx = input.forcing.feature_names_to_idx[explain_input]
            input_clean = backup_tensor[..., input_idx]
            # Calculate perturbation scale
            vals = backup_tensor[..., input_idx]
            std_perturbation = self.std_perturbation * (vals.max() - vals.min())
            # SmoothGrad: average gradients over multiple perturbations
            for _ in range(self.num_perturbations):
                # Create perturbed input
                input.forcing.tensor = backup_tensor.clone()  # Reset to clean state
                input_perturbed = input_clean.clone().detach() + torch.randn_like(input_clean) * std_perturbation
                input.forcing.tensor[..., input_idx] = input_perturbed

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
                    input.forcing.tensor = backup_tensor
                    if return_output:
                        return None, None
                    return None
                
                # Accumulate gradients for each prediction step
                for step_idx in range(len(gradient)):
                    grads[step_idx].append(gradient[step_idx].squeeze(-1))
                
                # Clear gradients to free memory
                # self.model.zero_grad(set_to_none=True)
            
            # Average gradients across perturbations for each prediction step
            grad_list_for_input = []
            for step_idx in range(input.num_pred_steps):
                averaged_grad = (torch.stack(grads[step_idx])**2).mean(dim=0).unsqueeze(-1)
     
                grad_list_for_input.append(averaged_grad)
            
            # Store all steps for this input variable
            
            grad_multiple[explain_input] = grad_list_for_input
            torch.cuda.empty_cache()

        # Restore original input
        input.forcing.tensor = backup_tensor
        
        if return_output:
            return grad_multiple, output
        return grad_multiple