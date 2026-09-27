from py4castxai.explainers.XGrad import XGrad
import torch
from py4castxai.utils_test.infer_utils import predict_step

from py4castxai.explainers.inputxgrad import InputxGrad
from py4castxai.explainers.basegrad import BaseGrad

import torch
class VarGrad(XGrad):

    def __init__(self, model, base_explainer="BaseGrad", std_perturbations=0.1, num_perturbations=50):
        super().__init__(model)
        if base_explainer == "InputxGrad":
            self.base_explainer = InputxGrad(model)
        else:
            self.base_explainer = BaseGrad(model)
        
        self.std_perturbation = std_perturbations
        self.num_perturbations = num_perturbations

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
            backup_tensor = input.forcing.tensor.clone()
            input_idx = input.forcing.feature_names_to_idx[cfg_xai["explain"]["input"]]
            input_clean = input.forcing.tensor[..., input_idx]
            vals = input.forcing.tensor[..., input_idx]
            std_perturbation = self.std_perturbation * (vals.max() - vals.min())   
            
            for perturbation_idx in range(self.num_perturbations): 
                input_perturbed = input_clean.clone().detach() + torch.randn_like(input_clean) * std_perturbation
                
                input.forcing.tensor = input.forcing.tensor.clone()
                input.forcing.tensor[..., input_idx] = input_perturbed
                gradient, _ = self.base_explainer.compute_explanations(input=input, batch_idx=batch_idx, checkpoint=checkpoint, cfg_model=cfg_model, cfg_dataset=cfg_dataset, cfg_xai=cfg_xai, dataset_info=dataset_info, infer_ds=infer_ds, list_run_hour=list_run_hour, use_old_weights=use_old_weights, target=target, plot_explanations=False, extent=extent, fig_path=fig_path, return_output=True)

                if gradient is None:
                    if return_output:
                        return None, None
                    return
                
                for i in range(len(gradient)):
                    grads[i].append(gradient[i].squeeze(-1) if gradient[i].dim() > len(input_clean.shape) else gradient[i])
                
                self.model.zero_grad(set_to_none=True)

        else:
            backup_tensor = input.inputs.tensor.clone()

            input_idx = input.inputs.feature_names_to_idx[cfg_xai["explain"]["input"]]
            input_clean = input.inputs.tensor[..., input_idx]
            vals = input.inputs.tensor[..., input_idx] 
            stats = dataset_info.stats
            
            std_perturbation = self.std_perturbation * (vals.max() - vals.min())
            perturbations_to_plot = [0, self.num_perturbations // 2, self.num_perturbations - 1]
            perturbed_samples = []

            for perturbation_idx in range(self.num_perturbations): 
                input_perturbed = input_clean.clone().detach() + torch.randn_like(input_clean) * std_perturbation
                
                if perturbation_idx in perturbations_to_plot:
                    perturbed_samples.append((perturbation_idx, input_perturbed.clone()))
                
                input.inputs.tensor = input.inputs.tensor.clone()
                input.inputs.tensor[..., input_idx] = input_perturbed
                
                gradient, _ = self.base_explainer.compute_explanations(input=input, batch_idx=batch_idx, checkpoint=checkpoint, cfg_model=cfg_model,
                                                                       
                                                                        cfg_dataset=cfg_dataset, cfg_xai=cfg_xai, dataset_info=dataset_info, infer_ds=infer_ds,
                                                                          list_run_hour=list_run_hour, use_old_weights=use_old_weights, target=target, plot_explanations=False, extent=extent, fig_path=fig_path, 
                                                                          return_output=True)

                if gradient is None:
                    if return_output:
                         return None, None
                    return
                
                for i in range(len(gradient)):
                    grads[i].append(gradient[i].squeeze(-1))
                
                self.model.zero_grad(set_to_none=True)



        # -------------------------------------------------------------------
        # VARGRAD AGGREGATION:
        # -------------------------------------------------------------------
        grad_list = []
        
        if not cfg_xai["explain"]["end_to_end"]:
            for i in range(input.num_pred_steps):
                
                var_grad = torch.stack(grads[i]).var(dim=0)
                grad_list.append(var_grad.unsqueeze(-1))
        
        else: 
            grad_list.append(torch.stack(grads[0]).var(dim=0).unsqueeze(-1))




        # Restore backup
        if cfg_xai["explain"]["forcing_input"]:
            input.forcing.tensor = backup_tensor
        else:
            input.inputs.tensor = backup_tensor
        
        if plot_explanations:
            batch_size = input.inputs.tensor.shape[0]
            idx_samples = [batch_idx * batch_size + b for b in range(batch_size)]
            samples = [infer_ds.sample_list[idx] for idx in idx_samples]
            runtimes = [sample.timestamps.datetime.strftime("%Y%m%d%H") for sample in samples]
            
            stats = dataset_info.stats

            # Remove extra channel dim for plotting if it exists
            if grad_list[0].dim() > 3:
                 grad_list[0] = grad_list[0].squeeze(1)

            for i in range(len(grad_list)):
                output_idx = output.feature_names_to_idx[cfg_xai["explain"]["output"]] 
                if cfg_xai["explain"]["forcing_input"]:
                    input_idx = input.forcing.feature_names_to_idx[cfg_xai["explain"]["input"]]
                else:
                    input_idx = input.inputs.feature_names_to_idx[cfg_xai["explain"]["input"]]
                
                fig_path_step = fig_path + f"/{i+1}_step/input_{cfg_xai['explain']['input']}_output_{cfg_xai['explain']['output']}"
                self.plot_explanations(grad_list[i], input, output, extent, "VarGrad", fig_path_step,
                                       runtime=runtimes[0], stats=stats, step=i,
                                       input_name=cfg_xai["explain"]["input"],
                                       output_name=cfg_xai["explain"]["output"],
                                       input_idx=input_idx, output_idx=output_idx, 
                                       target=cfg_xai["target"], forcing=cfg_xai["explain"]["forcing_input"], 
                                       save_figs=cfg_xai["explain"]["exp_plot"]["save_figs"])

        if return_output:
            return grad_list, output         
        
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
                
                # Accumulate gradients for each prediction step
                for step_idx in range(len(gradient)):
                    grads[step_idx].append(gradient[step_idx].squeeze(-1))
                
                # Clear gradients to free memory
                # self.model.zero_grad(set_to_none=True)
            
            # Average gradients across perturbations for each prediction step
            grad_list_for_input = []
     
            grad_list_for_input.append(torch.stack(grads[0]).var(dim=0).unsqueeze(-1))
            
            # Store all steps for this input variable
            
            grad_multiple[explain_input] = grad_list_for_input
            torch.cuda.empty_cache()

        # Restore original input
        input.inputs.tensor = backup_tensor
        
        if return_output:
            return grad_multiple, output
        return grad_multiple
    