from py4castxai.explainers.XGrad import XGrad
from py4castxai.utils_test.infer_utils import predict_step
import numpy as np

def constant_roi_blob(H, W, roi_center_ij, sigma_px, extent=None):
    """
    A fixed Gaussian blob centered on the ROI, independent of any input.
    roi_center_ij = (i, j) pixel coords of the ROI center (e.g. Paris).
    """
    yy, xx = np.mgrid[0:H, 0:W]
    ci, cj = roi_center_ij
    blob = np.exp(-((yy - ci)**2 + (xx - cj)**2) / (2 * sigma_px**2))
    blob = np.abs(blob)
    return blob / (blob.sum() + 1e-12)   # normalize to a probability measure, like WGBary

class BaseGrad(XGrad):


    def __init__(self,model):
        super().__init__(model)


    def compute_explanations(self,input,batch_idx,checkpoint,cfg_model,cfg_dataset,cfg_xai,dataset_info,infer_ds,list_run_hour,
                             use_old_weights,target,plot_explanations=False,extent=[-12, 16, 37.5, 55.4],return_output=False,fig_path=".figs/attributions"):
        


        output,grad_list = predict_step(self.model,input,batch_idx,checkpoint,cfg_model,cfg_dataset,cfg_xai,dataset_info,infer_ds,list_run_hour,use_old_weights,compute_grads=True)

        if output is None:
                if return_output:
                     return None,None
                
                return
        
        if cfg_xai["explain"]["forcing_input"]:
            input_idx = input.forcing.feature_names_to_idx[cfg_xai["explain"]["input"]]
        else:
            input_idx = input.inputs.feature_names_to_idx[cfg_xai["explain"]["input"]] 

            
        for i in range(len(grad_list)):
                    grad_list[i] = grad_list[i][...,input_idx].unsqueeze(-1)
        # gradients = self.compute_grad(input.inputs,output,target)

        if plot_explanations:
            batch_size = input.inputs.tensor.shape[0]

            idx_samples = [batch_idx * batch_size + b for b in range(batch_size)]
            samples = [infer_ds.sample_list[idx] for idx in idx_samples]
            runtimes = [
                sample.timestamps.datetime.strftime("%Y%m%d%H") for sample in samples
            ]
            output_idx =output.feature_names_to_idx[cfg_xai["explain"]["output"]] 

            
            stats = dataset_info.stats
            
            grad_list[0] = grad_list[0].squeeze(1)
            if cfg_xai["explain"]["end_to_end"]:
                step = input.num_pred_steps
                fig_path_step =fig_path+ f"/end_to_end/{step}_step/input_{cfg_xai['explain']['input']}_output_{cfg_xai['explain']['output']}"

                self.plot_explanations(grad_list[i],input,output,
                                        extent,"BaseGrad",fig_path_step,
                                        runtime=runtimes[0],stats=stats,step=step-1,
                                        input_name = cfg_xai["explain"]["input"],
                                        output_name = cfg_xai["explain"]["output"],
                                       input_idx=input_idx,output_idx = output_idx,target=cfg_xai["target"],forcing=cfg_xai["explain"]["forcing_input"],
                                       save_figs=cfg_xai["explain"]["exp_plot"]["save_figs"])
            
            else: 
                for i in range(len(grad_list)):
                    
                            
                        
                    fig_path_step =fig_path+ f"/{i+1}_step/input_{cfg_xai['explain']['input']}_output_{cfg_xai['explain']['output']}"
                        
                    self.plot_explanations(grad_list[i],input,output,
                                            extent,"BaseGrad",fig_path_step,
                                            runtime=runtimes[0],stats=stats,step=i,
                                            input_name = cfg_xai["explain"]["input"],
                                           output_name = cfg_xai["explain"]["output"],
                                            input_idx=input_idx,output_idx = output_idx,target=cfg_xai["target"],forcing=cfg_xai["explain"]["forcing_input"],save_figs=cfg_xai["explain"]["exp_plot"]["save_figs"])
 

        if return_output:
           
            return grad_list,output
        

        return grad_list
        

    def compute_different_explanations(self,input,batch_idx,checkpoint,cfg_model,cfg_dataset,cfg_xai,dataset_info,infer_ds,list_run_hour,use_old_weights,target,plot_explanations=True,extent=[-12, 16, 37.5, 55.4],return_output=False,fig_path=".figs/attributions"):
        # compute x times gradient wrt x 
        
        output,grad_list = predict_step(self.model,input,batch_idx,checkpoint,cfg_model,cfg_dataset,cfg_xai,dataset_info,infer_ds,list_run_hour,use_old_weights,compute_grads=True)

        if output is None:
                if return_output:
                     return None,None
                
                return
        grad_multiple = {explain_input:[] for explain_input in cfg_xai["multiple_explain"]}
        for explain_input in cfg_xai["multiple_explain"]:
        # if cfg_xai["explain"]["forcing_input"]:
        #     input_idx = input.forcing.feature_names_to_idx[cfg_xai["explain"]["input"]]
        # else:
            cfg_xai["explain"]["input"] = explain_input

            input_idx = input.inputs.feature_names_to_idx[explain_input] 

            
            for i in range(len(grad_list)):
                        grad_multiple[explain_input].append(grad_list[i][...,input_idx].unsqueeze(-1))
        # gradients = self.compute_grad(input.inputs,output,target)

           

        if return_output:
           
            return grad_multiple,output
        

        return grad_multiple
        

   
