
"""
#  This module is a pure-PyTorch reimplementation of the inference 
#  logic originally defined in `py4cast/lightning.py` (class 
#  AutoRegressiveLightning). 
#  https://github.com/meteofrance/py4cast

"""


import torch
import json
from copy import deepcopy
from mfai.pytorch.models.utils import (
    expand_to_batch,
    features_last_to_second,
    features_second_to_last,
)
from typing import Dict, List, Literal, Tuple, Union
from torch.utils.checkpoint import checkpoint
from py4cast.datasets.base import DatasetInfo, ItemBatch, NamedTensor, Statics


from py4cast.io.outputs import (
    OutputSavingSettings,
    save_gifs,
    save_named_tensors_to_grib,
)
def load_checkpoint(checkpoint):
        """
        We load our feature and dim names from the checkpoint (.ckpt)
        """
        input_feature_names = checkpoint["input_feature_names"]
        output_feature_names = checkpoint["output_feature_names"]
        output_dim_names = checkpoint["output_dim_names"]
        output_dtype = checkpoint["output_dtype"]

        return input_feature_names, output_feature_names, output_dim_names, output_dtype

def get_strategy_params(config):
        """
        Return the parameters for the desired strategy:
        - force_border
        - scale_y
        - num_inter_steps
        """
        training_strategy = config["training_strategy"]
        num_inter_steps = config["num_inter_steps"]
        force_border: bool = True if training_strategy == "scaled_ar" else False
        scale_y: bool = True if training_strategy == "scaled_ar" else False
        # raise if mismatch between strategy and num_inter_steps
        if training_strategy == "diff_ar":
            if num_inter_steps != 1:
                raise ValueError(
                    "Diff AR strategy requires exactly 1 intermediary step."
                )

        return force_border, scale_y, num_inter_steps, training_strategy







def predict_step(model, batch: ItemBatch,batch_idx: int, checkpoint, cfg_model, cfg_dataset,
                 cfg_xai,dataset_info,infer_ds,list_run_hour,use_old_weights,compute_grads=False,enable_checkpoint=True) -> torch.Tensor:
        """
        Check if the feature names are the same as the one used during training
        and make a prediction and accumulate if io_conf =/= none.
        """
        
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        input_feature_names, output_feature_names, output_dim_names, output_dtype = load_checkpoint(checkpoint)

        # if input_feature_names != batch.inputs.feature_names:
        #         raise ValueError(
        #             f"Input Feature names mismatch between training and inference. "
        #             f"Training: {input_feature_names}, Inference: {batch.inputs.feature_names}"
        #         )
        

        
        io_conf = cfg_model["io_conf"]

        if io_conf is None:
            return None,None
        # Save gribs if a io config file is given
        with open(io_conf, "r") as f:
            save_settings = OutputSavingSettings(**json.load(f))

        batch_size = batch.batch_size

        idx_samples = [batch_idx * batch_size + b for b in range(batch_size)]
        samples = [infer_ds.sample_list[idx] for idx in idx_samples]
        runtimes = [
            sample.timestamps.datetime.strftime("%Y%m%d%H") for sample in samples
        ]

        list_run_hour = list_run_hour
        samples_accepted_in_batch = [
            sample.timestamps.datetime.hour in list_run_hour
            for sample in samples
        ]

        if not any(samples_accepted_in_batch):
            
            if compute_grads == False:
                return None
            
            return None,None

        # If the weights are old, it could be not possible to use them as ckpt.
        # Weights should then be loaded with this argument.
        if use_old_weights:
            weights = load_weigths(
                use_old_weights, map_location=device
            )
            model.load_state_dict(weights)

        if compute_grads:
            preds,grad_list,output = common_step(model,batch, batch_idx, cfg_model, cfg_xai, 
                                                 output_feature_names,output_dim_names,output_dtype,
                                                 dataset_info,compute_grads,enable_checkpoint=enable_checkpoint)
        
        else:
            preds,output = common_step(model,batch, batch_idx, cfg_model, cfg_xai, 
                                       output_feature_names,output_dim_names,output_dtype,dataset_info,enable_checkpoint=enable_checkpoint)

        stats = dataset_info.stats
        # Unormalize data
        # if compute_grads:
        #         std = torch.asarray(stats[cfg_xai["explain"]["output"]]["std"])
        #         for i in range(len(grad_list)):
        #             grad_list[i]*= std

        for feature_name in preds.feature_names:
            means = torch.asarray(stats[feature_name]["mean"])
            std = torch.asarray(stats[feature_name]["std"])
            
            preds.tensor[:, :, :, :, preds.feature_names_to_idx[feature_name]] *= std

            preds.tensor[:, :, :, :, preds.feature_names_to_idx[feature_name]] += means

        if cfg_xai["inference"]["infer_plot"]["plot_results"]:
            grid = infer_ds.grid
            for idx, pred in enumerate(preds.iter_dim(dim_name="batch")):
                if not samples_accepted_in_batch[idx]:
                    continue

                runtime = runtimes[idx]
                #sample = samples[idx]


                # Write GIFS
                if cfg_dataset["save_gifs"]:
                    print("Saving gifs...")
                    if cfg_xai["inference"]["infer_plot"]["plot_diff"]:
                        pred.tensor = pred.tensor -batch.inputs.tensor[idx,:,:,:,:]
                    # pred.tensor=pred.tensor.detach()
                    save_gifs(pred, runtime, grid, save_settings)
                
                # if cfg_dataset["save_gribs"]:
                #     print("Writing gribs...")
                #     save_named_tensors_to_grib(
                #         pred.detatch().numpy(), infer_ds, sample, save_settings, runtime
                #     )

        # print(preds.tensor.min(), preds.tensor.max())
        # # print((preds.tensor > 0).float().mean())

        if compute_grads:
            return preds,grad_list

        return preds








def load_weigths(path, map_location):
        """
        delete "model." in keys in the dict.
        """
        from collections import OrderedDict

        weights = torch.load(path, map_location)
        new_state_dict = OrderedDict()
        for k, v in weights.items():
            new_key = k.replace("model.", "")
            new_state_dict[new_key] = v
        return new_state_dict

def step_diffs(
        diff_stats, feature_names: List[str], device: torch.device
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Get the mean and std of the differences between two consecutive states on the desired device.
        """
        step_diff_std = diff_stats.to_list("std", feature_names).to(
            device,
            non_blocking=True,
        )
        step_diff_mean = diff_stats.to_list("mean", feature_names).to(
            device, non_blocking=True
        )
        return step_diff_std, step_diff_mean


def next_x(
         batch: ItemBatch, prev_states: NamedTensor, step_idx: int, training_strategy, mask_on_nan, grid_static_features
    ) -> torch.Tensor:
        """
        Build the next x input for the model at timestep step_idx using the :
        - previous states
        - forcing
        - static features

        If downscaling strategy, the previous_states are set to 0.
        """
        forcing = batch.forcing.select_dim("timestep", step_idx)
        ds = training_strategy == "downscaling_only"
        inputs = [
            prev_states.select_tensor_dim("timestep", idx)
            for idx in range(batch.num_input_steps)
        ]

        mask_list = []

        # create a mask that corresponds to the union of the nans in the input and the forcings
        if mask_on_nan:
            combined_mask = torch.zeros_like(inputs[0][:, :, :, 0], dtype=torch.bool)

            # Combine masks for inputs
            for input in inputs:
                mask = torch.isnan(input)
                for i in range(mask.shape[-1]):
                    combined_mask = (
                        combined_mask | mask[:, :, :, i]
                    )  # Union of size masks (batch, lat, lon)

            # Combine masks for forcing
            mask = torch.isnan(forcing.tensor)
            for i in range(mask.shape[-1]):
                combined_mask = (
                    combined_mask | mask[:, :, :, i]
                )  # Union of size masks (batch, lat, lon)

            mask_list.append(
                ~combined_mask.unsqueeze(-1)  # unsqueeze and invert combined_mask
            )  # shape [(batch, lat, lon, param)]

            # replace nan by 0 in inputs
            inputs = [torch.nan_to_num(input, nan=0) for input in inputs]
            # replace nan by 0 in forcing
            forcing.tensor = torch.nan_to_num(forcing.tensor, nan=0)

        # If downscapred_outling only, inputs are not concatenated: only use static features and forcings.
       
        x = torch.cat(
            inputs * (1 - ds)  # = [] if downscaling strategy
            + [grid_static_features[: batch.batch_size], forcing.tensor]
            + mask_list,
            dim=forcing.dim_index("features"),
        )

        return x,forcing.tensor






def common_step(
        model, batch: ItemBatch,batch_idx: int, cfg_model, cfg_xai, output_feature_names,output_dim_names,
        output_dtype,dataset_info,compute_grad=False,enable_checkpoint=True
    ) -> Tuple[NamedTensor, NamedTensor]:
        """
        Two Autoregressive strategies are implemented here for train, val, test and inference:
        - scaled_ar:
            * Boundary forcing with y_true/true_state
            * Scaled Differential update next_state = prev_state + y * std + mean
            * Intermediary steps for which we have no y_true data

        - diff_ar:
            * No Boundary forcing
            * Differential update next_state = prev_state + y
            * No Intermediary steps

        Another training stratgey is implemented (still experimental) is the downscaling, with
            * No Boundary forcing
            * Update next_state = y
            * No Intermediary steps


        In inference mode, we assume batch.outputs is None and we disable output based border forcing.
        """
        force_border, scale_y, num_inter_steps, training_strategy = get_strategy_params(cfg_model)
       

        ### uncomment the if for graph based models

        # if  model.model_type == ModelType.GRAPH:
        #     # Stack original shape to reshape later
        #     original_shape = batch.inputs.tensor.shape
        #     # Graph model, we flatten the batch spatial dims
        #     batch.inputs.flatten_("ngrid", *batch.inputs.spatial_dim_idx)

        #     batch.forcing.flatten_("ngrid", *batch.forcing.spatial_dim_idx)

        # we save the feature names at the first batch of training
        # to check at inference time if the feature names are the same
        # also useful to build NamedTensor outputs with same feature and dim names
        # If model type is graph, flat the lon/lat dim before saving the dims
        

        prev_states = batch.inputs
        prediction_list = []
        gradient_list = []
        diff_stats = dataset_info.diff_stats
        statics = deepcopy(dataset_info.statics)
        mask_on_nan =False
        channels_last = cfg_model["channels_last"]
        mask_ratio = cfg_model["mask_ratio"]  
    
        # Set model input/output grid features based on dataset tensor shapes
        grid_static_features = statics.grid_statics
        grid_static_features = expand_to_batch(
            statics.grid_statics.tensor, batch.batch_size)
        grid_static_features = grid_static_features.to("cuda")
        
        # Here we do the autoregressive prediction looping
        if compute_grad:
            batch.inputs.tensor.requires_grad_(True)
        
        
        
        # for the desired number of ar steps.
        
        for i in range(batch.num_pred_steps):

            if scale_y:
                # statistics of temporal_difference of output feature
                
                step_diff_std, step_diff_mean = step_diffs(
                    diff_stats,
                    output_feature_names,
                    prev_states.device
                )



            # Intermediary steps for which we have no y_true data

            for k in range(num_inter_steps):

                # prepare the input for the model
                x,forcing_tensor = next_x(batch, prev_states, i,training_strategy, mask_on_nan, grid_static_features)

                # Graph (B, N_grid, d_f) or Conv (B, N_lat,N_lon d_f)
                if channels_last:
                    x = x.to(memory_format=torch.channels_last)
                if mask_ratio != 0:  # maskedautoencoder strategy
                    x = mask_tensor(x, mask_ratio)
                # Here we adapt our tensors to the order of dimensions of CNNs and ViTs
                
                # if model.features_second:
                #     x = features_last_to_second(x)
                  
                #     y = model(x)
                #     y = features_second_to_last(y)
                # else:
                
                #     y = model(x)

                #TODO: use the forward function 

                def forward_fn(x_in):
                    if model.features_second:
                        x_in = features_last_to_second(x_in)
                        out = model(x_in)
                        return features_second_to_last(out)
                    return model(x_in)
                if enable_checkpoint:
                    y = checkpoint(forward_fn, x, use_reentrant=False)
                else:
                    y = forward_fn(x)
                ds = training_strategy == "downscaling_only"

                # select the last timestep
                last_prev_state = prev_states.select_tensor_dim("timestep", -1)
                if mask_on_nan:
                    last_prev_state = torch.nan_to_num(last_prev_state, nan=0)

                # We update the latest of our prev_states with the network output
                if scale_y:
                    predicted_state = (
                        # select the last timestep
                        last_prev_state * (1 - ds)
                        + y * step_diff_std
                        + step_diff_mean
                    )
                else:
                    predicted_state = last_prev_state * (1 - ds) + y

                # Overwrite border with true state
                # Force it to true state for all intermediary step
                
                new_state = predicted_state

                # Only update the prev_states if we are not at the last step
                if i < batch.num_pred_steps - 1 or k < num_inter_steps - 1:
                    # Update input states for next iteration: drop oldest, append new_state
                    timestep_dim_index = batch.inputs.dim_index("timestep")
                    new_prev_states_tensor = torch.cat(
                        [
                            # Drop the oldest timestep (select all but the first)
                            prev_states.index_select_tensor_dim(
                                "timestep",
                                range(1, prev_states.dim_size("timestep")),
                            ),
                            # Add the timestep dimension to the new state
                            new_state.unsqueeze(timestep_dim_index),
                        ],
                        dim=timestep_dim_index,
                    )

                    # Make a new NamedTensor with the same dim and
                    # feature names as the original prev_states
                    prev_states = NamedTensor.new_like(
                        new_prev_states_tensor, prev_states
                    )
            
            if compute_grad and not cfg_xai["explain"]["end_to_end"]:
                #Compute the grad wrt to all inputs
                


                #TODO: compute gradient wrt to a specific chosen input

                input_name = cfg_xai["explain"]["input"]
                output_name = cfg_xai["explain"]["output"]
                
                if cfg_xai["explain"]["forcing_input"]:
                    output_idx = int(batch.outputs.feature_names_to_idx[output_name])
                    input_tensor = forcing_tensor
                else:
                    
                    output_idx = int(batch.outputs.feature_names_to_idx[output_name])
                    if i>0:
                        
                        input_tensor = prediction_list[-1]#[...,input_idx:input_idx+1]
                        
                    else: 
                        input_tensor = batch.inputs.tensor#[...,input_idx:input_idx+1]
                    
              
                target = cfg_xai["target"]
                if target:    
                    extent = cfg_xai["extent"]
                    
                    lon0,lon1 = target[0], target[1]
                    lat0,lat1 = target[2], target[3]
                    H, W = batch.inputs.tensor.shape[2:4]
                    lat_min, lat_max = extent[2], extent[3]
                    lon_min, lon_max = extent[0], extent[1]

                    # Compute grid spacing
                    lat_step = (lat_max - lat_min) / H
                    lon_step = (lon_max - lon_min) / W
                
                    i_min = int((lat0-lat_min)/lat_step)
                    i_max =  int((lat1-lat_min)/lat_step)
                    j_min = int ((lon0-lon_min)/lon_step)
                    j_max = int((lon1-lon_min)/lon_step)
                    
                    output_tensor = new_state[...,i_min:i_max,j_min:j_max,output_idx].unsqueeze(1).unsqueeze(-1)#.clone().detach().requires_grad_(True)
                else:
                    output_tensor = new_state[...,output_idx].unsqueeze(1).unsqueeze(-1)#.clone().detach().requires_grad_(True)

                new_grad = torch.autograd.grad(
                                outputs=output_tensor,
                                inputs=input_tensor,
                                grad_outputs=torch.ones_like(output_tensor), retain_graph=True
                            
                            )[0]
                gradient_list.append(new_grad)

        
            prediction_list.append(new_state)

        if compute_grad and cfg_xai["explain"]["end_to_end"]:
            input_name = cfg_xai["explain"]["input"]
            output_name = cfg_xai["explain"]["output"]
            
            if cfg_xai["explain"]["forcing_input"]:
                output_idx = int(batch.outputs.feature_names_to_idx[output_name])
                input_tensor = forcing_tensor
            else:
                
                output_idx = int(batch.outputs.feature_names_to_idx[output_name])
                if i>0:
                    
                    input_tensor = prediction_list[-1]#[...,input_idx:input_idx+1]
                    
                else: 
                    input_tensor = batch.inputs.tensor#[...,input_idx:input_idx+1]
                
            
            target = cfg_xai["target"]
            if target:    
                extent = cfg_xai["extent"]
                
                lon0,lon1 = target[0], target[1]
                lat0,lat1 = target[2], target[3]
                H, W = batch.inputs.tensor.shape[2:4]
                lat_min, lat_max = extent[2], extent[3]
                lon_min, lon_max = extent[0], extent[1]

                # Compute grid spacing
                lat_step = (lat_max - lat_min) / H
                lon_step = (lon_max - lon_min) / W
            
            
                i_min = int((lat0-lat_min)/lat_step)
                i_max =  int((lat1-lat_min)/lat_step)
                j_min = int ((lon0-lon_min)/lon_step)
                j_max = int((lon1-lon_min)/lon_step)

                output_tensor = prediction_list[-1][...,i_min:i_max,j_min:j_max,output_idx].unsqueeze(1).unsqueeze(-1)#.clone().detach().requires_grad_(True)
            else:
                output_tensor = prediction_list[-1][...,output_idx].unsqueeze(1).unsqueeze(-1)#.clone().detach().requires_grad_(True)

            g_end_to_end = torch.autograd.grad(
                    outputs=output_tensor,
                    inputs=batch.inputs.tensor,
                    grad_outputs=torch.ones_like(output_tensor),
                    retain_graph=False,  # free graph after, you don't need it anymore
                )[0]
            
            gradient_list.append(g_end_to_end)

       

        prediction = torch.stack(
            prediction_list, dim=1
        )  # Stacking is done on time step. (B, pred_steps, N_grid, d_f) or (B, pred_steps, N_lat, N_lon, d_f)

        # In inference mode we use a "trained" module which MUST have the output feature names
        # and the output dim names attributes set.
        pred_out = NamedTensor(
                prediction.type(output_dtype),
                output_dim_names,
                output_feature_names,
            )
       
        if compute_grad:
            return pred_out,gradient_list, batch.outputs
        

        return pred_out,batch.outputs



def mask_tensor(x, mask_ratio):
        _, height, width, _ = x.shape
        num_blocks = int((1 - mask_ratio) * height * width)
        block_size_h = height // int(height**0.5)
        block_size_w = width // int(width**0.5)
        mask = torch.ones_like(x, dtype=torch.bool)
        block_indices = torch.randperm(height * width)[:num_blocks]
        for i in block_indices:
            row = i // width
            col = i % width
            mask[
                :,
                row * block_size_h : (row + 1) * block_size_h,
                col * block_size_w : (col + 1) * block_size_w,
                :,
            ] = False
        return x * mask