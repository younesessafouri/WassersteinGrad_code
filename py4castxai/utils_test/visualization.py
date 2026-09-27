"""
Visualization Module - Autoregressive & GIF Generation
=======================================================
"""

import os
import logging
import torch
import numpy as np
import matplotlib
matplotlib.use('Agg')  # must be before importing pyplot
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from matplotlib.patches import Rectangle
import copy
import imageio
from natsort import natsorted
from typing import List, Dict, Optional
from ..explainers import explainers_registry

class AutoregressiveVisualizer:
    """Visualizes autoregressive attribution evolution."""
    
    def __init__(self, model, config, dataloader, dataset_info, infer_ds):
        self.model = model
        self.config = config
        self.dataloader = dataloader
        self.dataset_info = dataset_info
        self.infer_ds = infer_ds
        self.logger = logging.getLogger(__name__)
        
    def run(self):
        """Run autoregressive visualization."""
        from py4castxai.explainers import SmoothGrad, InputxGrad, BaseGrad
        
        cfg_xai = self.config.cfg_xai
        saved_frames = []
        
        for explainer_name, expl_params in cfg_xai["explain"]["explainers"].items():
            self.logger.info(f"Generating autoregressive viz for {explainer_name}")
            
            # Get explainer
            explainer_map = {
                "SmoothGrad": SmoothGrad,
                "InputxGrad": InputxGrad,
                "BaseGrad": BaseGrad
            }
            explainer_cls = explainer_map.get(explainer_name, BaseGrad)
            explainer = explainer_cls(self.model, **expl_params)
            
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            
            for batch_idx, batch in enumerate(self.dataloader):
                batch.inputs.tensor = batch.inputs.tensor.to(device)
                if batch.outputs is not None:
                    batch.outputs.tensor = batch.outputs.tensor.to(device)
                batch.forcing.tensor = batch.forcing.tensor.to(device)
                
                with torch.autograd.set_grad_enabled(True):
                    batch.inputs.tensor.requires_grad_()
                    batch.forcing.tensor.requires_grad_()
                    
                    exp, output = explainer.compute_different_explanations(
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
                        plot_explanations=True,
                        extent=cfg_xai["extent"],
                        fig_path=cfg_xai["explain"]["exp_plot"]["fig_path"],
                        return_output=True
                    )
                    
                    if exp is None:
                        continue
                    
                    if output is not None:
                        all_explanations = exp
                        
                        # Get runtime info
                        batch_size = batch.inputs.tensor.shape[0]
                        idx_samples = [batch_idx * batch_size + b for b in range(batch_size)]
                        samples = [self.infer_ds.sample_list[idx] for idx in idx_samples]
                        runtimes = [
                            sample.timestamps.datetime.strftime("%Y-%m-%d %H") 
                            for sample in samples
                        ]
                        
                        # Plot each timestep
                        for i in range(len(all_explanations[cfg_xai["multiple_explain"][0]])):
                            fig_path = os.path.join(
                                cfg_xai["explain"]["exp_plot"]["fig_path"],
                                f"{i+1}_step"
                            )
                            
                            self._plot_timestep(
                                all_explanations, batch, output,
                                extent=cfg_xai["extent"],
                                method=explainer_name,
                                fig_path=fig_path,
                                runtime=runtimes[0],
                                stats=self.dataset_info.stats,
                                step=i,
                                output_name="aro_tp_0m",
                                target=cfg_xai["target"],
                                forcing=None,
                                save_figs=True,
                                input_names=cfg_xai["multiple_explain"]
                            )
                            
                            frame_path = os.path.join(
                                fig_path, explainer_name, f"{runtimes[0]}.png"
                            )
                            saved_frames.append(frame_path)
                
                torch.cuda.empty_cache()
            
            # Create GIF
            self.logger.info(f"Creating GIF for {explainer_name}")
            self._create_gif(explainer_name, saved_frames, fps=2)
    
    def _plot_timestep(self, explanations, batch, output, extent, method,
                      fig_path, runtime, stats, step, output_name,
                      target, forcing, save_figs, input_names):
        """Plot attribution maps for a single timestep."""
        
        def denormalize(t, mean, std):
            return t * std + mean
        
        def normalize_attr(a):
            if isinstance(a, np.ndarray):
                a = torch.from_numpy(a).float()
            return a / (a.abs().max() + 1e-8)
        
        n_inputs = len(input_names)
        
        fig = plt.figure(figsize=(4.5 * n_inputs, 8))
        gs = fig.add_gridspec(
            2, n_inputs,
            height_ratios=[1, 1],
            hspace=0.25,
            wspace=0.15,
            left=0.05, right=0.98, top=0.94, bottom=0.04
        )
        
        projection = ccrs.PlateCarree()
        
        # Target box
        lon0, lon1, lat0, lat1 = target
        rect = Rectangle(
            (lon0, lat0), lon1 - lon0, lat1 - lat0,
            linewidth=0.7, edgecolor='#556B2F', facecolor='none',
            transform=projection, zorder=5
        )
        
        # Get global vmin/vmax for inputs
        global_vals = []
        for input_name in input_names:
            if forcing:
                input_idx = batch.forcing.feature_names_to_idx[input_name]
            else:
                input_idx = batch.inputs.feature_names_to_idx[input_name]
            
            if step == 0:
                if forcing:
                    tmp = batch.forcing.tensor[0, 0, :, :, input_idx].detach().cpu()
                else:
                    tmp = batch.inputs.tensor[0, 0, :, :, input_idx].detach().cpu()
                
                tmp = denormalize(
                    tmp,
                    torch.tensor(stats[input_name]["mean"]),
                    torch.tensor(stats[input_name]["std"])
                ).numpy()
            else:
                tmp = output.tensor[0, step - 1, :, :, input_idx].detach().cpu().numpy()
            
            global_vals.append(tmp)
        
        global_vals = np.stack(global_vals)
        vmin_global = np.min(global_vals)
        vmax_global = np.max(global_vals)
        
        # Plot each input
        for col, input_name in enumerate(input_names):
            # Input axis
            ax_in = fig.add_subplot(gs[0, col], projection=projection)
            ax_in.add_feature(cfeature.COASTLINE, linewidth=0.6)
            ax_in.set_extent(extent, crs=projection)
            
            input_idx = batch.forcing.feature_names_to_idx[input_name]
            
            if step == 0:
                inp = batch.forcing.tensor[0, 0, :, :, input_idx].detach().cpu()
                inp = denormalize(
                    inp,
                    torch.tensor(stats[input_name]["mean"]),
                    torch.tensor(stats[input_name]["std"])
                ).numpy()
            else:
                inp = output.tensor[0, step - 1, :, :, input_idx].detach().cpu().numpy()
            
            im_in = ax_in.imshow(
                inp, cmap="RdBu_r", origin="lower", extent=extent,
                transform=projection, vmin=vmin_global, vmax=vmax_global
            )
            ax_in.set_title(f"Input (t+{step}) — {input_name}", fontsize=11, pad=6)
            ax_in.add_patch(copy.deepcopy(rect))
            
            cax_in = ax_in.inset_axes([1.005, 0.15, 0.012, 0.7])
            fig.colorbar(im_in, cax=cax_in)
            
            # Attribution axis
            ax_attr = fig.add_subplot(gs[1, col], projection=projection)
            ax_attr.add_feature(cfeature.COASTLINE, linewidth=0.6)
            ax_attr.set_extent(extent, crs=projection)
            
            raw_attr = explanations[input_name][step]
            
            if raw_attr.dim() == 5:
                raw_attr = raw_attr[0, 0, :, :, 0].cpu()
            else:
                raw_attr = raw_attr[0, :, :, 0].cpu()
            
            attr = normalize_attr(raw_attr)
            
            im_attr = ax_attr.imshow(
                attr, cmap="RdBu_r", origin="lower",
                extent=extent, vmin=-1, vmax=1,
                transform=projection
            )
            ax_attr.set_title(f"Attribution — {input_name}", fontsize=11, pad=6)
            ax_attr.add_patch(copy.deepcopy(rect))
            
            cax_attr = ax_attr.inset_axes([1.005, 0.15, 0.012, 0.7])
            fig.colorbar(im_attr, cax=cax_attr)
        
        fig.suptitle(
            f"Attribution Maps - {runtime} + {step+1}h",
            fontsize=16, y=0.998
        )
        
        fig_path_full = os.path.join(fig_path, method)
        os.makedirs(fig_path_full, exist_ok=True)
        
        if save_figs:
            plt.savefig(
                os.path.join(fig_path_full, f"{runtime}.png"),
                dpi=300, bbox_inches="tight"
            )
            plt.close(fig)
        else:
            plt.show()
    
    def _create_gif(self, explainer_name: str, frame_paths: List[str], fps: int = 2):
        """Create GIF from saved frames."""
        cfg_xai = self.config.cfg_xai
        images = []
        
        for img_path in frame_paths:
            if os.path.exists(img_path):
                images.append(imageio.imread(img_path))
        
        if not images:
            self.logger.warning(f"No frames found for {explainer_name}")
            return
        
        gif_dir = os.path.join(
            cfg_xai["explain"]["exp_plot"]["fig_path"],
            explainer_name
        )
        os.makedirs(gif_dir, exist_ok=True)
        
        gif_path = os.path.join(gif_dir, "timestep_evolution.gif")
        imageio.mimsave(gif_path, images, fps=fps, palettesize=256, subrectangles=True)
        
        self.logger.info(f"GIF saved: {gif_path}")


class MultiLevelVisualizer:
    """Visualizes explanations across different altitude levels."""
    
    def __init__(self, model, config, dataloader, dataset_info, infer_ds):
        self.model = model
        self.config = config
        self.dataloader = dataloader
        self.dataset_info = dataset_info
        self.infer_ds = infer_ds
        self.logger = logging.getLogger(__name__)

    def run(self):
        import matplotlib
        matplotlib.use('Agg')

        from .comparison import plot_diff_levels_optionA,plot_diff_levels_zoomed

        cfg_xai    = self.config.cfg_xai
        num_steps  = self.config.cfg_dataset["data"]["num_pred_steps_val_test"]
        device     = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        for explainer_name, expl_params in cfg_xai["explain"]["explainers"].items():
            self.logger.info(f"Running {explainer_name}")
            explainer_cls = explainers_registry.get(explainer_name)
            explainer     = explainer_cls(self.model, **expl_params)

            for batch_idx, batch in enumerate(self.dataloader):

                # ── Runtime label ────────────────────────────────────────────
                batch_size  = batch.inputs.tensor.shape[0]
                idx_samples = [batch_idx * batch_size + b for b in range(batch_size)]
                samples     = [self.infer_ds.sample_list[i] for i in idx_samples]
                runtime     = samples[0].timestamps.datetime.strftime("%Y-%m-%d %H")

                # ── Move to device ───────────────────────────────────────────
                batch.inputs.tensor  = batch.inputs.tensor.to(device)
                batch.forcing.tensor = batch.forcing.tensor.to(device)
                if batch.outputs is not None:
                    batch.outputs.tensor = batch.outputs.tensor.to(device)

                # ── Collect attributions for every step ──────────────────────
                # steps_explanations[step] = {input_name: tensor (512,640)}
                steps_explanations = {}
                last_output        = None

                for step in range(1, num_steps + 1):
                    # Tell the model to run up to this horizon only
                    batch.num_pred_steps = step

                  
                    with torch.autograd.set_grad_enabled(True):
                        batch.inputs.tensor.requires_grad_()
                        batch.forcing.tensor.requires_grad_()

                        exp, output = explainer.compute_different_explanations(
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
                            fig_path=cfg_xai["explain"]["exp_plot"]["fig_path"],
                            return_output=True,
                        )

                    if exp is None:
                        continue

                    # exp = {input_name: [tensor]}  — list has 1 element (this step)
                    steps_explanations[step] = {
                        k: v[0].reshape(512, 640) for k, v in exp.items()
                    }
                    last_output = output

                # Restore original num_steps
                self.config.cfg_dataset["data"]["num_pred_steps_val_test"] = num_steps

                # ── Single NeurIPS figure for this batch ─────────────────────
                if steps_explanations and last_output is not None:
                    # plot_diff_levels_zoomed(
                    #         explanations=steps_explanations[1],
                    #         batch=batch, output=last_output,
                    #         extent=cfg_xai["extent"],
                    #         method=explainer_name,
                    #         fig_path=cfg_xai["explain"]["exp_plot"]["fig_path"],
                    #         runtime=runtime,
                    #         stats=self.dataset_info.stats,
                    #         output_name=cfg_xai["explain"]["output"],
                    #         target=cfg_xai["target"],
                    #         forcing=cfg_xai["explain"]["forcing_input"],
                    #         save_figs=True,step=0,
                    #         input_names=cfg_xai["multiple_explain"]
                    #     )
                    plot_diff_levels_optionA(
                            steps_explanations=steps_explanations,
                            batch=batch, output=last_output,
                            extent=cfg_xai["extent"],
                            method=explainer_name,
                            fig_path=cfg_xai["explain"]["exp_plot"]["fig_path"],
                            runtime=runtime,
                            stats=self.dataset_info.stats,
                            output_name=cfg_xai["explain"]["output"],
                            target=cfg_xai["target"],
                            forcing=cfg_xai["explain"]["forcing_input"],
                            save_figs=True,
                            input_names=cfg_xai["multiple_explain"]
                        )
                torch.cuda.empty_cache()

              