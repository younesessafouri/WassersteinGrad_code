"""
Spatial Analysis and PCA Analysis Modules
==========================================
"""

import os
import logging
import torch
import numpy as np
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from matplotlib.patches import Rectangle
import copy
from typing import Dict, List, Optional, Tuple


# ============================================================================
# SPATIAL ANALYSIS MODULE (spatial_analysis.py)
# ============================================================================

class SpatialAnalyzer:
    """Analyzes spatial distribution of attributions."""
    
    def __init__(self, model, config, dataloader, dataset_info, infer_ds):
        self.model = model
        self.config = config
        self.dataloader = dataloader
        self.dataset_info = dataset_info
        self.infer_ds = infer_ds
        self.logger = logging.getLogger(__name__)
        
    def run(self):
        """Run spatial distribution analysis."""
        from .plot_utils import plot_radial_distribution
        
        cfg_xai = self.config.cfg_xai
        
        # Collect explanations for specified inputs
        all_explanations = self._collect_explanations()
        
        # Plot radial distribution
        save_path = os.path.join(
            cfg_xai["explain"]["exp_plot"]["fig_path"],
            "spatial_analysis"
        )
        
        plot_radial_distribution(
            all_explanations,
            extent=cfg_xai["extent"],
            target=cfg_xai["target"],
            num_bins=50,
            save_path=save_path
        )
        
        self.logger.info(f"Spatial analysis plots saved to {save_path}")
        return all_explanations
    
    def _collect_explanations(self) -> Dict:
        """Collect explanations for spatial distribution analysis."""
        from py4castxai.explainers import SmoothGrad, InputxGrad, BaseGrad
        
        cfg_xai = self.config.cfg_xai
        all_explanations = {}
        
        # Get first explainer (assuming all produce similar spatial patterns)
        explainer_name = list(cfg_xai["explain"]["explainers"].keys())[0]
        expl_params = cfg_xai["explain"]["explainers"][explainer_name]
        
        # Map explainer names to classes
        explainer_map = {
            "SmoothGrad": SmoothGrad,
            "InputxGrad": InputxGrad,
            "BaseGrad": BaseGrad
        }
        
        explainer_cls = explainer_map.get(explainer_name, BaseGrad)
        
        # Collect for each input in spatial_distribution config
        for input_name in cfg_xai.get("spatial_distribution", []):
            self.logger.info(f"Collecting explanations for: {input_name}")
            
            # Temporarily set input name
            original_input = cfg_xai["explain"].get("input")
            cfg_xai["explain"]["input"] = input_name
            
            explainer = explainer_cls(self.model, **expl_params)
            explanations = self._compute_explanations_for_input(explainer, input_name)
            
            all_explanations[input_name] = explanations
            
            # Restore original
            if original_input:
                cfg_xai["explain"]["input"] = original_input
        
        return all_explanations
    
    def _compute_explanations_for_input(self, explainer, input_name: str) -> List:
        """Compute explanations for a specific input."""
        cfg_xai = self.config.cfg_xai
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        explanations = []
        
        for batch_idx, batch in enumerate(self.dataloader):
            batch.inputs.tensor = batch.inputs.tensor.to(device)
            if batch.outputs is not None:
                batch.outputs.tensor = batch.outputs.tensor.to(device)
            batch.forcing.tensor = batch.forcing.tensor.to(device)
            
            with torch.autograd.set_grad_enabled(True):
                batch.inputs.tensor.requires_grad_()
                batch.forcing.tensor.requires_grad_()
                
                explanation = explainer.compute_explanations(
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
                    fig_path=None
                )
                
                if explanation is not None:
                    explanations.append(explanation)
            
            torch.cuda.empty_cache()
        
        return explanations


