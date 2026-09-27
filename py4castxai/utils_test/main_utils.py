"""
Core utilities for Py4CastXAI experiments.

This module contains configuration/model/data management plus the experiment
runner used for inference, explanation generation, evaluation, and the retained
analysis modes.
"""

import logging
import os
import pickle
import time
from dataclasses import dataclass
from typing import Dict, Optional

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import yaml

from py4cast.datasets import get_datasets
from py4cast.models import build_model_from_settings
from py4castxai.explainers import explainers_registry
from py4castxai.utils_test.infer_utils import predict_step


def set_seed(seed: int=42):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)

@dataclass
class ExperimentConfig:
    """Centralized experiment configuration."""
    checkpoint: Dict
    cfg: Dict
    cfg_dataset: Dict
    cfg_model: Dict
    cfg_xai: Dict

    @classmethod
    def from_path(cls, xai_path: str) -> 'ExperimentConfig':
        """Load all configs from XAI config path."""
        with open(xai_path) as f:
            cfg_xai = yaml.safe_load(f)
        with open(cfg_xai['dataset_path']) as f:
            cfg_dataset = yaml.safe_load(f)
        with open(cfg_xai['model_path']) as f:
            cfg_model = yaml.safe_load(f)
        with open(cfg_xai['trainer_path']) as f:
            cfg = yaml.safe_load(f)
        checkpoint = torch.load(cfg_xai['ckpt_path'], weights_only=False)
        return cls(checkpoint, cfg, cfg_dataset, cfg_model, cfg_xai)

class ModelManager:
    """Handles model building and loading."""

    def __init__(self, config: ExperimentConfig):
        self.config = config
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    def build_model(self, dataset_info, weight_path: Optional[str]=None):
        """Build and load model with optional custom weights."""
        cfg_model = self.config.cfg_model
        cfg_dataset = self.config.cfg_dataset
        settings = cfg_model['model']['settings_init_args']
        model_name = cfg_model['model']['model_name']
        statics = dataset_info.statics
        num_input_features = cfg_dataset['data']['num_input_steps'] * dataset_info.weather_dim + statics.grid_statics.dim_size('features') + dataset_info.forcing_dim
        num_output_features = dataset_info.weather_dim
        model, _ = build_model_from_settings(model_name, num_input_features, num_output_features, settings, statics.grid_shape)
        state_dict = {k.replace('model.', ''): v for k, v in self.config.checkpoint['state_dict'].items()}
        model.load_state_dict(state_dict, strict=False)
        if weight_path:
            self._load_custom_weights(model, weight_path)
        model.eval()
        model.to(self.device)
        return model

    def _load_custom_weights(self, model, weight_path: str):
        """Load custom weights from .pth file."""
        from collections import OrderedDict
        weights = torch.load(weight_path, map_location='cpu')
        if 'state_dict' in weights:
            weights = weights['state_dict']
        new_state_dict = OrderedDict()
        for k, v in weights.items():
            new_key = k.replace('model.', '')
            new_state_dict[new_key] = v
        model.load_state_dict(new_state_dict, strict=False)

class DataManager:
    """Handles dataset loading and dataloader creation."""

    def __init__(self, config: ExperimentConfig):
        self.config = config

    def get_dataloader(self):
        """Create prediction dataloader and get dataset info."""
        cfg_data = self.config.cfg_dataset['data']
        train_ds, _, infer_ds = get_datasets(cfg_data['dataset_name'], cfg_data['num_input_steps'], cfg_data['num_pred_steps_train'], cfg_data['num_pred_steps_val_test'], cfg_data['dataset_conf'])
        dataloader = infer_ds.torch_dataloader(batch_size=cfg_data['batch_size'], num_workers=cfg_data['num_workers'], shuffle=False, prefetch_factor=cfg_data['prefetch_factor'], pin_memory=cfg_data['pin_memory'])
        return (dataloader, infer_ds.dataset_info, infer_ds)

class ExplainerManager:
    """Manages explainer initialization and execution."""

    def __init__(self, model, config: ExperimentConfig):
        self.model = model
        self.config = config
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    def get_explainer(self, explainer_name: str, params: Dict):
        """Initialize explainer by name."""
        explainer_cls = explainers_registry.get(explainer_name)
        if explainer_cls is None:
            raise ValueError(f'Unknown explainer: {explainer_name}')
        return explainer_cls(self.model, **params)

    def compute_explanations(self, dataloader, dataset_info, infer_ds, explainer_name: str, explainer_params: Dict, return_explanations: bool=False):
        """Compute explanations for all batches."""
        explainer = self.get_explainer(explainer_name, explainer_params)
        explanations = []
        cfg_xai = self.config.cfg_xai
        for batch_idx, batch in enumerate(dataloader):
            batch = self._prepare_batch(batch)
            extent = cfg_xai['extent']
            target = cfg_xai['target']
            lon0, lon1 = (target[0], target[1])
            lat0, lat1 = (target[2], target[3])
            H, W = batch.inputs.tensor.shape[2:4]
            lat_min, lat_max = (extent[2], extent[3])
            lon_min, lon_max = (extent[0], extent[1])
            lat_step = (lat_max - lat_min) / H
            lon_step = (lon_max - lon_min) / W
            i_min = int((lat0 - lat_min) / lat_step)
            i_max = int((lat1 - lat_min) / lat_step)
            j_min = int((lon0 - lon_min) / lon_step)
            j_max = int((lon1 - lon_min) / lon_step)
            output_idx = batch.outputs.feature_names_to_idx[cfg_xai['explain']['output']]
            with torch.autograd.set_grad_enabled(True):
                batch.inputs.tensor.requires_grad_()
                batch.forcing.tensor.requires_grad_()
                explanation = explainer.compute_explanations(batch, batch_idx, self.config.checkpoint, self.config.cfg_model['model'], self.config.cfg_dataset['data'], cfg_xai, dataset_info, infer_ds, cfg_xai['list_run_hour'], cfg_xai['use_old_weights'], cfg_xai['target'], plot_explanations=cfg_xai['explain']['exp_plot']['plot_explanations'], extent=cfg_xai['extent'], fig_path=cfg_xai['explain']['exp_plot']['fig_path'])
                if explanation is not None and return_explanations:
                    input_idx = batch.inputs.feature_names_to_idx[cfg_xai['explain']['input']]
                    output_idx = batch.outputs.feature_names_to_idx[cfg_xai['explain']['output']]
                    input = batch.inputs.tensor[0, 0, :, :, input_idx].detach().cpu()
                    pred = predict_step(self.model, batch, batch_idx, self.config.checkpoint, self.config.cfg_model['model'], self.config.cfg_dataset['data'], cfg_xai, dataset_info, infer_ds, cfg_xai['list_run_hour'], cfg_xai['use_old_weights'], compute_grads=False)
                    output = pred.tensor[0, -1, :, :, output_idx].detach().cpu()
                    idx_samples = [batch_idx * 1 + b for b in range(1)]
                    samples = [infer_ds.sample_list[idx] for idx in idx_samples]
                    runtimes = [sample.timestamps.datetime.strftime('%Y%m%d%H') for sample in samples]
                    return (explanation[0], input, input_idx, output, output_idx, runtimes[0])
            torch.cuda.empty_cache()
        return None

    def _prepare_batch(self, batch):
        """Move batch tensors to device."""
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        batch.inputs.tensor = batch.inputs.tensor.to(device)
        if batch.outputs is not None:
            batch.outputs.tensor = batch.outputs.tensor.to(device)
        batch.forcing.tensor = batch.forcing.tensor.to(device)
        return batch

class ExperimentRunner:
    """Main experiment orchestrator."""

    def __init__(self, xai_config_path: str):
        self.config = ExperimentConfig.from_path(xai_config_path)
        self.logger = self._setup_logging()
        self.data_manager = DataManager(self.config)
        self.dataloader, self.dataset_info, self.infer_ds = self.data_manager.get_dataloader()
        if 'extent' not in self.config.cfg_xai or self.config.cfg_xai['extent'] is None:
            self.config.cfg_xai['extent'] = self.infer_ds.dataset_info.domain_info.grid_limits
        self.model_manager = ModelManager(self.config)
        self.model = self.model_manager.build_model(self.dataset_info)
        self.explainer_manager = ExplainerManager(self.model, self.config)

    def _setup_logging(self):
        """Configure logging."""
        logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        return logging.getLogger(__name__)

    def run(self):
        """Execute the configured experiment mode."""
        mode = self.config.cfg_xai['mode']
        mode_handlers = {'introduction': self._run_introduction, 'climatological': self._run_climatological_sensitivity, 'infer': self._run_inference, 'explain': self._run_explanation, 'eval': self._run_evaluation, 'compare': self._run_comparison, 'spatial': self._run_spatial_analysis, 'pca': self._run_pca_analysis, 'autoregressive': self._run_autoregressive, 'multi_levels': self._run_multi_levels, 'distribution': self._run_distribution, 'explain_noise': self._run_explanation_noise, 'time': self._run_time, 'multi_comparison': self.run_multi_comparison, 'centroid': self._run_centroid_displacement, 'centroid_per_event': self._plot_per_event_curves, 'precip': self.analyze_precipitation_thresholds}
        try:
            handler = mode_handlers[mode]
        except KeyError as exc:
            valid = ', '.join(sorted(mode_handlers))
            raise ValueError(f'Unknown mode: {mode!r}. Available modes: {valid}') from exc
        self.logger.info('Running experiment in %s mode', mode)
        return handler()

    def _plot_per_event_curves(self, cache_dir: str='./cache_figs/centroid_step5', output_dir: str='neurips_figs/per_event', max_events: int | None=None):
        """
            Plot individual centroid and peak displacement curves for each cached
            event. Useful for diagnosing outliers, checking per-event consistency,
            and supplementing the aggregated Figure 1(c) with a per-event view.
            
            Parameters
            ----------
            cache_dir : path where event_XXXX.pkl files were stored by 
                        _run_centroid_displacement
            output_dir : where to write the per-event PDFs
            max_events : if set, only plot the first max_events cached events
            """
        import os
        import pickle
        import glob
        import numpy as np
        import matplotlib.pyplot as plt
        os.makedirs(output_dir, exist_ok=True)
        cache_files = sorted(glob.glob(os.path.join(cache_dir, 'event_*.pkl')))
        if max_events is not None:
            cache_files = cache_files[:max_events]
        self.logger.info(f'Plotting {len(cache_files)} per-event curves')
        BLUE = '#1f77b4'
        ORANGE = '#ff7f0e'
        RED = '#d62728'
        for cache_file in cache_files:
            with open(cache_file, 'rb') as f:
                result = pickle.load(f)
            if result is None:
                continue
            event_id = result.get('event_id', 'unknown')
            noises = result['noises']
            centroid_mc = result['centroid_disp_mc']
            peak_mc = result['peak_disp_mc']
            rmse_mc = result['rmse_mc']
            c_mean = np.nanmean(centroid_mc, axis=1)
            c_std = np.nanstd(centroid_mc, axis=1, ddof=1)
            p_mean = np.nanmean(peak_mc, axis=1)
            p_std = np.nanstd(peak_mc, axis=1, ddof=1)
            r_mean = np.nanmean(rmse_mc, axis=1)
            r_std = np.nanstd(rmse_mc, axis=1, ddof=1)
            fig, ax1 = plt.subplots(figsize=(5.5, 3.8))
            ax2 = ax1.twinx()
            ax1.fill_between(noises, c_mean - c_std, c_mean + c_std, color=BLUE, alpha=0.2)
            ax1.plot(noises, c_mean, color=BLUE, linewidth=1.6, label='Centroid displacement')
            ax1.fill_between(noises, p_mean - p_std, p_mean + p_std, color=ORANGE, alpha=0.15)
            ax1.plot(noises, p_mean, color=ORANGE, linewidth=1.4, linestyle='--', label='Peak displacement')
            ax2.fill_between(noises, r_mean - r_std, r_mean + r_std, color=RED, alpha=0.25)
            ax2.plot(noises, r_mean, color=RED, linewidth=1.6, label='Relative prediction error')
            ax1.set_xlabel('Noise level $\\sigma$')
            ax1.set_ylabel('Displacement (pixels)', color=BLUE)
            ax2.set_ylabel('Relative prediction error', color=RED)
            lines1, labels1 = ax1.get_legend_handles_labels()
            lines2, labels2 = ax2.get_legend_handles_labels()
            ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper left', fontsize=8)
            ax1.grid(True, axis='y', linestyle=':', linewidth=0.5, alpha=0.6)
            precip = result.get('roi_precip', None)
            precip_str = f', ROI precip = {precip:.2f} mm' if precip is not None else ''
            fig.text(0.99, 0.01, f'Event {event_id}{precip_str} | Mean ± std across MC samples', ha='right', fontsize=7, color='gray', style='italic')
            plt.tight_layout()
            out_path = os.path.join(output_dir, f'event_{event_id:04d}.pdf')
            plt.savefig(out_path, bbox_inches='tight', dpi=200)
            plt.close(fig)
        self.logger.info(f'Per-event plots written to {output_dir}/')

    def _plot_centroid_displacement(self, agg, output_path, show_peak=True, show_scatter=False, km_per_pixel=2.5):
        """
            Plot aggregated centroid and peak displacement vs noise σ in a 
            side-by-side layout: peak displacement on the left, centroid + RMSE 
            on the right. Displacement is shown in kilometers. Error bands are 
            ±1 SEM across events.
            
            Parameters
            ----------
            agg : dict
                Aggregated results from _aggregate_centroid_results.
            output_path : str
                Path to save the figure.
            show_peak : bool
                If True, include peak displacement in the left panel.
            show_scatter : bool
                If True, overlay per-event centroid curves in the right panel.
            km_per_pixel : float
                Spatial resolution (default 2.5 km/pixel for TITAN).
            """
        import numpy as np
        import matplotlib.pyplot as plt
        BLUE = '#1f77b4'
        RED = '#d62728'
        ORANGE = '#ff7f0e'
        noises = agg['noises']
        n_events = agg['n_events']
        peak_mean_km = agg['peak_disp_mean'] * km_per_pixel
        peak_sem_km = agg['peak_disp_sem'] * km_per_pixel
        centroid_mean_km = agg['centroid_disp_mean'] * km_per_pixel
        centroid_sem_km = agg['centroid_disp_sem'] * km_per_pixel
        fig, (ax_left, ax_right) = plt.subplots(1, 2, figsize=(11.0, 4.0), sharex=True)
        if show_peak:
            ax_left.fill_between(noises, peak_mean_km - peak_sem_km, peak_mean_km + peak_sem_km, color=ORANGE, alpha=0.2)
            ax_left.plot(noises, peak_mean_km, color=ORANGE, linewidth=1.8, linestyle='--', label='Peak displacement')
            ax_left.set_xlabel('Noise level $\\sigma$ (fraction of input range)')
            ax_left.set_ylabel('Peak displacement (km)', color=ORANGE)
            ax_left.tick_params(axis='y', labelcolor=ORANGE)
            ax_left.grid(True, axis='y', linestyle=':', linewidth=0.5, alpha=0.6)
            ax_left.legend(loc='upper left', fontsize=9)
        ax_right.fill_between(noises, centroid_mean_km - centroid_sem_km, centroid_mean_km + centroid_sem_km, color=BLUE, alpha=0.2)
        ax_right.plot(noises, centroid_mean_km, color=BLUE, linewidth=1.8, label='Centroid displacement')
        if show_scatter:
            for event_curve in agg['centroid_disp_perevent']:
                ax_right.plot(noises, event_curve * km_per_pixel, color=BLUE, alpha=0.08, linewidth=0.6)
        ax_right.set_xlabel('Noise level $\\sigma$ (fraction of input range)')
        ax_right.set_ylabel('Centroid displacement (km)', color=BLUE)
        ax_right.tick_params(axis='y', labelcolor=BLUE)
        ax_right.grid(True, axis='y', linestyle=':', linewidth=0.5, alpha=0.6)
        ax_rmse = ax_right.twinx()
        mr = agg['rmse_mean']
        sr = agg['rmse_sem']
        ax_rmse.fill_between(noises, mr - sr, mr + sr, color=RED, alpha=0.25)
        ax_rmse.plot(noises, mr, color=RED, linewidth=1.8, label='Relative prediction error')
        ax_rmse.set_ylabel('Relative prediction error', color=RED)
        ax_rmse.tick_params(axis='y', labelcolor=RED)
        lines_right, labels_right = ax_right.get_legend_handles_labels()
        lines_rmse, labels_rmse = ax_rmse.get_legend_handles_labels()
        ax_right.legend(lines_right + lines_rmse, labels_right + labels_rmse, loc='upper left', fontsize=9)
        fig.text(0.99, 0.01, f'Mean ± SEM across n={n_events} events', ha='right', fontsize=7, color='gray', style='italic')
        plt.tight_layout()
        plt.savefig(output_path, bbox_inches='tight', dpi=300)
        plt.close(fig)

    def _aggregate_centroid_results(self, per_event_results):
        """
            Aggregate per-event centroid displacement arrays into a summary
            suitable for plotting.
            
            Aggregation strategy (two-level):
            1. For each event, average over MC samples → per-event mean curve.
            2. Across events, compute mean and SEM of these per-event means.
            
            This gives error bars that reflect variability ACROSS EVENTS, which
            is the honest uncertainty for the claim "input perturbations systematically
            displace attributions". Pure MC variability of a single event would be
            an underestimate.
            
            Returns
            -------
            dict with keys:
                "noises"                  : (num_noise_levels,)
                "centroid_disp_mean"      : (num_noise_levels,) mean across events
                "centroid_disp_sem"       : (num_noise_levels,) SEM across events
                "centroid_disp_perevent"  : (n_events, num_noise_levels) per-event means
                                            (for scatter overlay if desired)
                "peak_disp_mean"          : (num_noise_levels,)
                "peak_disp_sem"           : (num_noise_levels,)
                "peak_disp_perevent"      : (n_events, num_noise_levels)
                "rmse_mean"               : (num_noise_levels,)
                "rmse_sem"                : (num_noise_levels,)
                "n_events"                : int
            """
        import numpy as np
        noises = per_event_results[0]['noises']
        n_events = len(per_event_results)
        num_noise_levels = len(noises)
        centroid_stack = np.stack([r['centroid_disp_mc'] for r in per_event_results], axis=0)
        peak_stack = np.stack([r['peak_disp_mc'] for r in per_event_results], axis=0)
        rmse_stack = np.stack([r['rmse_mc'] for r in per_event_results], axis=0)
        centroid_perevent = np.nanmean(centroid_stack, axis=2)
        peak_perevent = np.nanmean(peak_stack, axis=2)
        rmse_perevent = np.nanmean(rmse_stack, axis=2)

        def mean_sem(x):
            mean = np.nanmean(x, axis=0)
            std = np.nanstd(x, axis=0, ddof=1)
            n = np.sum(~np.isnan(x), axis=0)
            sem = std / np.sqrt(np.maximum(n, 1))
            return (mean, sem)
        centroid_mean, centroid_sem = mean_sem(centroid_perevent)
        peak_mean, peak_sem = mean_sem(peak_perevent)
        rmse_mean, rmse_sem = mean_sem(rmse_perevent)
        return {'noises': noises, 'centroid_disp_mean': centroid_mean, 'centroid_disp_sem': centroid_sem, 'centroid_disp_perevent': centroid_perevent, 'peak_disp_mean': peak_mean, 'peak_disp_sem': peak_sem, 'peak_disp_perevent': peak_perevent, 'rmse_mean': rmse_mean, 'rmse_sem': rmse_sem, 'n_events': n_events}

    def _run_centroid_displacement(self):
        """
            Compute centroid/peak displacement of attribution maps under input 
            perturbations, aggregated across multiple atmospheric events filtered
            by mean precipitation over the Paris ROI.
            
            Produces Figure 1(c) of the paper: displacement vs. noise level,
            with error bars reflecting variability across events (not across
            MC samples of a single event).
            """
        import os
        import pickle
        import numpy as np
        import torch
        from tqdm import tqdm
        cfg_centroid = self.config.cfg_xai.get('centroid', {})
        cfg_xai = self.config.cfg_xai
        max_events = cfg_centroid.get('max_events', 50)
        sigma_step = cfg_centroid.get('sigma_step', 0.1)
        num_noise_levels = cfg_centroid.get('num_noise_levels', 10)
        num_monte_carlo = cfg_centroid.get('num_monte_carlo', 10)
        base_explainer = cfg_centroid.get('base_explainer', 'BaseGrad')
        saliency_thresh = cfg_centroid.get('saliency_threshold', None)
        cache_dir = cfg_centroid.get('cache_dir', './cache/centroid')
        output_path = cfg_centroid.get('output_path', 'figures/centroid_displacement.pdf')
        force_recompute = cfg_centroid.get('force_recompute', False)
        min_precip_threshold = cfg_centroid.get('min_precip_threshold', 5.8)
        os.makedirs(cache_dir, exist_ok=True)
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        extent = cfg_xai['extent']
        target = cfg_xai['target']
        output_name = cfg_xai['explain']['output']
        lon0, lon1 = (target[0], target[1])
        lat0, lat1 = (target[2], target[3])
        lat_min, lat_max = (extent[2], extent[3])
        lon_min, lon_max = (extent[0], extent[1])
        first_batch = next(iter(self.dataloader))
        first_batch = self.explainer_manager._prepare_batch(first_batch)
        H, W = first_batch.inputs.tensor.shape[2:4]
        lat_step = (lat_max - lat_min) / H
        lon_step = (lon_max - lon_min) / W
        i_min = int((lat0 - lat_min) / lat_step)
        i_max = int((lat1 - lat_min) / lat_step)
        j_min = int((lon0 - lon_min) / lon_step)
        j_max = int((lon1 - lon_min) / lon_step)
        centroid_type = cfg_centroid['type']
        print(centroid_type)
        centroid_explainer = explainers_registry[centroid_type](model=self.model, base_explainer=base_explainer, sigma_step=sigma_step, num_noise_levels=num_noise_levels, num_monte_carlo=num_monte_carlo, saliency_threshold=saliency_thresh)
        self.logger.info(f'[centroid_displacement] Collecting up to {max_events} events (min_precip={min_precip_threshold})')
        per_event_results = []
        skipped_low_precip = 0
        skipped_failed = []
        accepted_events = 0
        for batch_idx, batch in enumerate(tqdm(self.dataloader, desc='Events')):
            if accepted_events >= max_events:
                break
            batch = self.explainer_manager._prepare_batch(batch)
            output_idx = batch.outputs.feature_names_to_idx[output_name]
            roi_precip = batch.outputs.tensor[:, 0, i_min:i_max, j_min:j_max, output_idx].mean().item()
            if roi_precip < min_precip_threshold:
                self.logger.debug(f'  batch {batch_idx:04d} skipped (precip={roi_precip:.3f} < {min_precip_threshold})')
                skipped_low_precip += 1
                torch.cuda.empty_cache()
                continue
            cache_file = os.path.join(cache_dir, f'event_{batch_idx:04d}.pkl')
            if os.path.exists(cache_file) and (not force_recompute):
                self.logger.info(f'Loading cached result for event {batch_idx}')
                with open(cache_file, 'rb') as f:
                    result = pickle.load(f)
                if result is not None:
                    result['roi_precip'] = roi_precip
                    per_event_results.append(result)
                    accepted_events += 1
                else:
                    skipped_failed.append(batch_idx)
                torch.cuda.empty_cache()
                continue
            try:
                result = centroid_explainer.compute_explanations(input=batch, batch_idx=batch_idx, checkpoint=self.config.checkpoint, cfg_model=self.config.cfg_model['model'], cfg_dataset=self.config.cfg_dataset['data'], cfg_xai=cfg_xai, dataset_info=self.dataset_info, infer_ds=self.infer_ds, list_run_hour=cfg_xai['list_run_hour'], use_old_weights=cfg_xai['use_old_weights'], target=cfg_xai['explain'].get('target', None))
            except Exception as e:
                self.logger.warning(f'Event {batch_idx} failed: {e}')
                result = None
            if result is not None:
                result['roi_precip'] = roi_precip
            with open(cache_file, 'wb') as f:
                pickle.dump(result, f)
            if result is not None:
                per_event_results.append(result)
                accepted_events += 1
            else:
                skipped_failed.append(batch_idx)
            torch.cuda.empty_cache()
        n_events = len(per_event_results)
        self.logger.info(f'[centroid_displacement] Summary:\n  Accepted events: {n_events}\n  Skipped (low precipitation): {skipped_low_precip}\n  Skipped (computation failed): {len(skipped_failed)}')
        if n_events == 0:
            self.logger.error('No successful events — nothing to plot.')
            return
        agg = self._aggregate_centroid_results(per_event_results)
        agg['min_precip_threshold'] = min_precip_threshold
        agg['n_skipped_low_precip'] = skipped_low_precip
        agg['n_skipped_failed'] = len(skipped_failed)
        with open(os.path.join(cache_dir, 'aggregated.pkl'), 'wb') as f:
            pickle.dump({'per_event': per_event_results, 'aggregated': agg}, f)
        self._plot_centroid_displacement(agg, output_path=output_path)
        self.logger.info(f'Saved centroid displacement figure → {output_path}')
        return agg

    def _run_introduction(self):
        """Run inference mode."""
        from py4castxai.utils_test.infer_utils import predict_step
        for batch_idx, batch in enumerate(self.dataloader):
            import torch
            import numpy as np
            import matplotlib.pyplot as plt
            import matplotlib.colors as mcolors
            import cartopy.crs as ccrs
            import cartopy.feature as cfeature
            from mpl_toolkits.mplot3d import Axes3D
            EXTENT = [-6.0, 9.975, 40.125, 52.9]
            feature_names = ['aro_tp_0m', 'aro_t2m_2m', 'aro_r2_2m', 'aro_u10_10m', 'aro_u_250hpa', 'aro_z_250hpa']
            feature_ids = []
            for feat_name in feature_names:
                feature_ids.append(batch.inputs.feature_names_to_idx[feat_name])
            CMAPS = ['RdBu_r', 'RdBu_r', 'viridis', 'PuOr', 'coolwarm', 'coolwarm']
            LAYER_GAP = 1.1
            DPI = 120
            DOWNSAMPLE = 2
            batch = self.explainer_manager._prepare_batch(batch)
            x = batch.inputs.tensor.detach().cpu()[0, 0]
            x = x[::DOWNSAMPLE, ::DOWNSAMPLE, :]
            lat, lon, F = x.shape

            def field_to_rgba(data, cmap_name, extent, dpi=DPI):
                """
                    Render a 2D field onto a Cartopy PlateCarree axis,
                    return the rasterised RGBA array (H, W, 4).
                    """
                fig_c, ax_c = plt.subplots(figsize=(lon / dpi * 4, lat / dpi * 4), subplot_kw={'projection': ccrs.PlateCarree()}, dpi=dpi)
                ax_c.set_extent(extent, crs=ccrs.PlateCarree())
                lon_vals = np.linspace(extent[0], extent[1], data.shape[1])
                lat_vals = np.linspace(extent[2], extent[3], data.shape[0])
                lon2d, lat2d = np.meshgrid(lon_vals, lat_vals)
                vmin, vmax = np.percentile(data, [2, 98])
                ax_c.pcolormesh(lon2d, lat2d, data, cmap=cmap_name, vmin=vmin, vmax=vmax, transform=ccrs.PlateCarree(), shading='auto')
                ax_c.add_feature(cfeature.COASTLINE, linewidth=0.8, edgecolor='black')
                ax_c.add_feature(cfeature.BORDERS, linewidth=0.5, edgecolor='black', linestyle='--')
                ax_c.add_feature(cfeature.LAND, facecolor='none')
                ax_c.add_feature(cfeature.OCEAN, facecolor='#d0e8f0', alpha=0.3)
                ax_c.axis('off')
                fig_c.subplots_adjust(left=0, right=1, top=1, bottom=0)
                fig_c.canvas.draw()
                buf = fig_c.canvas.buffer_rgba()
                rgba = np.asarray(buf).copy()
                plt.close(fig_c)
                return rgba.astype(float) / 255.0
            fig = plt.figure(figsize=(12, 9), facecolor='white')
            ax3d = fig.add_subplot(111, projection='3d')
            ax3d.set_facecolor('white')
            X_unit = np.linspace(0, 1, lon)
            Y_unit = np.linspace(0, 1, lat)
            XX, YY = np.meshgrid(X_unit, Y_unit)
            for layer_idx, (fid, fname, cmap_name) in enumerate(zip(feature_ids, feature_names, CMAPS)):
                data = x[:, :, fid].numpy()
                rgba = field_to_rgba(data, cmap_name, EXTENT)
                from PIL import Image
                tex = Image.fromarray((rgba * 255).astype(np.uint8))
                tex = tex.resize((lon, lat), Image.LANCZOS)
                rgba_resized = np.array(tex) / 255.0
                ZZ = np.ones_like(XX) * layer_idx * LAYER_GAP
                ax3d.plot_surface(XX, YY, ZZ, facecolors=rgba_resized, rstride=1, cstride=1, linewidth=0, antialiased=False, shade=False, alpha=0.92)
                border_x = [0, 1, 1, 0, 0]
                border_y = [0, 0, 1, 1, 0]
                border_z = [layer_idx * LAYER_GAP] * 5
                ax3d.plot(border_x, border_y, border_z, color='black', linewidth=0.6, alpha=0.6)
            for cx, cy in [(0, 0), (1, 0), (1, 1), (0, 1)]:
                ax3d.plot([cx, cx], [cy, cy], [0, (len(feature_ids) - 1) * LAYER_GAP], color='gray', linewidth=0.4, alpha=0.4, linestyle=':')
            ax3d.set_xticks([])
            ax3d.set_yticks([])
            ax3d.set_zticks([])
            ax3d.set_axis_off()
            ax3d.view_init(elev=28, azim=-55)
            ax3d.set_box_aspect([1.6, 1.0, 1.8])
            plt.tight_layout()
            plt.savefig('atmospheric_stack.pdf', dpi=300, bbox_inches='tight', facecolor='white')
            plt.close()
            break

    def _run_inference(self):
        """Run inference mode."""
        from py4castxai.utils_test.infer_utils import predict_step
        for batch_idx, batch in enumerate(self.dataloader):
            batch = self.explainer_manager._prepare_batch(batch)
            predict_step(self.model, batch, batch_idx, self.config.checkpoint, self.config.cfg_model['model'], self.config.cfg_dataset['data'], self.config.cfg_xai, self.dataset_info, self.infer_ds, self.config.cfg_xai['list_run_hour'], self.config.cfg_xai['use_old_weights'], compute_grads=False)

    def _run_explanation(self):
        """Run explanation generation."""
        for explainer_name, expl_params in self.config.cfg_xai['explain']['explainers'].items():
            self.logger.info(f'Generating explanations with {explainer_name}')
            self.explainer_manager.compute_explanations(self.dataloader, self.dataset_info, self.infer_ds, explainer_name, expl_params, return_explanations=False)

    def _run_climatological_sensitivity(self):
        """
            Replicates Figure 1 from Torn & Hakim (2008): Maps the percentage 
            of forecast cycles for which grid-point significance is met.
            """
        import os
        import numpy as np
        import torch
        import matplotlib.pyplot as plt
        import cartopy.crs as ccrs
        import cartopy.feature as cfeature
        from tqdm import tqdm
        cfg_xai = self.config.cfg_xai
        cfg_clim = cfg_xai.get('climatological', {})
        confidence_level = cfg_clim.get('confidence_level', 0.95)
        max_cycles = cfg_clim.get('max_cycles', 100)
        output_path = cfg_clim.get('output_path', 'figures/climatological_sensitivity.pdf')
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        explainer_name = cfg_clim.get('explainer_name', 'StatGrad')
        expl_params = cfg_xai['explain']['explainers'].get(explainer_name, {'std_perturbations': 0.1, 'num_perturbations': 50})
        explainer = self.explainer_manager.get_explainer(explainer_name, expl_params)
        significance_counts = None
        total_cycles = 0
        self.logger.info(f'[climatological] Starting frequency mapping over up to {max_cycles} cycles...')
        for batch_idx, batch in enumerate(tqdm(self.dataloader, desc='Climatology')):
            if total_cycles >= max_cycles:
                break
            batch = self.explainer_manager._prepare_batch(batch)
            with torch.autograd.set_grad_enabled(True):
                batch.inputs.tensor.requires_grad_()
                batch.forcing.tensor.requires_grad_()
                try:
                    grad_list, _ = explainer.compute_explanations(input=batch, batch_idx=batch_idx, checkpoint=self.config.checkpoint, cfg_model=self.config.cfg_model['model'], cfg_dataset=self.config.cfg_dataset['data'], cfg_xai=cfg_xai, dataset_info=self.dataset_info, infer_ds=self.infer_ds, list_run_hour=cfg_xai['list_run_hour'], use_old_weights=cfg_xai['use_old_weights'], target=cfg_xai['target'], plot_explanations=False, extent=cfg_xai['extent'], fig_path=cfg_xai['explain']['exp_plot']['fig_path'], return_output=True)
                except Exception as e:
                    self.logger.warning(f'Cycle {batch_idx} failed: {e}')
                    continue
                if grad_list is None:
                    continue
                curr_grad = grad_list[0]
                if isinstance(curr_grad, torch.Tensor):
                    curr_grad = curr_grad.detach().cpu().numpy()
                curr_grad = np.squeeze(curr_grad)
                while curr_grad.ndim > 2:
                    curr_grad = curr_grad[0]
                if curr_grad.ndim != 2:
                    self.logger.warning(f'Skipping batch {batch_idx}: expected 2D spatial array, got shape {curr_grad.shape}')
                    continue
                is_significant = (np.abs(curr_grad) > 0).astype(float)
                if significance_counts is None:
                    significance_counts = np.zeros_like(is_significant)
                significance_counts += is_significant
                total_cycles += 1
            torch.cuda.empty_cache()
        if total_cycles == 0:
            self.logger.error('[climatological] No cycles processed successfully.')
            return
        frequency_map = significance_counts / total_cycles * 100.0
        fig, ax = plt.subplots(figsize=(9, 6), subplot_kw={'projection': ccrs.PlateCarree()})
        extent = self.config.cfg_xai['extent']
        ax.set_extent(extent, crs=ccrs.PlateCarree())
        im = ax.imshow(frequency_map, origin='lower', cmap='Blues', extent=extent, transform=ccrs.PlateCarree(), vmin=0, vmax=100)
        ax.add_feature(cfeature.COASTLINE, linewidth=0.8, edgecolor='black')
        ax.add_feature(cfeature.BORDERS, linewidth=0.5, edgecolor='black', linestyle='--')
        ax.add_feature(cfeature.LAND, facecolor='none')
        ax.add_feature(cfeature.OCEAN, facecolor='#f0f0f0', alpha=0.5)
        cbar = fig.colorbar(im, ax=ax, orientation='horizontal', pad=0.08, fraction=0.046)
        cbar.set_label(f'Percentage of cycles significant at {confidence_level * 100}% confidence (%)', fontsize=9)
        ax.set_title(f'Climatological Sensitivity Frequency (n={total_cycles} cycles)', fontsize=10)
        plt.savefig(output_path, bbox_inches='tight', dpi=300)
        plt.close(fig)
        self.logger.info(f'[climatological] Saved frequency map to {output_path}')
        return frequency_map

    def _run_time(self):
        """Run explanation generation and benchmark wall-time per explainer."""
        for explainer_name, expl_params in self.config.cfg_xai['explain']['explainers'].items():
            self.logger.info(f'Timing {explainer_name}')
            explainer_cls = explainers_registry.get(explainer_name)
            explainer = explainer_cls(self.model, **expl_params)
            cfg_xai = self.config.cfg_xai
            batch_times = []
            for batch_idx, batch in enumerate(self.dataloader):
                device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
                batch.inputs.tensor = batch.inputs.tensor.to(device)
                if batch.outputs is not None:
                    batch.outputs.tensor = batch.outputs.tensor.to(device)
                batch.forcing.tensor = batch.forcing.tensor.to(device)
                with torch.autograd.set_grad_enabled(True):
                    batch.inputs.tensor.requires_grad_()
                    batch.forcing.tensor.requires_grad_()
                    if torch.cuda.is_available():
                        torch.cuda.synchronize()
                    t_start = time.perf_counter()
                    explanation = explainer.compute_explanations(batch, batch_idx, self.config.checkpoint, self.config.cfg_model['model'], self.config.cfg_dataset['data'], cfg_xai, self.dataset_info, self.infer_ds, cfg_xai['list_run_hour'], cfg_xai['use_old_weights'], cfg_xai['target'], plot_explanations=cfg_xai['explain']['exp_plot']['plot_explanations'], extent=cfg_xai['extent'], fig_path=cfg_xai['explain']['exp_plot']['fig_path'])
                    if torch.cuda.is_available():
                        torch.cuda.synchronize()
                    t_end = time.perf_counter()
                if explanation is not None:
                    elapsed = t_end - t_start
                    batch_times.append(elapsed)
                    self.logger.info(f'[{explainer_name}] batch {batch_idx} | wall-time: {elapsed:.4f}s')
                torch.cuda.empty_cache()
            if batch_times:
                mean_t = sum(batch_times) / len(batch_times)
                std_t = (sum(((t - mean_t) ** 2 for t in batch_times)) / len(batch_times)) ** 0.5
                total_t = sum(batch_times)
                self.logger.info(f"\n{'=' * 50}\n[{explainer_name}] Timing Summary\n  Batches timed : {len(batch_times)}\n  Mean wall-time: {mean_t:.4f}s ± {std_t:.4f}s\n  Total wall-time: {total_t:.4f}s\n{'=' * 50}")
            else:
                self.logger.warning(f'[{explainer_name}] No valid batches were timed.')

    def _run_explanation_noise(self):
        for explainer_name, expl_params in self.config.cfg_xai['explain']['explainers'].items():
            for noise in self.config.cfg_xai['explain']['noise']:
                expl_params['std_perturbations'] = noise
                self.logger.info(f'Generating explanations with {explainer_name}')
                self.explainer_manager.compute_explanations(self.dataloader, self.dataset_info, self.infer_ds, explainer_name, expl_params, return_explanations=False)

    def _run_evaluation(self):
        """Run metric evaluation."""
        from .evaluation import EvaluationRunner, MetricAggregator
        set_seed(42)
        evaluator = EvaluationRunner(self.model, self.config, self.dataloader, self.dataset_info, self.infer_ds)
        results = evaluator.run()
        summary_table = MetricAggregator.create_summary_table(results)
        print('\n' + summary_table)
        return results
        return

    def _prepare_batch(self, batch):
        """Move batch tensors to device."""
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        batch.inputs.tensor = batch.inputs.tensor.to(device)
        if batch.outputs is not None:
            batch.outputs.tensor = batch.outputs.tensor.to(device)
        batch.forcing.tensor = batch.forcing.tensor.to(device)
        return batch

    def run_multi_comparison(self, max_dates: int=6, filename: str='appendix_multidates', min_precip_threshold: float=5.8):
        """
            Collect explanations across multiple dates and produce the
            appendix multi-date qualitative figure.

            Parameters
            ----------
            max_dates : int
                Number of dates (rows) to include in the appendix figure.
            filename : str
                Output filename stem (no extension).
            min_precip_threshold : float
                Minimum mean precipitation over the Paris ROI to accept a batch.
                Mirrors the filter in compute_explanations.
            """
        from .comparison import plot_qualitative_multidates
        cfg_xai = self.config.cfg_xai
        extent = cfg_xai['extent']
        target = cfg_xai['target']
        input_name = cfg_xai['explain']['input']
        output_name = cfg_xai['explain']['output']
        fig_path = cfg_xai['explain']['exp_plot']['fig_path']
        explainers = cfg_xai['explain']['explainers']
        lon0, lon1 = (target[0], target[1])
        lat0, lat1 = (target[2], target[3])
        lat_min, lat_max = (extent[2], extent[3])
        lon_min, lon_max = (extent[0], extent[1])
        first_batch = next(iter(self.dataloader))
        first_batch = self._prepare_batch(first_batch)
        H, W = first_batch.inputs.tensor.shape[2:4]
        lat_step = (lat_max - lat_min) / H
        lon_step = (lon_max - lon_min) / W
        i_min = int((lat0 - lat_min) / lat_step)
        i_max = int((lat1 - lat_min) / lat_step)
        j_min = int((lon0 - lon_min) / lon_step)
        j_max = int((lon1 - lon_min) / lon_step)
        all_explanations = []
        all_inputs = []
        all_preds = []
        all_runtimes = []
        self.logger.info(f'[run_multi_comparison] Collecting {max_dates} dates (min_precip={min_precip_threshold})')
        for batch_idx, batch in enumerate(self.dataloader):
            batch = self._prepare_batch(batch)
            output_idx = batch.outputs.feature_names_to_idx[output_name]
            input_idx = batch.inputs.feature_names_to_idx[input_name]
            roi_precip = batch.outputs.tensor[:, :, i_min:i_max, j_min:j_max, output_idx].mean().item()
            if roi_precip < min_precip_threshold:
                self.logger.debug(f'  batch {batch_idx:04d} skipped (precip={roi_precip:.3f} < {min_precip_threshold})')
                torch.cuda.empty_cache()
                continue
            idx_samples = [batch_idx * 1 + b for b in range(1)]
            samples = [self.infer_ds.sample_list[idx] for idx in idx_samples]
            runtime = samples[0].timestamps.datetime.strftime('%Y%m%d%H')
            self.logger.info(f'  batch {batch_idx:04d} | runtime {runtime} | precip={roi_precip:.3f} → computing explanations')
            date_explanations = {}
            inp_tensor = None
            pred_tensor = None
            for explainer_name, expl_params in explainers.items():
                self.logger.info(f'    [{runtime}] {explainer_name}')
                try:
                    with torch.autograd.set_grad_enabled(True):
                        batch.inputs.tensor.requires_grad_()
                        batch.forcing.tensor.requires_grad_()
                        explainer = self.explainer_manager.get_explainer(explainer_name, expl_params)
                        explanation = explainer.compute_explanations(batch, batch_idx, self.config.checkpoint, self.config.cfg_model['model'], self.config.cfg_dataset['data'], cfg_xai, self.dataset_info, self.infer_ds, cfg_xai['list_run_hour'], cfg_xai['use_old_weights'], target, plot_explanations=False, extent=extent, fig_path=fig_path)
                    if explanation is None:
                        self.logger.warning(f'    [{runtime}] {explainer_name} returned None — skipping date')
                        date_explanations = {}
                        break
                    date_explanations[explainer_name] = explanation[0]
                    if inp_tensor is None:
                        inp_tensor = batch.inputs.tensor[0, 0, :, :, input_idx].detach().cpu()
                        pred_raw = predict_step(self.model, batch, batch_idx, self.config.checkpoint, self.config.cfg_model['model'], self.config.cfg_dataset['data'], cfg_xai, self.dataset_info, self.infer_ds, cfg_xai['list_run_hour'], cfg_xai['use_old_weights'], compute_grads=False)
                        pred_tensor = pred_raw.tensor[0, -1, :, :, output_idx].detach().cpu()
                except Exception as e:
                    self.logger.error(f'    [{runtime}] {explainer_name} raised {e} — skipping date')
                    date_explanations = {}
                    break
                finally:
                    torch.cuda.empty_cache()
            if len(date_explanations) == len(explainers):
                all_explanations.append(date_explanations)
                all_inputs.append(inp_tensor)
                all_preds.append(pred_tensor)
                all_runtimes.append(runtime)
                self.logger.info(f'  ✓ date {runtime} added ({len(all_runtimes)}/{max_dates})')
            else:
                self.logger.warning(f'  ✗ date {runtime} dropped — not all explainers succeeded')
        if len(all_runtimes) == 0:
            self.logger.error('[run_multi_comparison] No valid dates collected. Check min_precip_threshold or dataloader.')
            return
        self.logger.info(f'[run_multi_comparison] Collected {len(all_runtimes)} dates: {all_runtimes}')
        plot_qualitative_multidates(all_explanations=all_explanations, all_inputs=all_inputs, all_preds=all_preds, all_runtimes=all_runtimes, extent=extent, stats=self.dataset_info.stats, fig_path=fig_path, input_name=input_name, output_name=output_name, input_idx=input_idx, output_idx=output_idx, target=target, filename=filename, save_figs=True)
        self.logger.info('[run_multi_comparison] Done.')

    def analyze_precipitation_thresholds(self):
        """
            Loops through the dataset to evaluate mean ROI precipitation.
            Counts samples above 1.0, 2.0, and 5.0 thresholds and logs the top 50 samples.
            """
        import torch
        cfg_xai = self.config.cfg_xai
        extent = cfg_xai['extent']
        target = cfg_xai['target']
        output_name = cfg_xai['explain']['output']
        lon0, lon1 = (target[0], target[1])
        lat0, lat1 = (target[2], target[3])
        lat_min, lat_max = (extent[2], extent[3])
        lon_min, lon_max = (extent[0], extent[1])
        first_batch = next(iter(self.dataloader))
        first_batch = self._prepare_batch(first_batch)
        H, W = first_batch.inputs.tensor.shape[2:4]
        lat_step = (lat_max - lat_min) / H
        lon_step = (lon_max - lon_min) / W
        i_min = int((lat0 - lat_min) / lat_step)
        i_max = int((lat1 - lat_min) / lat_step)
        j_min = int((lon0 - lon_min) / lon_step)
        j_max = int((lon1 - lon_min) / lon_step)
        count_1 = 0
        count_2 = 0
        count_5 = 0
        all_records = []
        self.logger.info('Starting dataset scan for precipitation thresholds...')
        with torch.no_grad():
            for batch_idx, batch in enumerate(self.dataloader):
                batch = self._prepare_batch(batch)
                output_idx = batch.outputs.feature_names_to_idx[output_name]
                roi_precip = batch.outputs.tensor[:, :, i_min:i_max, j_min:j_max, output_idx].mean().item()
                idx_samples = [batch_idx * 1 + b for b in range(1)]
                samples = [self.infer_ds.sample_list[idx] for idx in idx_samples]
                runtime = samples[0].timestamps.datetime.strftime('%Y%m%d%H')
                if roi_precip >= 1.0:
                    count_1 += 1
                if roi_precip >= 2.0:
                    count_2 += 1
                if roi_precip >= 5.0:
                    count_5 += 1
                all_records.append((runtime, roi_precip, batch_idx))
                del batch
                if batch_idx % 100 == 0:
                    self.logger.info(f'Scanned {batch_idx} samples...')
        all_records.sort(key=lambda x: x[1], reverse=True)
        top_50 = all_records[:25]
        total_samples = len(all_records)
        self.logger.info(f'=== Precipitation Threshold Analysis ({total_samples} total samples) ===')
        self.logger.info(f'Samples >= 1.0: {count_1} ({count_1 / total_samples * 100:.1f}%)')
        self.logger.info(f'Samples >= 2.0: {count_2} ({count_2 / total_samples * 100:.1f}%)')
        self.logger.info(f'Samples >= 5.0: {count_5} ({count_5 / total_samples * 100:.1f}%)')
        self.logger.info('==================================================================')
        self.logger.info('=== TOP 50 HIGHEST PRECIPITATION SAMPLES ===')
        self.logger.info(f"{'Rank':<6} | {'Runtime':<12} | {'Mean Precip':<12} | {'Batch Idx'}")
        self.logger.info('-' * 50)
        for rank, (runtime, precip, b_idx) in enumerate(top_50, 1):
            self.logger.info(f'{rank:<6} | {runtime:<12} | {precip:<12.4f} | {b_idx}')
        return all_records

    def _run_comparison(self):
        """Run explainer comparison."""
        from .comparison import plot_qualitative
        explanations = {}
        for explainer_name, expl_params in self.config.cfg_xai['explain']['explainers'].items():
            self.logger.info(f'Generating explanations with {explainer_name}')
            explanations[explainer_name], input, input_idx, output, output_idx, runtime = self.explainer_manager.compute_explanations(self.dataloader, self.dataset_info, self.infer_ds, explainer_name, expl_params, return_explanations=True)
        extent = self.config.cfg_xai['extent']
        stats = self.dataset_info.stats
        fig_path = self.config.cfg_xai['explain']['exp_plot']['fig_path']
        input_name = self.config.cfg_xai['explain']['input']
        output_name = self.config.cfg_xai['explain']['output']
        print('runtime', runtime)
        target = self.config.cfg_xai['target']
        plot_qualitative(explanations=explanations, input=input, pred=output, extent=extent, stats=stats, fig_path=fig_path, runtime=runtime, input_name=input_name, output_name=output_name, input_idx=input_idx, output_idx=output_idx, target=target, step=5, save_figs=True)

    def _run_spatial_analysis(self):
        """Run spatial distribution analysis."""
        from .spatial_analysis import SpatialAnalyzer
        analyzer = SpatialAnalyzer(self.model, self.config, self.dataloader, self.dataset_info, self.infer_ds)
        return analyzer.run()

    def _run_pca_analysis(self):
        """Run PCA-based analysis."""
        from .pca_analysis import PCAAnalyzer
        analyzer = PCAAnalyzer(self.model, self.config, self.dataloader, self.dataset_info, self.infer_ds)
        return analyzer.run()

    def _run_distribution(self):
        """Run PCA-based analysis."""
        from .distribution import DistributionRunner
        analyzer = DistributionRunner(self.model, self.config, self.dataloader, self.dataset_info, self.infer_ds)
        return analyzer.run()

    def _run_autoregressive(self):
        """Run autoregressive visualization."""
        from .visualization import AutoregressiveVisualizer
        visualizer = AutoregressiveVisualizer(self.model, self.config, self.dataloader, self.dataset_info, self.infer_ds)
        return visualizer.run()

    def _run_multi_levels(self):
        """Run autoregressive visualization."""
        from .visualization import MultiLevelVisualizer
        visualizer = MultiLevelVisualizer(self.model, self.config, self.dataloader, self.dataset_info, self.infer_ds)
        return visualizer.run()

def run_experiment(xai_config_path: str):
    """Main entry point for running experiments."""
    runner = ExperimentRunner(xai_config_path)
    return runner.run()