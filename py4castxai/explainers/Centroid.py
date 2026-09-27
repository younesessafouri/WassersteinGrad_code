from py4castxai.explainers.XGrad import XGrad
import torch
from py4castxai.utils_test.infer_utils import predict_step

from py4castxai.explainers.inputxgrad import InputxGrad
from py4castxai.explainers.basegrad import BaseGrad
from py4castxai.explainers import SmoothGrad,WassersteinGrad
from py4castxai.explainers.VarGrad import VarGrad
import numpy as np
from scipy.ndimage import center_of_mass
import torch
import ot

import numpy as np
import matplotlib
# matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from matplotlib.lines import Line2D
def normalize_expl(a: np.ndarray | torch.Tensor) -> torch.Tensor:
    """Normalize attribution map to [-1, 1]."""
    if isinstance(a, np.ndarray):
        a = torch.from_numpy(a).to(torch.float32)
    return a.abs() / (a.abs().max() + 1e-8)

def normalize_sum(a):
    a = np.abs(a)  # ensure non-negative
    return a / (np.max(a) + 1e-8)


def normalize_sum_wasser(a):
    a = np.abs(a)  # ensure non-negative
    return a / (a.sum() + 1e-8)



import numpy as np
import torch
from scipy.ndimage import center_of_mass


class Centroid(XGrad):
    """
    Measures geometric displacement of gradient attribution maps under 
    input perturbations of increasing magnitude. Returns per-event results
    for downstream aggregation across multiple events/dates.
    """

    def __init__(
        self,
        model,
        base_explainer: str = "BaseGrad",
        sigma_step: float = 0.1,
        num_noise_levels: int = 10,
        num_monte_carlo: int = 20,
        saliency_threshold: float | None = None,
    ):
        """
        Parameters
        ----------
        model : the forecasting model
        base_explainer : "BaseGrad" or "InputxGrad"
        sigma_step : increment of normalized noise level per step 
                     (e.g. 0.1 → σ ∈ {0.1, 0.2, ..., num_noise_levels * 0.1})
        num_noise_levels : number of σ values to evaluate
        num_monte_carlo : number of MC perturbation samples per σ level
        saliency_threshold : if None, uses full |gradient| (recommended for 
                             rigor). If a float in [0,1], applies threshold 
                             on normalized gradient before computing centroid.
        """
        super().__init__(model)
        if base_explainer == "InputxGrad":
            self.base_explainer = InputxGrad(model)
        else:
            self.base_explainer = BaseGrad(model)

        self.sigma_step = sigma_step
        self.num_noise_levels = num_noise_levels
        self.num_monte_carlo = num_monte_carlo
        self.saliency_threshold = saliency_threshold

    # ───────────────────────────────────────────────────────────────────
    # Helpers
    # ───────────────────────────────────────────────────────────────────

    def _compute_centroid(self, grad_2d: np.ndarray) -> tuple[float, float]:
        """
        Compute the spatial center of mass of |grad_2d|.
        If self.saliency_threshold is set, applies percentile thresholding first.
        """
        abs_grad = np.abs(grad_2d)

        if self.saliency_threshold is not None:
            # Use percentile to be scale-invariant

            abs_grad = np.where(abs_grad >= 0.1, abs_grad, 0.0)

        # Fallback: if everything is zero (degenerate), return domain center
        if abs_grad.sum() == 0:
            H, W = abs_grad.shape
            return float(H) / 2, float(W) / 2

        cy, cx = center_of_mass(abs_grad)
        return float(cx), float(cy)

    def _compute_peak(self, grad_2d: np.ndarray) -> tuple[float, float]:
        """
        Spatial coordinate of the maximum of |grad_2d|.
        Alternative displacement measure, robust to diffuse background mass.
        """
        abs_grad = np.abs(grad_2d)
        idx_flat = int(np.argmax(abs_grad))
        cy, cx = np.unravel_index(idx_flat, abs_grad.shape)
        return float(cx), float(cy)

    # ───────────────────────────────────────────────────────────────────
    # Main API
    # ───────────────────────────────────────────────────────────────────

    def compute_explanations(
        self,
        input,
        batch_idx,
        checkpoint,
        cfg_model,
        cfg_dataset,
        cfg_xai,
        dataset_info,
        infer_ds,
        list_run_hour,
        use_old_weights,
        target,
        plot_explanations: bool = False,
        extent=[-12, 16, 37.5, 55.4],
        fig_path: str = ".figs/attributions",
        return_output: bool = False,
    ):
        """
        Compute per-event centroid/peak displacement across noise levels.

        Returns
        -------
        result : dict with keys:
            "noises"                : np.ndarray, shape (num_noise_levels,)
            "centroid_disp_mc"      : np.ndarray, shape (num_noise_levels, num_monte_carlo)
                                      Per-MC centroid displacement in pixels
            "peak_disp_mc"          : np.ndarray, shape (num_noise_levels, num_monte_carlo)
                                      Per-MC peak displacement in pixels
            "rmse_mc"               : np.ndarray, shape (num_noise_levels, num_monte_carlo)
                                      Per-MC relative RMSE of the perturbed prediction
            "event_id"              : identifier of the event (batch_idx or runtime)
        
        None if clean gradient computation fails (event is skipped).
        """
        output_idx = input.outputs.feature_names_to_idx[cfg_xai["explain"]["output"]]

        # --- Clean forward pass and reference RMSE ---
        output = predict_step(
            self.model, input, batch_idx, checkpoint, cfg_model, cfg_dataset,
            cfg_xai, dataset_info, infer_ds, list_run_hour, use_old_weights,
            compute_grads=False)
        
        
        if output is None:
            return None

        output_gt = input.outputs.tensor[0, -1, ..., output_idx]
        err_ref = torch.sqrt(
            torch.mean((output_gt - output.tensor[0, -1, ..., output_idx]) ** 2)
        )

        # --- Clean gradient ---
        clean_gradient, _ = self.base_explainer.compute_explanations(
            input=input, batch_idx=batch_idx, checkpoint=checkpoint,
            cfg_model=cfg_model, cfg_dataset=cfg_dataset, cfg_xai=cfg_xai,
            dataset_info=dataset_info, infer_ds=infer_ds,
            list_run_hour=list_run_hour, use_old_weights=use_old_weights,
            target=target, plot_explanations=False, extent=extent,
            fig_path=fig_path, return_output=True,
        )
        if clean_gradient is None:
            return None

        clean_grad_2d = clean_gradient[0][0, 0, ..., 0].detach().cpu().numpy()
        clean_grad_2d = normalize_sum(clean_grad_2d)
        cx_clean_centroid, cy_clean_centroid = self._compute_centroid(clean_grad_2d)
        cx_clean_peak, cy_clean_peak = self._compute_peak(clean_grad_2d)

        # --- Setup perturbation source (input vs. forcing) ---
        if cfg_xai["explain"]["forcing_input"]:
            tensor_container = input.forcing
            input_idx = input.forcing.feature_names_to_idx[cfg_xai["explain"]["input"]]
        else:
            tensor_container = input.inputs
            input_idx = input.inputs.feature_names_to_idx[cfg_xai["explain"]["input"]]

        backup_tensor = tensor_container.tensor.clone()
        input_clean = backup_tensor[..., input_idx].clone()
        vals = input_clean
        val_range = float(vals.max() - vals.min())

        # --- Preallocate per-event result arrays ---
        noises = np.zeros(self.num_noise_levels)
        centroid_disp_mc = np.full(
            (self.num_noise_levels, self.num_monte_carlo), np.nan
        )
        peak_disp_mc = np.full(
            (self.num_noise_levels, self.num_monte_carlo), np.nan
        )
        rmse_mc = np.full(
            (self.num_noise_levels, self.num_monte_carlo), np.nan
        )

        # --- Loop over noise levels ---
        for noise_idx in range(self.num_noise_levels):
            sigma_frac = (noise_idx + 1) * self.sigma_step
            sigma = sigma_frac * val_range
            noises[noise_idx] = sigma_frac

            for mc_idx in range(self.num_monte_carlo):
                # Inject noise on the target channel only
                input_perturbed = input_clean + torch.randn_like(input_clean) * sigma
                new_tensor = backup_tensor.clone()
                new_tensor[..., input_idx] = input_perturbed

                if cfg_xai["explain"]["forcing_input"]:
                    input.forcing.tensor = new_tensor
                else:
                    input.inputs.tensor = new_tensor

                gradient, perturbed_output = self.base_explainer.compute_explanations(
                    input=input, batch_idx=batch_idx, checkpoint=checkpoint,
                    cfg_model=cfg_model, cfg_dataset=cfg_dataset, cfg_xai=cfg_xai,
                    dataset_info=dataset_info, infer_ds=infer_ds,
                    list_run_hour=list_run_hour, use_old_weights=use_old_weights,
                    target=target, plot_explanations=False, extent=extent,
                    fig_path=fig_path, return_output=True,
                )

                if gradient is None or perturbed_output is None:
                    continue  # leaves NaN in this slot

                # --- Centroid and peak displacement ---
                grad_2d = gradient[0][0, 0, ..., 0].detach().cpu().numpy()
                grad_2d = normalize_sum(grad_2d)
                cx_pert_c, cy_pert_c = self._compute_centroid(grad_2d)
                cx_pert_p, cy_pert_p = self._compute_peak(grad_2d)

                centroid_disp_mc[noise_idx, mc_idx] = np.sqrt(
                    (cx_pert_c - cx_clean_centroid) ** 2
                    + (cy_pert_c - cy_clean_centroid) ** 2
                )
                peak_disp_mc[noise_idx, mc_idx] = np.sqrt(
                    (cx_pert_p - cx_clean_peak) ** 2
                    + (cy_pert_p - cy_clean_peak) ** 2
                )

                # --- Relative prediction error ---
                out_perturbed = perturbed_output.tensor[0,-1, ..., output_idx]
                err = torch.sqrt(torch.mean((output_gt - out_perturbed) ** 2)) / err_ref
                rmse_mc[noise_idx, mc_idx] = float(err.detach().cpu())

                self.model.zero_grad(set_to_none=True)

        # --- Restore original tensor state (hygiene for caller) ---
        if cfg_xai["explain"]["forcing_input"]:
            input.forcing.tensor = backup_tensor
        else:
            input.inputs.tensor = backup_tensor

        return {
            "noises": noises,
            "centroid_disp_mc": centroid_disp_mc,
            "peak_disp_mc": peak_disp_mc,
            "rmse_mc": rmse_mc,
            "event_id": batch_idx,
        }

class CentroidSmoothGrad(XGrad):
    """
    Measures geometric displacement of gradient attribution maps under 
    input perturbations of increasing magnitude. Returns per-event results
    for downstream aggregation across multiple events/dates.
    """

    def __init__(
        self,
        model,
        base_explainer: str = "BaseGrad",sg_std_perturbations=0.1,sg_num_perturbations=5,
        sigma_step: float = 0.1,
        num_noise_levels: int = 10,
        num_monte_carlo: int = 20,
        saliency_threshold: float | None = None,
    ):
        """
        Parameters
        ----------
        model : the forecasting model
        base_explainer : "BaseGrad" or "InputxGrad"
        sigma_step : increment of normalized noise level per step 
                     (e.g. 0.1 → σ ∈ {0.1, 0.2, ..., num_noise_levels * 0.1})
        num_noise_levels : number of σ values to evaluate
        num_monte_carlo : number of MC perturbation samples per σ level
        saliency_threshold : if None, uses full |gradient| (recommended for 
                             rigor). If a float in [0,1], applies threshold 
                             on normalized gradient before computing centroid.
        """
        super().__init__(model)
        if base_explainer == "InputxGrad":
            self.base_explainer = InputxGrad(model)
        else:
            self.base_explainer = BaseGrad(model)

        self.explainer = SmoothGrad(model,base_explainer,sg_std_perturbations,sg_num_perturbations)
        self.sigma_step = sigma_step
        self.num_noise_levels = num_noise_levels
        self.num_monte_carlo = num_monte_carlo
        self.saliency_threshold = saliency_threshold

    # ───────────────────────────────────────────────────────────────────
    # Helpers
    # ───────────────────────────────────────────────────────────────────

    def _compute_centroid(self, grad_2d: np.ndarray) -> tuple[float, float]:
        """
        Compute the spatial center of mass of |grad_2d|.
        If self.saliency_threshold is set, applies percentile thresholding first.
        """
        abs_grad = np.abs(grad_2d)

        if self.saliency_threshold is not None:
            # Use percentile to be scale-invariant
            abs_thres = abs_grad/np.max(abs_grad)
            abs_grad = np.where(abs_thres >= 0.1, abs_grad, 0.0)

        # Fallback: if everything is zero (degenerate), return domain center
        if abs_grad.sum() == 0:
            H, W = abs_grad.shape
            return float(H) / 2, float(W) / 2

        cy, cx = center_of_mass(abs_grad)
        return float(cx), float(cy)

    def _compute_peak(self, grad_2d: np.ndarray) -> tuple[float, float]:
        """
        Spatial coordinate of the maximum of |grad_2d|.
        Alternative displacement measure, robust to diffuse background mass.
        """
        abs_grad = np.abs(grad_2d)
        idx_flat = int(np.argmax(abs_grad))
        cy, cx = np.unravel_index(idx_flat, abs_grad.shape)
        return float(cx), float(cy)

    # ───────────────────────────────────────────────────────────────────
    # Main API
    # ───────────────────────────────────────────────────────────────────

    def compute_explanations(
        self,
        input,
        batch_idx,
        checkpoint,
        cfg_model,
        cfg_dataset,
        cfg_xai,
        dataset_info,
        infer_ds,
        list_run_hour,
        use_old_weights,
        target,
        plot_explanations: bool = False,
        extent=[-12, 16, 37.5, 55.4],
        fig_path: str = ".figs/attributions",
        return_output: bool = False,
    ):
        """
        Compute per-event centroid/peak displacement across noise levels.

        Returns
        -------
        result : dict with keys:
            "noises"                : np.ndarray, shape (num_noise_levels,)
            "centroid_disp_mc"      : np.ndarray, shape (num_noise_levels, num_monte_carlo)
                                      Per-MC centroid displacement in pixels
            "peak_disp_mc"          : np.ndarray, shape (num_noise_levels, num_monte_carlo)
                                      Per-MC peak displacement in pixels
            "rmse_mc"               : np.ndarray, shape (num_noise_levels, num_monte_carlo)
                                      Per-MC relative RMSE of the perturbed prediction
            "event_id"              : identifier of the event (batch_idx or runtime)
        
        None if clean gradient computation fails (event is skipped).
        """
        output_idx = input.outputs.feature_names_to_idx[cfg_xai["explain"]["output"]]

        # --- Clean forward pass and reference RMSE ---
        output = predict_step(
            self.model, input, batch_idx, checkpoint, cfg_model, cfg_dataset,
            cfg_xai, dataset_info, infer_ds, list_run_hour, use_old_weights,
            compute_grads=False)
        
        
        if output is None:
            return None

        output_gt = input.outputs.tensor[0, -1, ..., output_idx]
        err_ref = torch.sqrt(
            torch.mean((output_gt - output.tensor[0, -1, ..., output_idx]) ** 2)
        )

        # --- Clean gradient ---
        clean_gradient, _ = self.base_explainer.compute_explanations(
            input=input, batch_idx=batch_idx, checkpoint=checkpoint,
            cfg_model=cfg_model, cfg_dataset=cfg_dataset, cfg_xai=cfg_xai,
            dataset_info=dataset_info, infer_ds=infer_ds,
            list_run_hour=list_run_hour, use_old_weights=use_old_weights,
            target=target, plot_explanations=False, extent=extent,
            fig_path=fig_path, return_output=True,
        )
        if clean_gradient is None:
            return None

        clean_grad_2d = clean_gradient[0][0, 0, ..., 0].detach().cpu().numpy()
        clean_grad_2d = normalize_sum(clean_grad_2d)
        cx_clean_centroid, cy_clean_centroid = self._compute_centroid(clean_grad_2d)
        cx_clean_peak, cy_clean_peak = self._compute_peak(clean_grad_2d)

        # --- Setup perturbation source (input vs. forcing) ---
        if cfg_xai["explain"]["forcing_input"]:
            tensor_container = input.forcing
            input_idx = input.forcing.feature_names_to_idx[cfg_xai["explain"]["input"]]
        else:
            tensor_container = input.inputs
            input_idx = input.inputs.feature_names_to_idx[cfg_xai["explain"]["input"]]

        backup_tensor = tensor_container.tensor.clone()
        input_clean = backup_tensor[..., input_idx].clone()
        vals = input_clean
        val_range = float(vals.max() - vals.min())

        # --- Preallocate per-event result arrays ---
        noises = np.zeros(self.num_noise_levels)
        centroid_disp_mc = np.full(
            (self.num_noise_levels, self.num_monte_carlo), np.nan
        )
        peak_disp_mc = np.full(
            (self.num_noise_levels, self.num_monte_carlo), np.nan
        )
        rmse_mc = np.full(
            (self.num_noise_levels, self.num_monte_carlo), np.nan
        )

        # --- Loop over noise levels ---
        for noise_idx in range(self.num_noise_levels):
            sigma_frac = (noise_idx + 1) * self.sigma_step
            sigma = sigma_frac * val_range
            noises[noise_idx] = sigma_frac

            for mc_idx in range(self.num_monte_carlo):
                # Inject noise on the target channel only
                input_perturbed = input_clean + torch.randn_like(input_clean) * sigma
                new_tensor = backup_tensor.clone()
                new_tensor[..., input_idx] = input_perturbed

                if cfg_xai["explain"]["forcing_input"]:
                    input.forcing.tensor = new_tensor
                else:
                    input.inputs.tensor = new_tensor

                gradient, perturbed_output = self.explainer.compute_explanations(
                    input=input, batch_idx=batch_idx, checkpoint=checkpoint,
                    cfg_model=cfg_model, cfg_dataset=cfg_dataset, cfg_xai=cfg_xai,
                    dataset_info=dataset_info, infer_ds=infer_ds,
                    list_run_hour=list_run_hour, use_old_weights=use_old_weights,
                    target=target, plot_explanations=False, extent=extent,
                    fig_path=fig_path, return_output=True,
                )

                if gradient is None or perturbed_output is None:
                    continue  # leaves NaN in this slot

                # --- Centroid and peak displacement ---
                grad_2d = gradient[0][0, 0, ..., 0].detach().cpu().numpy()
                grad_2d = normalize_sum(grad_2d)
                cx_pert_c, cy_pert_c = self._compute_centroid(grad_2d)
                cx_pert_p, cy_pert_p = self._compute_peak(grad_2d)

                centroid_disp_mc[noise_idx, mc_idx] = np.sqrt(
                    (cx_pert_c - cx_clean_centroid) ** 2
                    + (cy_pert_c - cy_clean_centroid) ** 2
                )
                peak_disp_mc[noise_idx, mc_idx] = np.sqrt(
                    (cx_pert_p - cx_clean_peak) ** 2
                    + (cy_pert_p - cy_clean_peak) ** 2
                )

                # --- Relative prediction error ---
                out_perturbed = perturbed_output.tensor[0,-1, ..., output_idx]
                err = torch.sqrt(torch.mean((output_gt - out_perturbed) ** 2)) / err_ref
                rmse_mc[noise_idx, mc_idx] = float(err.detach().cpu())

                self.model.zero_grad(set_to_none=True)

        # --- Restore original tensor state (hygiene for caller) ---
        if cfg_xai["explain"]["forcing_input"]:
            input.forcing.tensor = backup_tensor
        else:
            input.inputs.tensor = backup_tensor

        return {
            "noises": noises,
            "centroid_disp_mc": centroid_disp_mc,
            "peak_disp_mc": peak_disp_mc,
            "rmse_mc": rmse_mc,
            "event_id": batch_idx,
        }

class CentroidWassersteinGrad(XGrad):
    """
    Measures geometric displacement of gradient attribution maps under 
    input perturbations of increasing magnitude. Returns per-event results
    for downstream aggregation across multiple events/dates.
    """

    def __init__(
        self,
        model,
        base_explainer: str = "BaseGrad",sg_std_perturbations=0.1,sg_num_perturbations=5,reg=0.001,
        sigma_step: float = 0.1,
        num_noise_levels: int = 10,
        num_monte_carlo: int = 20,
        saliency_threshold: float | None = None,
    ):
        """
        Parameters
        ----------
        model : the forecasting model
        base_explainer : "BaseGrad" or "InputxGrad"
        sigma_step : increment of normalized noise level per step 
                     (e.g. 0.1 → σ ∈ {0.1, 0.2, ..., num_noise_levels * 0.1})
        num_noise_levels : number of σ values to evaluate
        num_monte_carlo : number of MC perturbation samples per σ level
        saliency_threshold : if None, uses full |gradient| (recommended for 
                             rigor). If a float in [0,1], applies threshold 
                             on normalized gradient before computing centroid.
        """
        super().__init__(model)
        if base_explainer == "InputxGrad":
            self.base_explainer = InputxGrad(model)
        else:
            self.base_explainer = BaseGrad(model)

        self.explainer = WassersteinGrad(model,base_explainer,sg_std_perturbations,sg_num_perturbations,reg)
        self.sigma_step = sigma_step
        self.num_noise_levels = num_noise_levels
        self.num_monte_carlo = num_monte_carlo
        self.saliency_threshold = saliency_threshold

    # ───────────────────────────────────────────────────────────────────
    # Helpers
    # ───────────────────────────────────────────────────────────────────

    def _compute_centroid(self, grad_2d: np.ndarray) -> tuple[float, float]:
        """
        Compute the spatial center of mass of |grad_2d|.
        If self.saliency_threshold is set, applies percentile thresholding first.
        """
        abs_grad = np.abs(grad_2d)

        if self.saliency_threshold is not None:
            # Use percentile to be scale-invariant

            abs_grad = np.where(abs_grad >= 0.1, abs_grad, 0.0)

        # Fallback: if everything is zero (degenerate), return domain center
        if abs_grad.sum() == 0:
            H, W = abs_grad.shape
            return float(H) / 2, float(W) / 2

        cy, cx = center_of_mass(abs_grad)
        return float(cx), float(cy)

    def _compute_peak(self, grad_2d: np.ndarray) -> tuple[float, float]:
        """
        Spatial coordinate of the maximum of |grad_2d|.
        Alternative displacement measure, robust to diffuse background mass.
        """
        abs_grad = np.abs(grad_2d)
        idx_flat = int(np.argmax(abs_grad))
        cy, cx = np.unravel_index(idx_flat, abs_grad.shape)
        return float(cx), float(cy)

    # ───────────────────────────────────────────────────────────────────
    # Main API
    # ───────────────────────────────────────────────────────────────────

    def compute_explanations(
        self,
        input,
        batch_idx,
        checkpoint,
        cfg_model,
        cfg_dataset,
        cfg_xai,
        dataset_info,
        infer_ds,
        list_run_hour,
        use_old_weights,
        target,
        plot_explanations: bool = False,
        extent=[-12, 16, 37.5, 55.4],
        fig_path: str = ".figs/attributions",
        return_output: bool = False,
    ):
        """
        Compute per-event centroid/peak displacement across noise levels.

        Returns
        -------
        result : dict with keys:
            "noises"                : np.ndarray, shape (num_noise_levels,)
            "centroid_disp_mc"      : np.ndarray, shape (num_noise_levels, num_monte_carlo)
                                      Per-MC centroid displacement in pixels
            "peak_disp_mc"          : np.ndarray, shape (num_noise_levels, num_monte_carlo)
                                      Per-MC peak displacement in pixels
            "rmse_mc"               : np.ndarray, shape (num_noise_levels, num_monte_carlo)
                                      Per-MC relative RMSE of the perturbed prediction
            "event_id"              : identifier of the event (batch_idx or runtime)
        
        None if clean gradient computation fails (event is skipped).
        """
        output_idx = input.outputs.feature_names_to_idx[cfg_xai["explain"]["output"]]

        # --- Clean forward pass and reference RMSE ---
        output = predict_step(
            self.model, input, batch_idx, checkpoint, cfg_model, cfg_dataset,
            cfg_xai, dataset_info, infer_ds, list_run_hour, use_old_weights,
            compute_grads=False)
        
        
        if output is None:
            return None

        output_gt = input.outputs.tensor[0, -1, ..., output_idx]
        err_ref = torch.sqrt(
            torch.mean((output_gt - output.tensor[0, -1, ..., output_idx]) ** 2)
        )

        # --- Clean gradient ---
        clean_gradient, _ = self.base_explainer.compute_explanations(
            input=input, batch_idx=batch_idx, checkpoint=checkpoint,
            cfg_model=cfg_model, cfg_dataset=cfg_dataset, cfg_xai=cfg_xai,
            dataset_info=dataset_info, infer_ds=infer_ds,
            list_run_hour=list_run_hour, use_old_weights=use_old_weights,
            target=target, plot_explanations=False, extent=extent,
            fig_path=fig_path, return_output=True,
        )
        if clean_gradient is None:
            return None

        clean_grad_2d = clean_gradient[0][0, 0, ..., 0].detach().cpu().numpy()
        clean_grad_2d = normalize_sum_wasser(clean_grad_2d)
        cx_clean_centroid, cy_clean_centroid = self._compute_centroid(clean_grad_2d)
        cx_clean_peak, cy_clean_peak = self._compute_peak(clean_grad_2d)

        # --- Setup perturbation source (input vs. forcing) ---
        if cfg_xai["explain"]["forcing_input"]:
            tensor_container = input.forcing
            input_idx = input.forcing.feature_names_to_idx[cfg_xai["explain"]["input"]]
        else:
            tensor_container = input.inputs
            input_idx = input.inputs.feature_names_to_idx[cfg_xai["explain"]["input"]]

        backup_tensor = tensor_container.tensor.clone()
        input_clean = backup_tensor[..., input_idx].clone()
        vals = input_clean
        val_range = float(vals.max() - vals.min())

        # --- Preallocate per-event result arrays ---
        noises = np.zeros(self.num_noise_levels)
        centroid_disp_mc = np.full(
            (self.num_noise_levels, self.num_monte_carlo), np.nan
        )
        peak_disp_mc = np.full(
            (self.num_noise_levels, self.num_monte_carlo), np.nan
        )
        rmse_mc = np.full(
            (self.num_noise_levels, self.num_monte_carlo), np.nan
        )

        # --- Loop over noise levels ---
        for noise_idx in range(self.num_noise_levels):
            sigma_frac = (noise_idx + 1) * self.sigma_step
            sigma = sigma_frac * val_range
            noises[noise_idx] = sigma_frac

            for mc_idx in range(self.num_monte_carlo):
                # Inject noise on the target channel only
                input_perturbed = input_clean + torch.randn_like(input_clean) * sigma
                new_tensor = backup_tensor.clone()
                new_tensor[..., input_idx] = input_perturbed

                if cfg_xai["explain"]["forcing_input"]:
                    input.forcing.tensor = new_tensor
                else:
                    input.inputs.tensor = new_tensor

                gradient, perturbed_output = self.explainer.compute_explanations(
                    input=input, batch_idx=batch_idx, checkpoint=checkpoint,
                    cfg_model=cfg_model, cfg_dataset=cfg_dataset, cfg_xai=cfg_xai,
                    dataset_info=dataset_info, infer_ds=infer_ds,
                    list_run_hour=list_run_hour, use_old_weights=use_old_weights,
                    target=target, plot_explanations=False, extent=extent,
                    fig_path=fig_path, return_output=True,
                )

                if gradient is None or perturbed_output is None:
                    continue  # leaves NaN in this slot

                # --- Centroid and peak displacement ---
                grad_2d = gradient[0].reshape(512,640).detach().cpu().numpy()
                grad_2d = normalize_sum_wasser(grad_2d)
                cx_pert_c, cy_pert_c = self._compute_centroid(grad_2d)
                cx_pert_p, cy_pert_p = self._compute_peak(grad_2d)

                centroid_disp_mc[noise_idx, mc_idx] = np.sqrt(
                    (cx_pert_c - cx_clean_centroid) ** 2
                    + (cy_pert_c - cy_clean_centroid) ** 2
                )
                peak_disp_mc[noise_idx, mc_idx] = np.sqrt(
                    (cx_pert_p - cx_clean_peak) ** 2
                    + (cy_pert_p - cy_clean_peak) ** 2
                )

                # --- Relative prediction error ---
                out_perturbed = perturbed_output.tensor[0,-1, ..., output_idx]
                err = torch.sqrt(torch.mean((output_gt - out_perturbed) ** 2)) / err_ref
                rmse_mc[noise_idx, mc_idx] = float(err.detach().cpu())

                self.model.zero_grad(set_to_none=True)

        # --- Restore original tensor state (hygiene for caller) ---
        if cfg_xai["explain"]["forcing_input"]:
            input.forcing.tensor = backup_tensor
        else:
            input.inputs.tensor = backup_tensor

        return {
            "noises": noises,
            "centroid_disp_mc": centroid_disp_mc,
            "peak_disp_mc": peak_disp_mc,
            "rmse_mc": rmse_mc,
            "event_id": batch_idx,
        }

# ── Global style (matches your map figure) ───────────────────────────────────
# plt.rcParams.update({
#     "font.family":          "serif",
#     "font.serif":           ["Times New Roman", "DejaVu Serif"],
#     "axes.titlesize":       11,
#     "axes.labelsize":       10,
#     "xtick.labelsize":      9,
#     "ytick.labelsize":      9,
#     "axes.linewidth":       0.7,
#     "xtick.major.width":    0.6,
#     "ytick.major.width":    0.6,
#     "xtick.minor.width":    0.4,
#     "ytick.minor.width":    0.4,
#     "xtick.direction":      "in",
#     "ytick.direction":      "in",
#     "xtick.minor.visible":  True,
#     "ytick.minor.visible":  True,
#     "axes.spines.top":      False,
#     "axes.spines.right":    False,
#     "legend.frameon":       True,
#     "legend.framealpha":    0.9,
#     "legend.edgecolor":     "#cccccc",
#     "legend.fontsize":      9,
#     "figure.dpi":           300,
#     "savefig.dpi":          300,
#     "savefig.bbox":         "tight",
#     "savefig.pad_inches":   0.05,
# })

# # ── Palette (colourblind-safe) ────────────────────────────────────────────────
# BLUE   = "#2166ac"   # mean curve
# RED = "#c2221c"
# FILL   = "#92c5de"   # ±1 std band
# DOT    = "#d6604d"   # individual run scatter (optional)


# def normalize_sum(a):
#     a = torch.abs(a)  # ensure non-negative
#     return a / (a.sum() + 1e-8)

# def plot_centroid_displacement(
#     noises,
#     centroid_means,
#     centroid_stds,
#     rmse_means,
#     rmse_stds,
#     n_samples:   int = 100,
#     show_scatter: bool = False,
#     centroid_all_runs = None,   # list-of-lists if show_scatter=True
# ):
#     """
#     Publication-ready plot of centroid displacement vs. perturbation noise σ.

#     Parameters
#     ----------
#     noises          : array-like  — noise std values on x-axis
#     centroid_means  : array-like  — mean displacement per noise level
#     centroid_stds   : array-like  — std  displacement per noise level
#     output_path     : str         — file to save (pdf recommended for NeurIPS)
#     input_name      : str         — variable name for subtitle
#     output_name     : str         — variable name for subtitle
#     n_samples       : int         — number of Monte-Carlo samples (for caption)
#     show_scatter    : bool        — overlay individual-run dots
#     centroid_all_runs : list      — raw per-run displacements (needed if show_scatter)
#     """
#     noises  = np.asarray(noises)
#     means   = np.asarray(centroid_means)
#     stds    = np.asarray(centroid_stds)
#     mean_rmse = np.asarray(rmse_means)
#     stds_rmse = np.asarray(rmse_stds)

#     fig, ax1 = plt.subplots(figsize=(5.5, 3.8))
#     ax2 = ax1.twinx()
#     ax2.spines["right"].set_visible(True)
#     # ax2.spines["right"].set_linewidth(1.0)
#     # ax2.spines["right"].set_color(RED)  # optional: match RMSE axis
#     # --- centroid (left axis) ---
#     ax1.fill_between(noises, means - stds, means + stds,
#                     color=BLUE, alpha=0.2)
#     ax1.plot(noises, means, color=BLUE, linewidth=1.8,
#             label="Centroid displacement")

#     # --- RMSE (right axis) ---
#     ax2.fill_between(noises, mean_rmse - stds_rmse, mean_rmse + stds_rmse,
#                     color=RED, alpha=0.25)
#     ax2.plot(noises, mean_rmse, color=RED, linewidth=1.8,
#             label="RMSE")

#     # Labels
#     ax1.set_xlabel(r"Noise $\sigma$")
#     ax1.set_ylabel("Centroid displacement (pixels)", color=BLUE)
#     ax2.set_ylabel("Relative Prediction Error", color=RED)

#     # Legends (combine both)
#     lines1, labels1 = ax1.get_legend_handles_labels()
#     lines2, labels2 = ax2.get_legend_handles_labels()
#     ax1.legend(lines1 + lines2, labels1 + labels2, loc="upper left")
#     ax1.grid(True, axis='y', linestyle=':', linewidth=0.5, alpha=0.6)
#     # # ── Subtle grid (y only, very light) ─────────────────────────────────────
#     # ax.yaxis.grid(True, linestyle=":", linewidth=0.4,
#     #               color="#cccccc", zorder=0)
#     # ax.set_axisbelow(True)

#     plt.tight_layout()
#     # plt.show()
#     plt.savefig("centroid_displacement.pdf", bbox_inches="tight")
#     plt.close(fig)

# class Centroid(XGrad):


#     def __init__(self,model,base_explainer="BaseGrad",std_perturbations=0.01,num_perturbations=50):
#         super().__init__(model)
#         if base_explainer == "InputxGrad":
#             self.base_explainer = InputxGrad(model)
#         else:
#             self.base_explainer = BaseGrad(model)
        
#         self.std_perturbation = std_perturbations
#         self.num_perturbations = num_perturbations

#     def compute_explanations(self,
#                              input,batch_idx,checkpoint,cfg_model,cfg_dataset,
#                              cfg_xai,dataset_info,infer_ds,list_run_hour,use_old_weights,
#                              target,plot_explanations = True,
#                              extent=[-12, 16, 37.5, 55.4],
#                              fig_path=".figs/attributions",return_output = False):
#         #print(input.forcing)
        
#         output_idx = input.outputs.feature_names_to_idx[cfg_xai["explain"]["output"]]

         
#         grads = {i:[] for i in range(input.num_pred_steps)}
     

#         output = predict_step(self.model,input,batch_idx,checkpoint,cfg_model,cfg_dataset,cfg_xai,dataset_info,infer_ds,list_run_hour,use_old_weights,compute_grads=False)
        
#         output_gt = input.outputs.tensor[0,0,...,output_idx]

#         err_ref =  torch.sqrt(torch.mean((output_gt - output.tensor[0,0,...,output_idx]) ** 2))
#         if output is None:
#             if return_output:
#                 return None,None
#             return

#         clean_gradient,_ = self.base_explainer.compute_explanations(input=input,batch_idx=batch_idx,checkpoint=checkpoint,cfg_model=cfg_model,cfg_dataset=cfg_dataset,cfg_xai=cfg_xai,dataset_info=dataset_info,infer_ds=infer_ds,list_run_hour=list_run_hour,use_old_weights=use_old_weights,target=target,plot_explanations=False,extent=extent,fig_path=fig_path,return_output=True)

#         if clean_gradient is None:
#                     if return_output:
#                         return None,None
#                     return
                
#         # for i in range(len(clean_gradient)):
#         #             clean_gradient[i].append(clean_gradient[i])
        
#         if cfg_xai["explain"]["forcing_input"]:
#             backup_tensor = input.forcing.tensor
#             input_idx = input.forcing.feature_names_to_idx[cfg_xai["explain"]["input"]]
#             input_clean = input.forcing.tensor[...,input_idx]
#             vals = input.forcing.tensor[..., input_idx]
#             std_perturbation = self.std_perturbation * (vals.max() - vals.min())   
            
#             for perturbation_idx in range(self.num_perturbations): 
                
#                 # input.inputs.tensor =input_clean+ torch.randn_like(input.inputs.tensor)*std_perturbation
#                 input_perturbed = input_clean.clone().detach() + torch.randn_like(input_clean) * std_perturbation
                
#                 input.forcing.tensor = input.forcing.tensor.clone()
#                 input.forcing.tensor[..., input_idx] = input_perturbed
#                 gradient,_ = self.base_explainer.compute_explanations(input=input,batch_idx=batch_idx,checkpoint=checkpoint,
#                                                                       cfg_model=cfg_model,cfg_dataset=cfg_dataset,cfg_xai=cfg_xai,dataset_info=dataset_info,infer_ds=infer_ds,
#                                                                       list_run_hour=list_run_hour,use_old_weights=use_old_weights,target=target,plot_explanations=False,extent=extent,fig_path=fig_path,return_output=True)

#                 if gradient is None:
#                     if return_output:
#                         return None,None
#                     return
                
#                 for i in range(len(gradient)):
#                     grads[i].append(gradient[i])
                
#                 self.model.zero_grad(set_to_none=True)

                
#         else:
#             backup_tensor = input.inputs.tensor

#             input_idx = input.inputs.feature_names_to_idx[cfg_xai["explain"]["input"]]
#             input_clean = input.inputs.tensor[...,input_idx]
#             vals = input.inputs.tensor[..., input_idx] 
#             stats = dataset_info.stats
          
#             std_perturbation = self.std_perturbation * (vals.max() - vals.min())
#     # Before the loop, initialize a list to store selected perturbations
#             centroid_means = []
#             centroid_stds = []
#             rmse_means = []
#             rmse_stds = []
#             noises = []
#             for perturbation_idx in range(self.num_perturbations):
#                 noise = (perturbation_idx+1)*self.std_perturbation
#                 std_perturbation = noise* (vals.max() - vals.min())
#                 centroid_level = []
#                 rmse = []
#                 for _ in range(20):
#                     input_perturbed = input_clean.clone().detach() + torch.randn_like(input_clean) * std_perturbation
                    
        

                
#                     input.inputs.tensor = backup_tensor.clone()
#                     input.inputs.tensor[..., input_idx] = input_perturbed
#                     gradient,perturbed_output = self.base_explainer.compute_explanations(input=input,batch_idx=batch_idx,checkpoint=checkpoint,cfg_model=cfg_model,cfg_dataset=cfg_dataset,
#                                                                         cfg_xai=cfg_xai,dataset_info=dataset_info,infer_ds=infer_ds,list_run_hour=list_run_hour,
#                                                                         use_old_weights=use_old_weights,target=target,plot_explanations=False,extent=extent,
#                                                                         fig_path=fig_path,return_output=True)

#                     if gradient is None:
#                         if return_output:
#                             return None,None
#                         return
                    
              
                
#                     if gradient is not None:
#                         # Get the 2D gradient map (assuming shape is [H, W])
#                         grad_2d = normalize_expl(gradient[0][0,0,...,0].cpu().numpy())
#                         clean_grad_2d = normalize_expl(clean_gradient[0][0,0,...,0].cpu().numpy())
#                     # Inside your loop: only keep the strongest gradients (the actual weather fronts)
#                         threshold_val = np.percentile(grad_2d, 98) # Keep top 10%
#                         salient_grad = np.where(grad_2d > 0.1, grad_2d, 0)
#                         clean_threshold_val = np.percentile(clean_grad_2d, 98)
#                         salient_clean_grad = np.where(clean_grad_2d > 0.1, clean_grad_2d, 0)

#                         # Now calculate Center of Mass on the salient features
#                         cx_pert, cy_pert = center_of_mass(salient_grad)
#                         cx_clean, cy_clean = center_of_mass(salient_clean_grad)
                        
#                         # Calculate Euclidean displacement in pixels
#                         displacement = np.sqrt((cx_pert - cx_clean)**2 + (cy_pert - cy_clean)**2)
#                         centroid_level.append(displacement)

#                         # out_clean = output.tensor[0,0,...,output_idx]
#                         out_clean = output_gt
#                         out_perturbed = perturbed_output.tensor[0,0,...,output_idx]
#                         error = out_clean - out_perturbed
#                         err = torch.sqrt(torch.mean(error ** 2))/err_ref
#                         rmse.append(err.detach().cpu().numpy())

#                 centroid_means.append(np.mean(centroid_level))
#                 centroid_stds.append(np.std(centroid_level))
#                 rmse_means.append(np.mean(rmse))
#                 rmse_stds.append(np.std(rmse))
#                 noises.append(noise)            
#                 print(f"displacement for noise {noise}:mean:{np.mean(centroid_level)},std:{np.std(centroid_level)}")
                
            
#         plot_centroid_displacement(
#         noises         = noises,
#         centroid_means = centroid_means,
#         centroid_stds  = centroid_stds,
#         rmse_means = rmse_means,
#         rmse_stds = rmse_stds,
  
#         n_samples      = 100
#     )
  
                



# class CentroidWasser(XGrad):


#     def __init__(self,model,base_explainer="BaseGrad",std_perturbations=0.01,num_perturbations=50):
#         super().__init__(model)
#         if base_explainer == "InputxGrad":
#             self.base_explainer = InputxGrad(model)
#         else:
#             self.base_explainer = BaseGrad(model)
        
#         self.std_perturbation = std_perturbations
#         self.num_perturbations = num_perturbations

#     def compute_explanations(self,
#                              input,batch_idx,checkpoint,cfg_model,cfg_dataset,
#                              cfg_xai,dataset_info,infer_ds,list_run_hour,use_old_weights,
#                              target,plot_explanations = True,
#                              extent=[-12, 16, 37.5, 55.4],
#                              fig_path=".figs/attributions",return_output = False):
#         #print(input.forcing)
        
#         output_idx = input.outputs.feature_names_to_idx[cfg_xai["explain"]["output"]]

         
#         grads = {i:[] for i in range(input.num_pred_steps)}
     

#         output = predict_step(self.model,input,batch_idx,checkpoint,cfg_model,cfg_dataset,cfg_xai,dataset_info,infer_ds,list_run_hour,use_old_weights,compute_grads=False)
        
        
#         if output is None:
#             if return_output:
#                 return None,None
#             return

#         clean_gradient,_ = self.base_explainer.compute_explanations(input=input,batch_idx=batch_idx,checkpoint=checkpoint,cfg_model=cfg_model,cfg_dataset=cfg_dataset,cfg_xai=cfg_xai,dataset_info=dataset_info,infer_ds=infer_ds,list_run_hour=list_run_hour,use_old_weights=use_old_weights,target=target,plot_explanations=False,extent=extent,fig_path=fig_path,return_output=True)

#         if clean_gradient is None:
#                     if return_output:
#                         return None,None
#                     return
                
#         # for i in range(len(clean_gradient)):
#         #             clean_gradient[i].append(clean_gradient[i])
        
#         if cfg_xai["explain"]["forcing_input"]:
#             backup_tensor = input.forcing.tensor
#             input_idx = input.forcing.feature_names_to_idx[cfg_xai["explain"]["input"]]
#             input_clean = input.forcing.tensor[...,input_idx]
#             vals = input.forcing.tensor[..., input_idx]
#             std_perturbation = self.std_perturbation * (vals.max() - vals.min())   
            
#             for perturbation_idx in range(self.num_perturbations): 
                
#                 # input.inputs.tensor =input_clean+ torch.randn_like(input.inputs.tensor)*std_perturbation
#                 input_perturbed = input_clean.clone().detach() + torch.randn_like(input_clean) * std_perturbation
                
#                 input.forcing.tensor = input.forcing.tensor.clone()
#                 input.forcing.tensor[..., input_idx] = input_perturbed
#                 gradient,_ = self.base_explainer.compute_explanations(input=input,batch_idx=batch_idx,checkpoint=checkpoint,cfg_model=cfg_model,cfg_dataset=cfg_dataset,cfg_xai=cfg_xai,dataset_info=dataset_info,infer_ds=infer_ds,list_run_hour=list_run_hour,use_old_weights=use_old_weights,target=target,plot_explanations=False,extent=extent,fig_path=fig_path,return_output=True)

#                 if gradient is None:
#                     if return_output:
#                         return None,None
#                     return
                
#                 for i in range(len(gradient)):
#                     grads[i].append(gradient[i])
                
#                 self.model.zero_grad(set_to_none=True)

                
#         else:
#             backup_tensor = input.inputs.tensor

#             input_idx = input.inputs.feature_names_to_idx[cfg_xai["explain"]["input"]]
#             input_clean = input.inputs.tensor[...,input_idx]
#             vals = input.inputs.tensor[..., input_idx] 
#             stats = dataset_info.stats
          
#             std_perturbation = self.std_perturbation * (vals.max() - vals.min())
#     # Before the loop, initialize a list to store selected perturbations
#             centroid_means,centroid_means_wasser = [],[]
#             centroid_stds,centroid_stds_wasser = [], []
#             noises = []
#             for perturbation_idx in range(self.num_perturbations):
#                 noise = (perturbation_idx+1)*self.std_perturbation
#                 std_perturbation = noise* (vals.max() - vals.min())
#                 centroid_level = []
#                 centroid_level_wasser = []
#                 for _ in range(20):
#                     grads = []
#                     for k in range(self.num_perturbations):
#                         input_perturbed = input_clean.clone().detach() + torch.randn_like(input_clean) * std_perturbation
                        
            

                    
#                         input.inputs.tensor = backup_tensor.clone()
#                         input.inputs.tensor[..., input_idx] = input_perturbed
#                         gradient,_ = self.base_explainer.compute_explanations(input=input,batch_idx=batch_idx,checkpoint=checkpoint,cfg_model=cfg_model,cfg_dataset=cfg_dataset,
#                                                                             cfg_xai=cfg_xai,dataset_info=dataset_info,infer_ds=infer_ds,list_run_hour=list_run_hour,
#                                                                             use_old_weights=use_old_weights,target=target,plot_explanations=False,extent=extent,
#                                                                             fig_path=fig_path,return_output=True)

#                         if gradient is None:
#                             if return_output:
#                                 return None,None
#                             return
                        
                
#                         grad = normalize_sum(gradient[0][0,0,...,0])
#                         grads.append(grad)

#                     A_maps = torch.stack(grads, dim =0)
            
              

#                     barycentre = ot.bregman.convolutional_barycenter2d(A_maps, reg=0.001)
#                     barycentre = torch.tensor(barycentre/barycentre.max())
                                
#                     if gradient is not None:
#                         # Get the 2D gradient map (assuming shape is [H, W])
#                         # grad_2d = normalize_expl(gradient[0][0,0,...,0].cpu().numpy())
#                         clean_grad_2d = normalize_expl(clean_gradient[0][0,0,...,0].cpu().numpy())
#                         smooth_grad = torch.stack(grads).mean(dim=0)
#                         smooth_grad = normalize_expl(smooth_grad.cpu().numpy())
#                         # Inside your loop: only keep the strongest gradients (the actual weather fronts)
#                         # threshold_val = np.percentile(grad_2d, 98) # Keep top 10%
#                         # salient_grad = np.where(grad_2d > 0.1, grad_2d, 0)
#                         # clean_threshold_val = np.percentile(clean_grad_2d, 98)
#                         # salient_clean_grad = np.where(clean_grad_2d > 0.1, clean_grad_2d, 0)

#                         # Now calculate Center of Mass on the salient features
#                         cx_pert, cy_pert = center_of_mass(barycentre.detach().cpu().numpy())
#                         cx_clean, cy_clean = center_of_mass(clean_grad_2d.detach().cpu().numpy())
#                         cx_smooth,cy_smooth = center_of_mass(smooth_grad.detach().cpu().numpy())
                        
#                         # Calculate Euclidean displacement in pixels
#                         displacement_wasser = np.sqrt((cx_pert - cx_clean)**2 + (cy_pert - cy_clean)**2)
#                         displacement = np.sqrt((cx_smooth - cx_clean)**2 + (cy_smooth - cy_clean)**2)
#                         centroid_level_wasser.append(displacement_wasser)
#                         centroid_level.append(displacement)

#                 centroid_means.append(np.mean(centroid_level))
#                 centroid_stds.append(np.std(centroid_level))
#                 centroid_means_wasser.append(np.mean(centroid_level_wasser))
#                 centroid_stds_wasser.append(np.std(centroid_level_wasser))
#                 noises.append(noise) 
#                 print("-----smooth------")         
#                 print(f"displacement for noise {noise}:mean:{np.mean(centroid_level)},std:{np.std(centroid_level)}")
#                 print("------wasser-------")
#                 print(f"displacement for noise {noise}:mean:{np.mean(centroid_level_wasser)},std:{np.std(centroid_level_wasser)}")

            
#         plot_centroid_displacement(
#         noises         = noises,
#         centroid_means = centroid_means,
#         centroid_stds  = centroid_stds,
  
#         n_samples      = 20
#     )
        
#         plot_centroid_displacement(
#             noises         = noises,
#             centroid_means = centroid_means_wasser,
#             centroid_stds  = centroid_stds_wasser,
    
#             n_samples      = 20
#         )
                    
     