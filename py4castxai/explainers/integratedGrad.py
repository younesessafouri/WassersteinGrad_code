from py4castxai.explainers.XGrad import XGrad
import torch
from py4castxai.utils_test.infer_utils import predict_step

from py4castxai.explainers.inputxgrad import InputxGrad
from py4castxai.explainers.basegrad import BaseGrad


class IntegratedGrad(XGrad):
    """
    Integrated Gradients (Sundararajan et al., ICML 2017).
    
    Computes attribution by integrating the gradient of the output with respect
    to the input along a straight-line path from a baseline x' to the input x:
    
        IG_i(x) = (x_i - x'_i) * ∫_{α=0}^{1} ∂f(x' + α(x - x')) / ∂x_i dα
    
    The integral is approximated via left-endpoint Riemann sum with num_steps steps.
    
    Parameters
    ----------
    model : the forecasting model
    base_explainer : "BaseGrad" or "InputxGrad" — the underlying gradient extractor
    num_steps : number of integration steps (default 50, standard in literature)
    baseline : str — "zeros" (default), "mean", or "blur"
        - "zeros": x' = 0 (on the normalised input, corresponds to the per-channel mean)
        - "mean": x' = channel-wise mean of the current input (equivalent for z-scored inputs)
        - "blur": x' = Gaussian-blurred input with large kernel (smoother reference)
    """

    def __init__(
        self,
        model,
        base_explainer: str = "BaseGrad",
        num_steps: int = 20,
        baseline: str = "zeros",
    ):
        super().__init__(model)
        if base_explainer == "InputxGrad":
            self.base_explainer = InputxGrad(model)
        else:
            self.base_explainer = BaseGrad(model)

        self.num_steps = num_steps
        self.baseline = baseline

    # ───────────────────────────────────────────────────────────────────
    # Baseline construction
    # ───────────────────────────────────────────────────────────────────
    def _get_baseline(self, input_clean: torch.Tensor) -> torch.Tensor:
        """
        Build the baseline input x' for integration.
        
        For normalised inputs (zero-mean, unit-std), x'=0 corresponds to
        the physical mean of the channel and is the standard choice.
        """
        if self.baseline == "zeros":
            return torch.zeros_like(input_clean)
        elif self.baseline == "mean":
            return torch.full_like(input_clean, input_clean.mean().item())
        elif self.baseline == "blur":
            # Strong Gaussian blur as baseline — preserves some spatial context
            # Useful when x'=0 is too far from the data manifold
            import torch.nn.functional as F
            # Simple Gaussian blur via avg pooling (approximation)
            blurred = F.avg_pool2d(
                input_clean.unsqueeze(0).unsqueeze(0) if input_clean.dim() == 2 else input_clean,
                kernel_size=15, stride=1, padding=7,
            )
            return blurred.squeeze()
        else:
            raise ValueError(f"Unknown baseline: {self.baseline}")

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
        plot_explanations: bool = True,
        extent=[-12, 16, 37.5, 55.4],
        fig_path: str = ".figs/attributions",
        return_output: bool = False,
    ):
        output_idx = input.outputs.feature_names_to_idx[cfg_xai["explain"]["output"]]

        grads = {i: [] for i in range(input.num_pred_steps)}

        # --- Clean forward pass (for plotting reference) ---
        output = predict_step(
            self.model, input, batch_idx, checkpoint, cfg_model, cfg_dataset,
            cfg_xai, dataset_info, infer_ds, list_run_hour, use_old_weights,
            compute_grads=False,
        )
        if output is None:
            if return_output:
                return None, None
            return

        # --- Setup: decide where we perturb (input vs. forcing) ---
        if cfg_xai["explain"]["forcing_input"]:
            tensor_container = input.forcing
            input_idx = input.forcing.feature_names_to_idx[cfg_xai["explain"]["input"]]
        else:
            tensor_container = input.inputs
            input_idx = input.inputs.feature_names_to_idx[cfg_xai["explain"]["input"]]

        backup_tensor = tensor_container.tensor.clone()
        input_clean = backup_tensor[..., input_idx].clone()

        # --- Build the baseline ---
        baseline = self._get_baseline(input_clean)

        # --- Riemann sum integration ---
        # IG(x) ≈ (x - x') * (1/N) * Σ_{k=0}^{N-1} ∇f(x' + (k/N)(x - x'))
        # We use left-endpoint rule; k ∈ {0, 1, ..., N-1}, so α ∈ {0, 1/N, ..., (N-1)/N}.
        for step in range(self.num_steps):
            alpha = step / self.num_steps
            interpolated = baseline + alpha * (input_clean - baseline)

            # Inject the interpolated tensor back into the input
            new_tensor = backup_tensor.clone()
            new_tensor[..., input_idx] = interpolated

            if cfg_xai["explain"]["forcing_input"]:
                input.forcing.tensor = new_tensor
            else:
                input.inputs.tensor = new_tensor

            # Compute gradient at the interpolated point
            gradient, _ = self.base_explainer.compute_explanations(
                input=input, batch_idx=batch_idx, checkpoint=checkpoint,
                cfg_model=cfg_model, cfg_dataset=cfg_dataset, cfg_xai=cfg_xai,
                dataset_info=dataset_info, infer_ds=infer_ds,
                list_run_hour=list_run_hour, use_old_weights=use_old_weights,
                target=target, plot_explanations=False, extent=extent,
                fig_path=fig_path, return_output=True,
            )

            if gradient is None:
                if return_output:
                    return None, None
                return

            for i in range(len(gradient)):
                grads[i].append(gradient[i].squeeze(-1))

            self.model.zero_grad(set_to_none=True)

        # --- Aggregate: mean of gradients, then multiply by (x - x') ---
        grad_list = []
        # Broadcastable difference: shape must match what grads[i] contains
        diff = (input_clean - baseline)  # shape: depends on tensor_container

        if not cfg_xai["explain"]["end_to_end"]:
            for i in range(input.num_pred_steps):
                # Average over integration steps
                grad_mean = torch.stack(grads[i]).mean(dim=0)  # shape: (..., spatial)
                # Shape-align diff to grad_mean. grad_mean is after .squeeze(-1),
                # so we also squeeze diff to match.
                ig_attribution = diff.squeeze(-1) * grad_mean if diff.dim() == grad_mean.dim() + 1 else diff * grad_mean
                grad_list.append(ig_attribution.unsqueeze(-1))
        else:
            grad_mean = torch.stack(grads[0]).mean(dim=0)
            ig_attribution = diff.squeeze(-1) * grad_mean if diff.dim() == grad_mean.dim() + 1 else diff * grad_mean
            grad_list.append(ig_attribution.unsqueeze(-1))

        # --- Restore original tensor state ---
        if cfg_xai["explain"]["forcing_input"]:
            input.forcing.tensor = backup_tensor
        else:
            input.inputs.tensor = backup_tensor

        # --- Plot if requested ---
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
                fig_path_step = (
                    fig_path
                    + f"/end_to_end/{step}_step/input_{cfg_xai['explain']['input']}_output_{cfg_xai['explain']['output']}"
                )
                self.plot_explanations(
                    grad_list[0], input, output,
                    extent, "IntegratedGradients", fig_path_step,
                    runtime=runtimes[0], stats=stats, step=step - 1,
                    input_name=cfg_xai["explain"]["input"],
                    output_name=cfg_xai["explain"]["output"],
                    input_idx=input_idx, output_idx=output_idx,
                    target=cfg_xai["target"],
                    forcing=cfg_xai["explain"]["forcing_input"],
                    save_figs=cfg_xai["explain"]["exp_plot"]["save_figs"],
                )
            else:
                for i in range(len(grad_list)):
                    output_idx = output.feature_names_to_idx[cfg_xai["explain"]["output"]]
                    if cfg_xai["explain"]["forcing_input"]:
                        input_idx = input.forcing.feature_names_to_idx[cfg_xai["explain"]["input"]]
                    else:
                        input_idx = input.inputs.feature_names_to_idx[cfg_xai["explain"]["input"]]
                    fig_path_step = (
                        fig_path
                        + f"/{i+1}_step/input_{cfg_xai['explain']['input']}_output_{cfg_xai['explain']['output']}"
                    )
                    self.plot_explanations(
                        grad_list[i], input, output,
                        extent, "IntegratedGradients", fig_path_step,
                        runtime=runtimes[0], stats=stats, step=i,
                        input_name=cfg_xai["explain"]["input"],
                        output_name=cfg_xai["explain"]["output"],
                        input_idx=input_idx, output_idx=output_idx,
                        target=cfg_xai["target"],
                        forcing=cfg_xai["explain"]["forcing_input"],
                        save_figs=cfg_xai["explain"]["exp_plot"]["save_figs"],
                    )

        if return_output:
            return grad_list, output

        return grad_list