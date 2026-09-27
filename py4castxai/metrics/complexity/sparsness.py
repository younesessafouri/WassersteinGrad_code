
from py4castxai.explainers import SmoothGrad, BaseGrad,InputxGrad,BaseGradTV
import numpy as np
import torch
from py4castxai.explainers import explainers_registry
from typing import Dict, List, Optional, Tuple, Any

def normalize_expl(a,abs=True):
    
            if isinstance(a, np.ndarray):
                a = torch.from_numpy(a).to(torch.float32)
            if abs:
                return a.abs() / (a.abs().max() + 1e-8)
            return a / (a.abs().max() + 1e-8)


class Sparsness:
    """
    Sparsity (Gini Coefficient) — Complexity Evaluation Metric.

    Measures spatial concentration of attribution maps.
    Higher Gini = more focused explanation.
    """

    def __init__(
        self,
        model,
        explainer: str = "BaseGrad",
        seed: int = 42,
        **expl_params
    ):
        self.model = model
        self.seed = seed
        self.explainer_name = explainer
        self.explainer = self._init_explainer(explainer, expl_params)

    def _init_explainer(self, name: str, params: Dict):
        cls = explainers_registry.get(name)
        if cls is None:
            raise ValueError(f"Unknown explainer: {name}")
        return cls(self.model, **params)

    @staticmethod
    def _gini(explain_flat: np.ndarray) -> np.ndarray:
        """
        Compute Gini coefficient for batch of attribution maps.

        Args:
            explain_flat: [B, N] non-negative, need not sum to 1
        Returns:
            scores: [B] Gini coefficients in [0, 1]
        """
        explain_flat = explain_flat + 1e-7          # avoid zeros
        explain_flat = np.sort(explain_flat, axis=-1)  # ascending

        B, N = explain_flat.shape
        arange = np.arange(1, N + 1)[np.newaxis, :]  # [1, N]

        numerator   = ((2 * arange - N - 1) * explain_flat).sum(axis=-1)
        denominator = N * explain_flat.sum(axis=-1)

        return numerator / denominator  # [B]

    def evaluate(
        self,
        dataloader,
        checkpoint,
        cfg_model,
        cfg_dataset,
        cfg_xai,
        dataset_info,
        infer_ds,
        list_run_hour,
        use_old_weights,
        target,
        plot_explanations,
        extent,
        fig_path,
        valid_runtimes=None ,precip_thresh=5   # ← pass ROAD valid events for consistency
    ):
        """
        Evaluate Gini sparsity of attribution maps.

        Parameters
        ----------
        valid_runtimes : set of str, optional
            If provided, only evaluate on these event timestamps.
            Should be the same set used for ROAD evaluation.
            Ensures all metrics reported on identical events.

        Returns
        -------
        all_scores      : dict {lead_time: [scores]}
        all_scores_mean : list of means per lead time
        all_scores_sem  : list of SEMs per lead time  ← SEM not STD
        """
        # Reproducibility
        torch.manual_seed(self.seed)
        np.random.seed(self.seed)

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        n_steps = (
            1 if cfg_xai["explain"]["end_to_end"]
            else int(cfg_dataset["num_pred_steps_val_test"])
        )
        print("igziegiuzgiug",precip_thresh)
        all_scores = {i: [] for i in range(n_steps)}
        reg_steps =int(cfg_dataset["num_pred_steps_val_test"])
        with torch.autograd.set_grad_enabled(True):

            for batch_idx, batch in enumerate(dataloader):
                runtime = (
                    infer_ds.sample_list[batch_idx]
                    .timestamps.datetime
                    .strftime("%Y%m%d%H")
                )
                # ── Runtime check ─────────────────────────────────
                if valid_runtimes is not None:
                    runtime = (
                        infer_ds.sample_list[batch_idx]
                        .timestamps.datetime
                        .strftime("%Y%m%d%H")
                    )
                    if runtime not in valid_runtimes:
                        continue

                # ── ROI setup ─────────────────────────────────────
                H, W = batch.inputs.tensor.shape[2:4]
                lon0, lon1 = target[0], target[1]
                lat0, lat1 = target[2], target[3]
                lat_min, lat_max = extent[2], extent[3]
                lon_min, lon_max = extent[0], extent[1]

                lat_step = (lat_max - lat_min) / H
                lon_step = (lon_max - lon_min) / W

                i_min = int((lat0 - lat_min) / lat_step)
                i_max = int((lat1 - lat_min) / lat_step)
                j_min = int((lon0 - lon_min) / lon_step)
                j_max = int((lon1 - lon_min) / lon_step)

                output_idx = batch.outputs.feature_names_to_idx[
                    cfg_xai["explain"]["output"]
                ]

                # ── Precipitation filter (if no valid_runtimes) ───
                if valid_runtimes is None:
                    precip_mean = batch.outputs.tensor[
                        :, 0, i_min:i_max, j_min:j_max, output_idx
                    ].mean().item()
                    if precip_mean <precip_thresh:
                        continue

                # ── Device setup ──────────────────────────────────
                batch.inputs.tensor  = batch.inputs.tensor.to(device)
                batch.forcing.tensor = batch.forcing.tensor.to(device)
                if batch.outputs is not None:
                    batch.outputs.tensor = batch.outputs.tensor.to(device)

                if cfg_xai["explain"]["forcing_input"]:
                    batch.forcing.tensor.requires_grad_()
                else:
                    batch.inputs.tensor.requires_grad_()

                # ── Compute attribution ───────────────────────────
                explain_list = self.explainer.compute_explanations(
                    batch, batch_idx, checkpoint,
                    cfg_model, cfg_dataset, cfg_xai, dataset_info,
                    infer_ds, list_run_hour, use_old_weights,
                    target,
                    plot_explanations=plot_explanations,
                    extent=extent,
                    fig_path=fig_path,
                    return_output=False
                )
                if explain_list is None:
                    continue

                for i, raw in enumerate(explain_list):

                    explain = normalize_expl(raw)
                    explain_flat = explain.reshape(1,-1)
                    

                    with torch.no_grad():
                        explain_np = explain_flat.cpu().detach().numpy()
                        scores = self._gini(explain_np)
                        all_scores[i].extend(scores.tolist())

                        # ── Per-event print (mirrors ROAD format) ─────
                        for b, s in enumerate(scores):
                            print(
                                f"[Sparsity] {runtime} | "
                                f"t+{reg_steps} | "
                                f"gini={s:.4f} | "
                                f"n_features={explain_np.shape[-1]}"
            )

        # ── Aggregate ─────────────────────────────────────────────
        all_scores_mean = []
        all_scores_sem  = []

        for i in all_scores:
            arr = np.array(all_scores[i])
            n   = len(arr)
            all_scores_mean.append(float(np.nanmean(arr)))
            all_scores_sem.append(
                float(np.nanstd(arr) / np.sqrt(n))  # SEM not STD
            )
            print(
                f"[Sparsity] t+{i+1}: "
                f"mean={all_scores_mean[-1]:.4f} "
                f"sem={all_scores_sem[-1]:.4f} "
                f"n={n}"
            )

        return all_scores, all_scores_mean, all_scores_sem

