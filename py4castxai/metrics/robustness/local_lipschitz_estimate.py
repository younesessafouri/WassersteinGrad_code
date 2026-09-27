# from py4castxai.explainers import SmoothGrad, BaseGrad,InputxGrad,BaseGradTV
# import numpy as np
# import torch

# from typing import Dict, List, Optional, Tuple, Any
# from py4castxai.explainers import explainers_registry


# def normalize_expl(a,abs=True):
            
#             if isinstance(a, np.ndarray):
#                 a = torch.from_numpy(a).to(torch.float32)
#             if abs:
#                 return a.abs() / (a.abs().max() + 1e-8)
#             return a / (a.abs().max() + 1e-8)




# class LocalLipschitzEstimate:
#     """
#     Local Lipschitz Estimate (LLE) — Robustness Metric.
    
#     LLE = max over perturbations of:
#           ||Φ(f, x) - Φ(f, x+δ)||₂ / ||δ||₂
    
#     Lower is better (↓) — more robust attribution.
#     Random baseline: high LLE (attribution changes a lot).
    
#     Computes both cosine and L2 variants simultaneously.
#     """

#     def __init__(
#         self,
#         model,
#         explainer: str = "BaseGrad",
#         std_noise: float = 0.1,
#         num_iter: int = 5,
#         seed: int = 42,
#         **expl_params
#     ):
#         self.model          = model
#         self.std_noise      = std_noise
#         self.num_iter       = num_iter
#         self.seed           = seed
#         self.explainer_name = explainer
#         self.explainer      = self._init_explainer(explainer, expl_params)
#         print("[LLE] Computing both cosine (scale-invariant) and L2 (standard) distances.")

#     def _init_explainer(self, name: str, params: Dict):
#         cls = explainers_registry.get(name)
#         if cls is None:
#             raise ValueError(f"Unknown explainer: {name}")
#         return cls(self.model, **params)

#     @staticmethod
#     def _to_flat(raw: torch.Tensor) -> torch.Tensor:
#         """Unified shape → [1, H*W] regardless of explainer."""
#         return raw.reshape(1, -1)

#     @staticmethod
 
#     def _cosine_distance(a: torch.Tensor, b: torch.Tensor) -> float:
#         """
#         Normalized Euclidean distance between two attribution maps.
#         Scale-invariant: measures structural change only.
#         Scales linearly with small angular changes, making it valid for LLE.
#         Returns value in [0, 1.414]: 0 = identical, 1.414 (sqrt(2)) = orthogonal.
#         """
#         a_flat = a.reshape(-1).float()
#         b_flat = b.reshape(-1).float()
        

#         # a_flat = a_flat / torch.sum(a_flat)
#         # b_flat = b_flat/torch.sum(b_flat)
#         # # Add epsilon to prevent division by zero
#         norm_a = torch.norm(a_flat) + 1e-8
#         norm_b = torch.norm(b_flat) + 1e-8
        
#         # norm_a = torch.sum(a_flat)
#         # norm_b = torch.sum(b_flat)
#         # Project both vectors onto the unit hypersphere
#         a_normalized = a_flat / norm_a
#         b_normalized = b_flat / norm_b
        
#         # Compute the L2 distance between the normalized vectors
#         dist = torch.norm(a_normalized - b_normalized)
        
#         return float(dist.item())
#     @staticmethod
#     def _l2_distance(a: torch.Tensor, b: torch.Tensor) -> float:
#         """
#         L2 distance between two attribution maps.
#         WARNING: biased toward sparse methods.
#         """
#         return torch.norm(a.reshape(-1).float() - b.reshape(-1).float()).item()

#     def _input_distance_cosine(self, delta: torch.Tensor, H: int, W: int) -> float:
#         """Normalised input perturbation for cosine mode (comparable across resolutions)."""
#         return torch.norm(delta).item() + 1e-8 #/ np.sqrt(H * W) + 1e-8

#     def _input_distance_l2(self, delta: torch.Tensor) -> float:
#         """Raw input perturbation norm for L2 mode."""
#         return torch.norm(delta).item() + 1e-8 #/ np.sqrt(512 * 640) + 1e-8

#     def evaluate(
#         self,
#         dataloader,
#         checkpoint,
#         cfg_model, cfg_dataset, cfg_xai,
#         dataset_info, infer_ds,
#         list_run_hour, use_old_weights,
#         target, plot_explanations, extent, fig_path,
#         valid_runtimes=None,precip_thresh=None
#     ):
#         torch.manual_seed(self.seed)
#         np.random.seed(self.seed)

#         device    = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#         n_steps   = int(cfg_dataset["num_pred_steps_val_test"])
#         # Track scores for both metrics independently
#         all_scores_cosine = {k: [] for k in range(n_steps)}
#         all_scores_l2     = {k: [] for k in range(n_steps)}
#         forcing_input = cfg_xai["explain"]["forcing_input"]

#         with torch.autograd.set_grad_enabled(True):

#             for batch_idx, batch in enumerate(dataloader):
#                 # ── Runtime filter ─────────────────────────────
#                 runtime = (
#                     infer_ds.sample_list[batch_idx]
#                     .timestamps.datetime
#                     .strftime("%Y%m%d%H")
#                 )
#                 if valid_runtimes is not None:
#                     if runtime not in valid_runtimes:
#                         continue

#                 # ── ROI setup ──────────────────────────────────
#                 H, W = batch.inputs.tensor.shape[2:4]
#                 lon0, lon1 = target[0], target[1]
#                 lat0, lat1 = target[2], target[3]
#                 lat_min, lat_max = extent[2], extent[3]
#                 lon_min, lon_max = extent[0], extent[1]
#                 lat_step = (lat_max - lat_min) / H
#                 lon_step = (lon_max - lon_min) / W
#                 i_min = int((lat0 - lat_min) / lat_step)
#                 i_max = int((lat1 - lat_min) / lat_step)
#                 j_min = int((lon0 - lon_min) / lon_step)
#                 j_max = int((lon1 - lon_min) / lon_step)

#                 output_idx = batch.outputs.feature_names_to_idx[
#                     cfg_xai["explain"]["output"]
#                 ]
#                 # ── Precipitation filter ───────────────────────
#                 if valid_runtimes is None:
#                     precip_mean = batch.outputs.tensor[
#                         :, 0, i_min:i_max, j_min:j_max, output_idx
#                     ].mean().item()
#                     if precip_mean < precip_thresh:
#                         continue

#                 # ── Device setup ───────────────────────────────
#                 batch.inputs.tensor  = batch.inputs.tensor.to(device)
#                 batch.forcing.tensor = batch.forcing.tensor.to(device)
#                 if batch.outputs is not None:
#                     batch.outputs.tensor = batch.outputs.tensor.to(device)
#                 batch.inputs.tensor.requires_grad_()

#                 # ── Input index ────────────────────────────────
#                 if forcing_input:
#                     input_idx = batch.forcing.feature_names_to_idx[
#                         cfg_xai["explain"]["input"]
#                     ]
#                     input_clean = batch.forcing.tensor[..., input_idx].clone()
#                 else:
#                     input_idx = batch.inputs.feature_names_to_idx[
#                         cfg_xai["explain"]["input"]
#                     ]
#                     input_clean = batch.inputs.tensor[..., input_idx].clone()

#                 noise_std      = self.std_noise * (input_clean.max() - input_clean.min()).item()
#                 explainer_seed = self.seed + batch_idx * 10000

#                 torch.manual_seed(explainer_seed)
#                 # ── Clean attribution ──────────────────────────
#                 explain_clean, output_clean = self.explainer.compute_explanations(
#                     batch, batch_idx, checkpoint,
#                     cfg_model, cfg_dataset, cfg_xai, dataset_info,
#                     infer_ds, list_run_hour, use_old_weights,
#                     target,
#                     plot_explanations=plot_explanations,
#                     extent=extent,
#                     fig_path=fig_path,
#                     return_output=True
#                 )
#                 if explain_clean is None:
#                     continue

#                 # ── LLE per lead time ──────────────────────────
#                 for k in range(len(explain_clean)):

#                     a_clean      = normalize_expl(explain_clean[k])
#                     a_clean_flat = self._to_flat(a_clean)  # [1, H*W]

#                     ratios_cosine = []
#                     ratios_l2     = []
#                     input_clean_full = batch.inputs.tensor.clone().detach()

#                     for i in range(self.num_iter):

#                         iter_seed = self.seed + batch_idx * 1000 + k * 100 + i
#                         torch.manual_seed(iter_seed)
#                         delta = torch.randn_like(input_clean) * noise_std

#                         # ── Apply perturbation ─────────────────
#                         try:
#                             perturbed_inputs = batch.inputs.tensor.clone().detach()
#                             perturbed_inputs[..., input_idx] = (input_clean + delta).detach()
#                             batch.inputs.tensor = perturbed_inputs
#                             batch.inputs.tensor.requires_grad_()

#                             torch.manual_seed(explainer_seed)
#                             explain_noised = self.explainer.compute_explanations(
#                                 batch, batch_idx, checkpoint,
#                                 cfg_model, cfg_dataset, cfg_xai,
#                                 dataset_info, infer_ds,
#                                 list_run_hour, use_old_weights,
#                                 target,
#                                 plot_explanations=False
#                             )
#                         finally:
#                             batch.inputs.tensor = input_clean_full.clone()
#                             batch.inputs.tensor.requires_grad_()

#                         if explain_noised is None:
#                             continue

#                         a_noised      = normalize_expl(explain_noised[k])
#                         a_noised_flat = self._to_flat(a_noised)

#                         # ── Compute both distances in one pass ─
#                         diff_cosine = self._cosine_distance(a_clean_flat, a_noised_flat)
#                         diff_l2     = self._l2_distance(a_clean_flat, a_noised_flat)

#                         input_dist_cosine = self._input_distance_cosine(delta, H, W)
#                         input_dist_l2     = self._input_distance_l2(delta)

#                         ratios_cosine.append(diff_cosine / input_dist_cosine)
#                         ratios_l2.append(diff_l2 / input_dist_l2)

#                     if ratios_cosine:
#                         lle_cosine = max(ratios_cosine)
#                         lle_l2     = max(ratios_l2)
#                         all_scores_cosine[k].append(lle_cosine)
#                         all_scores_l2[k].append(lle_l2)
#                         print(
#                             f"[LLE] {runtime} | t+{n_steps+1} | "
#                             f"cosine={lle_cosine:.7f} | "
#                             f"l2={lle_l2:.5f} | "
#                             f"n_iters={len(ratios_cosine)}"
#                         )

#         # ── Aggregate both metrics ─────────────────────────────
#         means_cosine, sems_cosine = [], []
#         means_l2,     sems_l2     = [], []

#         for scores, means, sems, label in [
#             (all_scores_cosine, means_cosine, sems_cosine, "cosine"),
#             (all_scores_l2,     means_l2,     sems_l2,     "l2"),
#         ]:
#             arr = np.array(scores[0])
#             n   = len(arr)
#             m   = float(np.nanmean(arr)) if n > 0 else float("nan")
#             s   = float(np.nanstd(arr) / np.sqrt(n)) if n > 0 else float("nan")
#             means.append(m)
#             sems.append(s)
#             print(
#                 f"[LLE] Aggregated t+{k+1} [{label}]: "
#                 f"mean={m:.5f} sem={s:.5f} n={n}"
#             )

#         return all_scores_cosine, all_scores_l2, means_cosine, means_l2, sems_cosine, sems_l2






"""
Local Lipschitz Estimate (LLE) with a swappable probe distribution.

    LLE = max_i  || Phi(f, x) - Phi(f, x + delta_i) ||  /  || delta_i ||

Lower is better. The point of this version is that `probe` controls the
distribution delta is drawn from:

    probe="iso"   delta ~ N(0, s^2 I) on the explained channel  (standard LLE)
    probe="eda"   delta drawn from the ensemble subspace, rescaled to the
                  SAME norm as the isotropic probe

Only the direction changes; the magnitude is held fixed, so the LLE
denominator is identical and any difference comes from the numerator.

WHY THIS MATTERS
----------------
G-iso SmoothGrad convolves the model in all D directions. EDA-SmoothGrad
convolves it in the (K-1)-dimensional ensemble span only. An isotropic probe
lies almost entirely OUTSIDE that span -- for a 512x640 channel the fraction
inside is sqrt(16/327680) ~ 0.7% -- so the isotropic probe tests EDA in
directions it never smoothed over, while testing G-iso in directions it did.
A full-rank smoother wins that comparison by construction, and sigma -> inf
wins it perfectly by returning a constant map.

Running both probes turns "EDA looks less robust" into a measurable statement
about the metric rather than about the attribution. If the ranking flips, the
robustness ordering is set by the probe distribution, not by the explanation.

FIXES vs the previous version
-----------------------------
1. Perturbed attributions were computed INSIDE the lead-time loop, so the same
   num_iter attributions were recomputed for every k. Now computed once per
   perturbation and indexed by k -- an n_steps-fold speedup.
2. Lead-time label printed `t+{n_steps+1}` for every k. Now `t+{k+1}`.
3. Aggregation hardcoded `scores[0]` and leaked `k` from the inner loop, so
   only the first lead time was ever reported. Now loops over all k.
4. Optional percentile normalisation. `normalize_expl` divides by max|a|, so
   one hot pixel -- exactly what shattered gradients produce in the G-iso map
   -- compresses the whole map and shrinks the L2 distance artificially.
   Cosine renormalises internally and is immune; report it as primary.
5. delta uses a dedicated torch.Generator instead of global manual_seed.

NOTE ON COMMON RANDOM NUMBERS
-----------------------------
The explainer classes seed their own generator from self.seed at the start of
every compute_explanations call, so the internal perturbations are identical
between the clean and probed calls. That is what you want here: the difference
between the two maps reflects the input probe, not the explainer's own Monte
Carlo noise. The old `torch.manual_seed(explainer_seed)` trick is no longer
needed and no longer has any effect on those classes.
"""

from typing import Dict, List, Optional

import numpy as np
import torch

from py4castxai.explainers import explainers_registry
from py4castxai.explainers.edasmoothGrad import EDASmoothGrad


def normalize_expl(a, abs: bool = True, mode: str = "max", pct: float = 99.9):
    """
    Normalise an attribution map.

    mode="max"  divide by max|a|  (original behaviour; sensitive to one pixel)
    mode="pct"  divide by the `pct`-th percentile of |a|  (robust)
    mode="l2"   divide by ||a||_2
    """
    if isinstance(a, np.ndarray):
        a = torch.from_numpy(a).to(torch.float32)
    a = a.abs() if abs else a

    if mode == "max":
        denom = a.abs().max()
    elif mode == "pct":
        denom = torch.quantile(a.abs().reshape(-1).float(), pct / 100.0)
    elif mode == "l2":
        denom = torch.norm(a.reshape(-1).float())
    else:
        raise ValueError(f"unknown mode {mode}")
    return a / (denom + 1e-8)


class LocalLipschitzEstimate:
    """
    Parameters
    ----------
    probe : {"iso", "eda"}
        Distribution the stability probe is drawn from. "eda" requires
        `eda_dir` and only works when the explained variable is a model input
        (not a forcing), since PEARO supplies analysis fields.
    std_noise : float
        Probe amplitude as a fraction of the explained channel's range. Used
        for BOTH probes, so they differ only in direction.
    num_iter : int
        Number of probes the max is taken over. LLE is a supremum estimate, so
        it is biased low and noisy at small num_iter -- and more so for the
        rougher map, which is exactly the comparison being made. 5 is too few;
        20+ is better. Cost is num_iter x N backprops per event.
    norm_mode : {"max", "pct", "l2"}
        Attribution normalisation. See `normalize_expl`.
    """

    def __init__(self, model, explainer: str = "BaseGrad",
                 std_noise: float = 0.1, num_iter: int = 10, seed: int = 42,
                 probe: str = "eda", eda_dir: Optional[str] = "/mnt/usb-Seagate_FireCuda_SSD_00000000NABG0DN9-0:0-part1/eda_perturbations/work/akodads/data/PEARO/",
                 num_members: int = 17, probe_scale: float = 1.0,
                 norm_mode: str = "max", norm_pct: float = 99.9,
                 verbose: bool = True, **expl_params):

        if probe not in ("iso", "eda"):
            raise ValueError("probe must be 'iso' or 'eda'")
        if probe == "eda" and eda_dir is None:
            raise ValueError("probe='eda' requires eda_dir")

        self.model = model
        self.std_noise = std_noise
        self.num_iter = int(num_iter)
        self.seed = int(seed)
        self.probe = probe
        self.eda_dir = eda_dir
        self.num_members = int(num_members)
        self.probe_scale = float(probe_scale)
        self.norm_mode = norm_mode
        self.norm_pct = norm_pct
        self.verbose = verbose

        self.explainer_name = explainer
        expl_params["verbose"] = False
        if explainer =="EDASmoothGrad":
            expl_params["eda_dir"] = eda_dir
        self.explainer = self._init_explainer(explainer, expl_params)

        self._eda_loader = None
        if probe == "eda":
            self._eda_loader = EDASmoothGrad(
                model, eda_dir=eda_dir, mode="subspace",
                num_members=num_members, seed=seed, verbose=False)

        if self.verbose:
            print(f"[LLE] explainer={explainer} probe={probe} "
                  f"num_iter={self.num_iter} norm_mode={norm_mode}")
            print("[LLE] cosine is scale-invariant and is the primary metric; "
                  "L2 is reported for reference.")

    def _init_explainer(self, name: str, params: Dict):
        cls = explainers_registry.get(name)
     
        if cls is None:
            raise ValueError(f"Unknown explainer: {name}")
        return cls(self.model, **params)

    # -- distances -----------------------------------------------------

    @staticmethod
    def _to_flat(raw: torch.Tensor) -> torch.Tensor:
        return raw.reshape(1, -1)

    @staticmethod
    def _cosine_distance(a: torch.Tensor, b: torch.Tensor) -> float:
        """
        L2 distance between the two maps projected onto the unit hypersphere.
        Scale-invariant: measures structural change only. In [0, sqrt(2)].
        """
        a_flat = a.reshape(-1).float()
        b_flat = b.reshape(-1).float()
        a_n = a_flat / (torch.norm(a_flat) + 1e-8)
        b_n = b_flat / (torch.norm(b_flat) + 1e-8)
        return float(torch.norm(a_n - b_n).item())

    @staticmethod
    def _l2_distance(a: torch.Tensor, b: torch.Tensor) -> float:
        """Raw L2. Biased by whatever normalisation was applied upstream."""
        return float(torch.norm(a.reshape(-1).float()
                                - b.reshape(-1).float()).item())

    # -- probe construction ---------------------------------------------

    def _load_eda_basis(self, batch, batch_idx, target_obj, input_idx,
                        dataset_info, infer_ds, cfg_dataset, device):
        """
        Return the K member deviations for the explained channel only,
        standardised, shaped like input_clean: (K, B, T, Y, X).

        Reuses EDASmoothGrad's loader so the caching, name mapping and
        standardisation are identical to the attribution side. Only one
        channel is kept (~22 MB instead of ~450 MB).
        """
        gen = torch.Generator(device=device)
        gen.manual_seed(self.seed)

        self._eda_loader._prepare_perturbations(
            backup_tensor=target_obj.tensor, target_obj=target_obj,
            dataset_info=dataset_info, device=device, generator=gen,
            infer_ds=infer_ds, batch_idx=batch_idx, cfg_dataset=cfg_dataset,
            eda_dir=self.eda_dir, eda_perturbations=None)

        basis = self._eda_loader._base[..., input_idx].clone()   # (K,B,T,Y,X)
        self._eda_loader._release()
        return basis

    def _sample_delta(self, ref_shape, target_norm, generator, device, dtype,
                      basis=None):
        """
        Draw one probe with L2 norm exactly `target_norm`.

        iso: white Gaussian, then rescaled.
        eda: random combination of member deviations, then rescaled to the
             SAME norm, so the two probes differ only in direction.
        """
        if self.probe == "iso" or basis is None:
            delta = torch.randn(ref_shape, generator=generator,
                                device=device, dtype=dtype)
        else:
            K = basis.shape[0]
            w = torch.randn(K, generator=generator, device=device,
                            dtype=basis.dtype) / np.sqrt(K - 1)
            delta = torch.einsum("k...,k->...", basis, w).to(dtype)

        return delta * (target_norm / (torch.norm(delta) + 1e-8))

    # -- main loop -------------------------------------------------------

    def evaluate(self, dataloader, checkpoint, cfg_model, cfg_dataset, cfg_xai,
                 dataset_info, infer_ds, list_run_hour, use_old_weights,
                 target, plot_explanations, extent, fig_path,
                 valid_runtimes=None, precip_thresh=None):

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        n_steps = int(cfg_dataset["num_pred_steps_val_test"])
        forcing_input = cfg_xai["explain"]["forcing_input"]

        if self.probe == "eda" and forcing_input:
            raise ValueError(
                "probe='eda' explains a forcing variable, but PEARO supplies "
                "analysis fields only. Use probe='iso' for forcing inputs.")

        all_scores_cosine = {k: [] for k in range(n_steps)}
        all_scores_l2 = {k: [] for k in range(n_steps)}
        probe_norms: List[float] = []

        with torch.autograd.set_grad_enabled(True):
            for batch_idx, batch in enumerate(dataloader):

                runtime = (infer_ds.sample_list[batch_idx]
                           .timestamps.datetime.strftime("%Y%m%d%H"))

                if valid_runtimes is not None and runtime not in valid_runtimes:
                    continue

                # -- ROI ------------------------------------------------
                H, W = batch.inputs.tensor.shape[2:4]
                lon0, lon1, lat0, lat1 = target[0], target[1], target[2], target[3]
                lat_min, lat_max = extent[2], extent[3]
                lon_min, lon_max = extent[0], extent[1]
                i_min = int((lat0 - lat_min) / ((lat_max - lat_min) / H))
                i_max = int((lat1 - lat_min) / ((lat_max - lat_min) / H))
                j_min = int((lon0 - lon_min) / ((lon_max - lon_min) / W))
                j_max = int((lon1 - lon_min) / ((lon_max - lon_min) / W))

                output_idx = batch.outputs.feature_names_to_idx[
                    cfg_xai["explain"]["output"]]

                if valid_runtimes is None:
                    precip_mean = batch.outputs.tensor[
                        :, 0, i_min:i_max, j_min:j_max, output_idx].mean().item()
                    if precip_mean < precip_thresh:
                        continue

                # -- device ---------------------------------------------
                batch.inputs.tensor = batch.inputs.tensor.to(device)
                batch.forcing.tensor = batch.forcing.tensor.to(device)
                if batch.outputs is not None:
                    batch.outputs.tensor = batch.outputs.tensor.to(device)
                batch.inputs.tensor.requires_grad_()

                target_obj = batch.forcing if forcing_input else batch.inputs
                input_idx = target_obj.feature_names_to_idx[
                    cfg_xai["explain"]["input"]]
                input_clean = target_obj.tensor[..., input_idx].clone()

                # Probe amplitude: identical for both probes, so only the
                # direction differs.
                noise_std = self.std_noise * float(
                    (input_clean.max() - input_clean.min()).item())
                target_norm = (self.probe_scale * noise_std
                               * float(np.sqrt(input_clean.numel())))

               
                gen = torch.Generator(device=device)
                gen.manual_seed(self.seed + batch_idx * 10000)

                # -- clean attribution ----------------------------------
                explain_clean, _ = self.explainer.compute_explanations(
                    batch, batch_idx, checkpoint, cfg_model, cfg_dataset,
                    cfg_xai, dataset_info, infer_ds, list_run_hour,
                    use_old_weights, target,
                    plot_explanations=plot_explanations, extent=extent,
                    fig_path=fig_path, return_output=True)
                if explain_clean is None:
                    continue
                
                basis = None
                if self.probe == "eda":
                    basis = self._load_eda_basis(
                        batch, batch_idx, target_obj, input_idx,
                        dataset_info, infer_ds, cfg_dataset, device)

                n_lead = len(explain_clean)
                clean_flat = [
                    self._to_flat(normalize_expl(explain_clean[k],
                                                 mode=self.norm_mode,
                                                 pct=self.norm_pct))
                    for k in range(n_lead)]

                ratios_cos = {k: [] for k in range(n_lead)}
                ratios_l2 = {k: [] for k in range(n_lead)}

                input_full = batch.inputs.tensor.clone().detach()

                # -- probes: OUTER loop, all lead times per probe --------
                for i in range(self.num_iter):
                    delta = self._sample_delta(
                        input_clean.shape, target_norm, gen, device,
                        input_clean.dtype, basis=basis)
                    probe_norms.append(float(torch.norm(delta).item()))

                    try:
                        perturbed = batch.inputs.tensor.clone().detach()
                        perturbed[..., input_idx] = (input_clean + delta).detach()
                        batch.inputs.tensor = perturbed
                        batch.inputs.tensor.requires_grad_()

                        explain_noised = self.explainer.compute_explanations(
                            batch, batch_idx, checkpoint, cfg_model,
                            cfg_dataset, cfg_xai, dataset_info, infer_ds,
                            list_run_hour, use_old_weights, target,
                            plot_explanations=False)
                    finally:
                        batch.inputs.tensor = input_full.clone()
                        batch.inputs.tensor.requires_grad_()

                    if explain_noised is None:
                        continue

                    d_in = float(torch.norm(delta).item()) + 1e-8
                    for k in range(n_lead):
                        noised_flat = self._to_flat(
                            normalize_expl(explain_noised[k],
                                           mode=self.norm_mode,
                                           pct=self.norm_pct))
                        ratios_cos[k].append(
                            self._cosine_distance(clean_flat[k], noised_flat) / d_in)
                        ratios_l2[k].append(
                            self._l2_distance(clean_flat[k], noised_flat) / d_in)

                # -- per lead time --------------------------------------
                for k in range(n_lead):
                    if not ratios_cos[k]:
                        continue
                    lle_cos = max(ratios_cos[k])
                    lle_l2 = max(ratios_l2[k])
                    all_scores_cosine[k].append(lle_cos)
                    all_scores_l2[k].append(lle_l2)
                    if self.verbose:
                        print(f"[LLE] {runtime} | t+{k+1} | probe={self.probe} "
                              f"| cosine={lle_cos:.7f} | l2={lle_l2:.5f} "
                              f"| n_iters={len(ratios_cos[k])}")

                del basis

        # -- aggregate over ALL lead times ---------------------------------
        means_cosine, sems_cosine, means_l2, sems_l2 = [], [], [], []
        for k in range(n_steps):
            for scores, means, sems, label in [
                (all_scores_cosine, means_cosine, sems_cosine, "cosine"),
                (all_scores_l2, means_l2, sems_l2, "l2"),
            ]:
                arr = np.asarray(scores[k], dtype=float)
                n = arr.size
                m = float(np.nanmean(arr)) if n else float("nan")
                s = float(np.nanstd(arr) / np.sqrt(n)) if n else float("nan")
                means.append(m)
                sems.append(s)
                print(f"[LLE] Aggregated t+{k+1} [{label}] probe={self.probe}: "
                      f"mean={m:.5f} sem={s:.5f} n={n}")

        if probe_norms and self.verbose:
            pn = np.asarray(probe_norms)
            print(f"[LLE] probe norm: mean={pn.mean():.4f} "
                  f"std={pn.std():.2e} (should be ~constant by construction)")

        return (all_scores_cosine, all_scores_l2,
                means_cosine, means_l2, sems_cosine, sems_l2)