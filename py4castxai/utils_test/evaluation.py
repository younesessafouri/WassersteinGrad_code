"""
Evaluation Module for Py4CastXai
=================================

Handles metric evaluation for different explainers.
"""

import logging
import torch
from typing import Dict, Any, List
from py4castxai.metrics import registry
from py4castxai.explainers import explainers_registry
import logging
import torch
import numpy as np
from scipy import stats
from typing import Dict, Any, List
import os
import warnings
import numpy as np
from scipy import stats

def paired_uncertainty_from_results(results, metric_name="LocalLipschitzEstimate",
                                    name_a="WassersteinGrad", name_b="SmoothGrad"):
    a = np.asarray(results[name_a][metric_name]["vals"], dtype=float).ravel()
    b = np.asarray(results[name_b][metric_name]["vals"], dtype=float).ravel()

    # keep only events valid in BOTH (drop any NaN in either)
    ok = np.isfinite(a) & np.isfinite(b)
    a, b = a[ok], b[ok]

    assert a.shape == b.shape, f"misaligned: {a.shape} vs {b.shape}"
    n = len(a)
    d = a - b                              # per-event difference (WG - SG)

    mean_diff = d.mean()
    sem_diff  = d.std(ddof=1) / np.sqrt(n)
    t_crit    = stats.t.ppf(0.975, df=n-1)
    ci = (mean_diff - t_crit*sem_diff, mean_diff + t_crit*sem_diff)

    t_stat, p_t   = stats.ttest_rel(a, b)          # paired t-test
    w_stat, p_w   = stats.wilcoxon(a, b)           # non-parametric backup

    print(f"{name_a} − {name_b}  ({metric_name}, n={n} paired events)")
    print(f"  {name_a} mean : {a.mean():.6f}")
    print(f"  {name_b} mean : {b.mean():.6f}")
    print(f"  paired diff  : {mean_diff:.6f} ± {sem_diff:.6f} (SEM)")
    print(f"  95% CI       : [{ci[0]:.6f}, {ci[1]:.6f}]")
    print(f"  paired t p   : {p_t:.2e}")
    print(f"  Wilcoxon p   : {p_w:.2e}")
    return {"mean_diff": mean_diff, "sem_diff": sem_diff, "ci": ci,
            "p_t": p_t, "p_wilcoxon": p_w, "n": n}

# usage:
# paired_uncertainty_from_results(self.results, "LocalLipschitzEstimate")
class MetricEvaluator:
    """Evaluates a single metric for an explainer."""
    
    def __init__(self, model, metric_type: str, metric_name: str, 
                 explainer_name: str, metric_params: Dict, explainer_params: Dict):
        self.model = model
        self.metric_type = metric_type
        self.metric_name = metric_name
        self.explainer_name = explainer_name
        self.logger = logging.getLogger(__name__)
        
        # Get metric class and initialize
        metric_cls = registry[metric_type][metric_name]
        self.metric = metric_cls(
            model, 
            explainer=explainer_name,
            **metric_params,
            **explainer_params
        )
    
    def evaluate(self, dataloader, checkpoint, cfg_model, cfg_dataset, cfg_xai,
                 dataset_info, infer_ds, list_run_hour, use_old_weights, 
                 target, plot_explanations=False, extent=None, fig_path=None,precip_thresh=None):
        """Run metric evaluation."""
        self.logger.info(f"Evaluating {self.metric_name} for {self.explainer_name}")
        
        result = self.metric.evaluate(
            dataloader, checkpoint,
            cfg_model, cfg_dataset, cfg_xai,
            dataset_info, infer_ds,
            list_run_hour, use_old_weights, target,
            plot_explanations=plot_explanations,
            extent=extent,
            fig_path=fig_path,precip_thresh=precip_thresh
        )
        
        return self._format_result(result)
    
    def _format_result(self, result):
        """Format evaluation result based on metric type."""
        if self.metric_name == "bla":
            return {"vals": result[-1]}
        else:
            return {"vals": result[0], "mean": result[1],"std":result[2]}


class EvaluationRunner:
    """Orchestrates evaluation across multiple explainers and metrics."""
    
    def __init__(self, model, config, dataloader, dataset_info, infer_ds):
        self.model = model
        self.config = config
        self.dataloader = dataloader
        self.dataset_info = dataset_info
        self.infer_ds = infer_ds
        self.logger = logging.getLogger(__name__)
        
        self.results = {}
    
    def run(self) -> Dict[str, Dict[str, Any]]:
        """Run evaluation for all configured explainers and metrics."""
        cfg_xai = self.config.cfg_xai
        
        for explainer_name, expl_params in cfg_xai["compare"]["explainers"].items():
            self.logger.info(f"Evaluating explainer: {explainer_name}")
            self.results[explainer_name] = self._evaluate_explainer(
                explainer_name, expl_params
            )
   
        
        # Generate visualizations if requested
        # if cfg_xai["eval"]["eval_plot"]["plot_evaluations"]:
        #     self._generate_plots()
        
        return self.results
    
    # def _evaluate_explainer(self, explainer_name: str, 
    #                        explainer_params: Dict) -> Dict[str, Any]:
    #     """Evaluate all metrics for a single explainer."""
    #     explainer_results = {}
    #     cfg_xai = self.config.cfg_xai
        
    #     plot_first_only = True  # Only plot explanations for first metric
        
    #     for metric_type, metrics_dict in cfg_xai["compare"]["metrics"].items():
    #         for metric_name, metric_info in metrics_dict.items():
    #             metric_params = metric_info.get("params", {})
                
    #             evaluator = MetricEvaluator(
    #                 self.model, metric_type, metric_name,
    #                 explainer_name, metric_params, explainer_params
    #             )
                
    #             result = evaluator.evaluate(
    #                 self.dataloader,
    #                 self.config.checkpoint,
    #                 self.config.cfg_model["model"],
    #                 self.config.cfg_dataset["data"],
    #                 cfg_xai,
    #                 self.dataset_info,
    #                 self.infer_ds,
    #                 cfg_xai["list_run_hour"],
    #                 cfg_xai["use_old_weights"],
    #                 cfg_xai["target"],
    #                 plot_explanations=plot_first_only and cfg_xai["explain"]["exp_plot"]["plot_explanations"],
    #                 extent=cfg_xai["extent"],
    #                 fig_path=cfg_xai["explain"]["exp_plot"]["fig_path"]
    #             )
                
    #             explainer_results[metric_name] = result
    #             plot_first_only = False  # Disable plotting for subsequent metrics
                
    #             torch.cuda.empty_cache()
        
    #     return explainer_results
    def _evaluate_explainer(self, explainer_name: str, 
                           explainer_params: Dict) -> Dict[str, Any]:
        explainer_results = {}
        cfg_xai = self.config.cfg_xai
        
        # Setup the grid
        extents = cfg_xai.get("extents", [cfg_xai.get("extent")])
        run_hours_list = cfg_xai.get("list_run_hours", [cfg_xai.get("list_run_hour")])
        targets = cfg_xai.get("targets", [cfg_xai.get("target")])

        plot_first_only = True 
        
        for metric_type, metrics_dict in cfg_xai["compare"]["metrics"].items():
            for metric_name, metric_info in metrics_dict.items():
                metric_params = metric_info.get("params", {})
                
                evaluator = MetricEvaluator(
                    self.model, metric_type, metric_name,
                    explainer_name, metric_params, explainer_params
                )

                all_vals = []
                # thresh_list = [2.1907,0.9903,2.1176]
                # thresh_list = [2.1907] #top25
                thresh_list = [-210000000.1907] #top25

#                thresh_list = [2.1907]
                # Iterate over Targets, Regions, and Input-Output pairs
                for (target,precip_thresh) in zip(targets,thresh_list):
                    for extent in extents:
                        # for run_hour in run_hours_list:
                            
                            self.logger.info(f"Evaluating {metric_name} | Target: {target} | Extent: {extent} | thresh: {(precip_thresh)}")

                            do_plot = plot_first_only and cfg_xai["explain"]["exp_plot"]["plot_explanations"]
                            cfg_xai["target"] = target
                            result = evaluator.evaluate(
                                self.dataloader,
                                self.config.checkpoint,
                                self.config.cfg_model["model"],
                                self.config.cfg_dataset["data"],
                                cfg_xai,
                                self.dataset_info,
                                self.infer_ds,
                                cfg_xai["list_run_hour"],
                                cfg_xai["use_old_weights"],
                                target,
                                plot_explanations=do_plot,
                                extent=extent,
                                fig_path=cfg_xai["explain"]["exp_plot"]["fig_path"],precip_thresh=precip_thresh
                            )
                            
                            # ---------------------------------------------------------
                            # NEW LOGIC: Safely get values. If 'vals' is missing or NaN,
                            # fall back to the pre-computed 'mean' from the metric.
                            # ---------------------------------------------------------
                            v = result.get("vals", None)
                            
                            # Check if v is literally the scalar float 'NaN'
                            is_scalar_nan = isinstance(v, (float, np.floating)) and np.isnan(v)
                            
                            if v is not None and not is_scalar_nan:
                                all_vals.append(v)
                            elif result.get("mean") is not None:
                                # Metric didn't return raw vals, use its internal mean instead
                                all_vals.append(result["mean"])
                            
                            plot_first_only = False 
                            torch.cuda.empty_cache()
                
                # Recursively aggregate statistics
                if all_vals:
                    vals, mean, std, sem = self._aggregate_stats(all_vals)
                    explainer_results[metric_name] = {
                        "vals": vals,
                        "mean": mean,
                        "std": std,
                        "sem": sem
                    }
                else:
                    explainer_results[metric_name] = {"vals": [], "mean": [], "std": [], "sem": []}
        
        return explainer_results
    
    def _evaluate_explainer_in_out(self, explainer_name: str, 
                           explainer_params: Dict) -> Dict[str, Any]:
        explainer_results = {}
        cfg_xai = self.config.cfg_xai
        
        # Setup the grid

        targets = cfg_xai.get("targets", [cfg_xai.get("target")])
        inputs = cfg_xai["explain"].get("inputs", [cfg_xai.get("input")])
        outputs = cfg_xai["explain"].get("outputs", [cfg_xai.get("outputs")])
        plot_first_only = True 
        
        for metric_type, metrics_dict in cfg_xai["compare"]["metrics"].items():
            for metric_name, metric_info in metrics_dict.items():
                metric_params = metric_info.get("params", {})
                
                evaluator = MetricEvaluator(
                    self.model, metric_type, metric_name,
                    explainer_name, metric_params, explainer_params
                )

                all_vals = []
                
                # Iterate over Targets, Regions, and Input-Output pairs
                for input in inputs:
                    for output in outputs:
                        # for run_hour in run_hours_list:
                            self.logger.info(f"Evaluating {metric_name} | input: {input} | output: {output}")

                            do_plot = plot_first_only and cfg_xai["explain"]["exp_plot"]["plot_explanations"]
                            cfg_xai["explain"]["input"] = input
                            cfg_xai["explain"]["output"] = output
                            result = evaluator.evaluate(
                                self.dataloader,
                                self.config.checkpoint,
                                self.config.cfg_model["model"],
                                self.config.cfg_dataset["data"],
                                cfg_xai,
                                self.dataset_info,
                                self.infer_ds,
                                cfg_xai["list_run_hour"],
                                cfg_xai["use_old_weights"],
                                cfg_xai["target"],
                                plot_explanations=do_plot,
                                extent=cfg_xai["extent"],
                                fig_path=cfg_xai["explain"]["exp_plot"]["fig_path"],precip_thresh=2.1907
                            )
                            
                            # ---------------------------------------------------------
                            # NEW LOGIC: Safely get values. If 'vals' is missing or NaN,
                            # fall back to the pre-computed 'mean' from the metric.
                            # ---------------------------------------------------------
                            v = result.get("vals", None)
                            
                            # Check if v is literally the scalar float 'NaN'
                            is_scalar_nan = isinstance(v, (float, np.floating)) and np.isnan(v)
                            
                            if v is not None and not is_scalar_nan:
                                all_vals.append(v)
                            elif result.get("mean") is not None:
                                # Metric didn't return raw vals, use its internal mean instead
                                all_vals.append(result["mean"])
                            
                            plot_first_only = False 
                            torch.cuda.empty_cache()
                
                # Recursively aggregate statistics
                if all_vals:
                    vals, mean, std, sem = self._aggregate_stats(all_vals)
                    explainer_results[metric_name] = {
                        "vals": vals,
                        "mean": mean,
                        "std": std,
                        "sem": sem
                    }
                else:
                    explainer_results[metric_name] = {"vals": [], "mean": [], "std": [], "sem": []}
        
        return explainer_results

    def _compute_numpy_stats(self, items):
        """Computes statistics only on pure numerical arrays safely."""
        try:
            arr = np.array(items, dtype=float)
        except (TypeError, ValueError) as e:
            # If an unparseable object reaches here, print a warning so you know EXACTLY what it is
            print(f"\n[DEBUG] Could not convert metric values to float array: {e}.")
            print(f"[DEBUG] Data structure received: {type(items)}. First element: {items[0] if items else 'Empty'}")
            return items, np.nan, np.nan, np.nan
            
        # Ignore warning for "Mean of empty slice" if array happens to be empty
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=RuntimeWarning)
            m = np.nanmean(arr, axis=0)
            s = np.nanstd(arr, axis=0)
            se = stats.sem(arr, axis=0, nan_policy='omit')
        
        # Handle masked arrays returned by scipy if all values are NaN or length is 1
        if np.isscalar(se) and np.ma.is_masked(se): 
            se = 0.0
        elif isinstance(se, np.ma.MaskedArray):
            se = se.filled(0.0)
            
        return (
            arr.tolist(),
            m.tolist() if isinstance(m, np.ndarray) else float(m),
            s.tolist() if isinstance(s, np.ndarray) else float(s),
            se.tolist() if isinstance(se, np.ndarray) else float(se)
        )
    def _generate_plots(self):
        """Generate all evaluation plots."""
        from .plot_utils import (
            plot_boxplot, plot_bar_ROAD
        )
        import os
        
        cfg_xai = self.config.cfg_xai
        num_steps = self.config.cfg_dataset["data"]["num_pred_steps_val_test"]
        save_figs = cfg_xai["eval"]["eval_plot"]["save_figs"]
        base_path = cfg_xai["eval"]["eval_plot"]["fig_path"]
        
        os.makedirs(base_path, exist_ok=True)
        
        for metrics_dict in cfg_xai["compare"]["metrics"].values():
            for metric_name in metrics_dict.keys():
                metric_path = os.path.join(base_path, f"{metric_name}_comparison")
                os.makedirs(metric_path, exist_ok=True)
                
                # if metric_name == "ROAD":
                #     plot_bar_ROAD(
                #         self.results, metric_name, metric_path,
                #         num_steps=num_steps, save_figs=save_figs
                #     )
                # else:
                plot_boxplot(
                    self.results, metric_name, metric_path,
                    num_steps=num_steps, save_figs=save_figs
                )
        
        self.logger.info(f"Evaluation plots saved to {base_path}")
    def _aggregate_stats(self, items):
        """Recursively drills down lists/dicts/tuples to aggregate numbers properly."""
        if not items:
            return [], [], [], []
            
        first = items[0]
        
        # 1. Dictionary case: process each key separately
        if isinstance(first, dict):
            res_vals, res_mean, res_std, res_sem = {}, {}, {}, {}
            for k in first.keys():
                vals_k = [item[k] for item in items if isinstance(item, dict) and k in item]
                
                # FIX: For dictionaries, the lists inside are batches of runs. 
                # We concatenate all batches across all loops (targets/regions) into 
                # one continuous list of independent runs.
                pooled = []
                for v in vals_k:
                    if isinstance(v, (list, tuple, np.ndarray)):
                        pooled.extend(v)  # Flatten the batch dimension into the pool
                    elif v is not None and not (isinstance(v, float) and np.isnan(v)):
                        pooled.append(v)
                        
                # Now compute the global mean and SEM over the pooled runs
                v_res, m_res, s_res, se_res = self._compute_numpy_stats(pooled)
                res_vals[k] = v_res
                res_mean[k] = m_res
                res_std[k] = s_res
                res_sem[k] = se_res
                
            return res_vals, res_mean, res_std, res_sem
            
        # 2. Iterable case (lists, tuples, arrays)
        elif isinstance(first, (list, tuple, np.ndarray)):
            try:
                np.array(items, dtype=float)
                return self._compute_numpy_stats(items)
            except (TypeError, ValueError):
                # Contains dicts or uneven shapes. Transpose and process element by element.
                try:
                    transposed = list(zip(*items))
                except TypeError:
                    return self._compute_numpy_stats(items)
                    
                v_list, m_list, s_list, se_list = [], [], [], []
                for t_items in transposed:
                    v, m, s, se = self._aggregate_stats(list(t_items))
                    v_list.append(v); m_list.append(m); s_list.append(s); se_list.append(se)
                return v_list, m_list, s_list, se_list
                
        # 3. Base scalar case
        else:
            return self._compute_numpy_stats(items)

    # def _compute_numpy_stats(self, items):
    #     """Computes statistics only on pure numerical arrays safely."""
    #     try:
    #         arr = np.array(items, dtype=float)
    #     except (TypeError, ValueError):
    #         # Absolute fallback: if an unparseable object reaches here, don't crash
    #         return items, np.nan, np.nan, np.nan
            
    #     m = np.nanmean(arr, axis=0)
    #     s = np.nanstd(arr, axis=0)
    #     se = stats.sem(arr, axis=0, nan_policy='omit')
        
    #     # Handle masked arrays returned by scipy if all values are NaN or length is 1
    #     if np.isscalar(se) and np.ma.is_masked(se): 
    #         se = 0.0
    #     elif isinstance(se, np.ma.MaskedArray):
    #         se = se.filled(0.0)
            
    #     return (
    #         arr.tolist(),
    #         m.tolist() if isinstance(m, np.ndarray) else float(m),
    #         s.tolist() if isinstance(s, np.ndarray) else float(s),
    #         se.tolist() if isinstance(se, np.ndarray) else float(se)
    #     )
class MetricAggregator:
    
    @staticmethod
    def _format_value(name, mean, sem, indent=2):
        """Recursively formats metric outputs for the summary table."""
        lines = []
        prefix = " " * indent
        
        if isinstance(mean, dict):
            lines.append(f"{prefix}{name}:")
            for k, v in mean.items():
                s = sem.get(k, None) if sem and isinstance(sem, dict) else None
                lines.extend(MetricAggregator._format_value(k, v, s, indent + 2))
                
        elif isinstance(mean, list):
            # Check if it's a list containing nested dictionaries
            if len(mean) > 0 and isinstance(mean[0], dict):
                lines.append(f"{prefix}{name}:")
                for i, (m, s) in enumerate(zip(mean, sem)):
                    lines.extend(MetricAggregator._format_value(f"Step {i}", m, s, indent + 2))
            else:
                m_str = ", ".join([f"{v:.4f}" for v in mean])
                if sem is not None:
                    s_str = ", ".join([f"{v:.4f}" for v in sem])
                    lines.append(f"{prefix}{name}: Mean [{m_str}] ± SEM [{s_str}]")
                else:
                    lines.append(f"{prefix}{name}: [{m_str}]")
                    
        else:
            if sem is not None:
                lines.append(f"{prefix}{name}: {mean:.4f} ± {sem:.4f} (SEM)")
            else:
                lines.append(f"{prefix}{name}: {mean:.4f}")
                
        return lines

    @staticmethod
    def create_summary_table(results: Dict[str, Dict[str, Any]]) -> str:
        """Create a formatted summary table of results."""
        lines = ["=" * 80, "Evaluation Summary", "=" * 80]
        
        for explainer_name, metrics in results.items():
            lines.append(f"\n{explainer_name}:")
            lines.append("-" * 40)
            
            for metric_name, metric_data in metrics.items():
                if "mean" in metric_data:
                    formatted = MetricAggregator._format_value(
                        metric_name, 
                        metric_data["mean"], 
                        metric_data.get("sem")
                    )
                    lines.extend(formatted)
                    
        lines.append("=" * 80)
        return "\n".join(lines)
# class MetricAggregator:
#     """Aggregates and compares metrics across explainers."""
    
#     @staticmethod
#     def create_summary_table(results: Dict[str, Dict[str, Any]]) -> str:
#         """Create a formatted summary table of results."""
#         lines = ["=" * 80]
#         lines.append("Evaluation Summary")
#         lines.append("=" * 80)
        
#         for explainer_name, metrics in results.items():
#             lines.append(f"\n{explainer_name}:")
#             lines.append("-" * 40)
            
#             for metric_name, metric_data in metrics.items():
#                 if "mean" in metric_data:
#                     mean_val = metric_data["mean"]
#                     if isinstance(mean_val, list):
#                         mean_str = ", ".join([f"{v:.4f}" for v in mean_val])
#                         lines.append(f"  {metric_name}: [{mean_str}]")
#                     else:
#                         lines.append(f"  {metric_name}: {mean_val:.4f}")
#                 elif "auc" in metric_data:
#                     auc_val = metric_data["auc"]
#                     if isinstance(auc_val, list):
#                         auc_str = ", ".join([f"{v:.4f}" for v in auc_val])
#                         lines.append(f"  {metric_name} AUC: [{auc_str}]")
#                     else:
#                         lines.append(f"  {metric_name} AUC: {auc_val:.4f}")
        
#         lines.append("=" * 80)
#         return "\n".join(lines)
    
#     @staticmethod
#     def rank_explainers(results: Dict[str, Dict[str, Any]], 
#                        metric_name: str, 
#                        lower_is_better: bool = True) -> List[tuple]:
#         """Rank explainers by a specific metric."""
#         scores = []
        
#         for explainer_name, metrics in results.items():
#             if metric_name not in metrics:
#                 continue
            
#             metric_data = metrics[metric_name]
#             if "mean" in metric_data:
#                 # Average across steps if multiple
#                 score = metric_data["mean"]
#                 if isinstance(score, list):
#                     score = sum(score) / len(score)
#             elif "auc" in metric_data:
#                 score = metric_data["auc"]
#                 if isinstance(score, list):
#                     score = sum(score) / len(score)
#             else:
#                 continue
            
#             scores.append((explainer_name, score))
        
#         # Sort by score
#         scores.sort(key=lambda x: x[1], reverse=not lower_is_better)
#         return scores