"""Quantitative evaluation (Table 1): every configured metric for every explainer."""

import logging

import numpy as np

from .metrics import metrics_registry

logger = logging.getLogger(__name__)


def summarize(scores):
    """Mean and standard error (std / sqrt(n)) over events, as reported in the paper."""
    scores = np.asarray(scores, dtype=float)
    n = len(scores)
    if n == 0:
        return {"mean": float("nan"), "sem": float("nan"), "n": 0, "scores": []}
    return {
        "mean": float(np.nanmean(scores)),
        "sem": float(np.nanstd(scores) / np.sqrt(n)),
        "n": n,
        "scores": scores.tolist(),
    }


def evaluate(model, explainers, metrics, events, ctx, seed=42):
    """Score every explainer with every metric.

    Args:
        explainers: ``{name: Explainer}``.
        metrics: ``{metric name: keyword arguments}``, see ``metrics_registry``.
        events: callable returning a fresh iterable of ``(batch_idx, runtime, batch)``.

    Returns:
        ``{explainer: {score: {"mean", "sem", "n", "scores"}}}``.
    """
    results = {}
    for name, explainer in explainers.items():
        results[name] = {}
        for metric_name, params in metrics.items():
            logger.info("Evaluating %s with %s", name, metric_name)
            metric = metrics_registry[metric_name](
                model, explainer, **{"seed": seed, **(params or {})}
            )
            for score_name, scores in metric.evaluate(events(), ctx).items():
                results[name][score_name] = summarize(scores)
    return results


def format_results(results):
    """One row per explainer and one ``mean ± SEM`` column per score."""
    columns = list(dict.fromkeys(score for scores in results.values() for score in scores))
    lines = ["".join(f"{c:<22}" for c in ["Explainer", *columns])]
    for name, scores in results.items():
        cells = [
            f"{scores[c]['mean']:.4g} ± {scores[c]['sem']:.2g}" if c in scores else "-"
            for c in columns
        ]
        lines.append("".join(f"{c:<22}" for c in [name, *cells]))
    return "\n".join(lines)
