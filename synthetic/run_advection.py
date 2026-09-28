"""Toy experiment: SmoothGrad and WassersteinGrad on an extrapolation nowcast.

    python -m synthetic.run_advection

writes the three figures and ``metrics.json`` to ``results/``. For every event,
lead time and noise level, SmoothGrad and WassersteinGrad aggregate the same
perturbed gradients, and every noise level uses the same standard normal draws
scaled by the noise level (common random numbers).
"""

import argparse
import json
import math
from pathlib import Path

import torch

from . import metrics, plots
from .advection import Event, Nowcast
from .methods import base_grad, noisy_gradients, wasserstein_barycenter

NOISE_LEVELS = (0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4)  # alpha: noise std / input range
LEADS = (6, 24)  # steps
N_SAMPLES, REG = 20, 1e-3  # hyperparameters of the paper
FIGURE_LEVEL, COMPARISON_LEVELS = 0.2, (0.1, 0.2, 0.4)
DEMO_EVENT = Event(source=(32.0, 32.0), velocity=(0.8, 0.35), radius=3.0)


def sample_events(num, seed):
    """Cells near the domain centre, with random sub-pixel position, radius, speed and direction."""
    generator = torch.Generator().manual_seed(seed)

    def uniform(low, high):
        return low + (high - low) * torch.rand((), generator=generator, dtype=torch.float64).item()

    events = []
    for _ in range(num):
        speed, angle = uniform(0.5, 1.0), uniform(0, 2 * math.pi)
        events.append(Event(source=(uniform(31, 33), uniform(31, 33)),
                            velocity=(speed * math.cos(angle), speed * math.sin(angle)),
                            radius=uniform(2.5, 4.0)))
    return events


def explain(model, event, noise_level, seed):
    """Clean gradient, perturbed gradients, SmoothGrad and WassersteinGrad maps of one event."""
    f, x = model.target(event), model.observation(event)
    gradients = noisy_gradients(f, x, noise_level, N_SAMPLES, torch.Generator().manual_seed(seed))
    maps = {"SmoothGrad": gradients.mean(dim=0), "WassersteinGrad": wasserstein_barycenter(gradients, REG)}
    return base_grad(f, x), gradients, maps


def scores(model, event, clean, gradients, maps):
    """Metrics of one event, in pixels."""
    source, centre = model.source_centre(event), metrics.centroid(clean)
    displacements = [metrics.distance(metrics.centroid(g), centre, g.shape) for g in gradients]
    out = {"perturbed": {"displacement": sum(displacements) / len(displacements)}}
    for name, a in {"clean": clean, **maps}.items():
        out[name] = {"centroid_error": metrics.centroid_error(a, source),
                     "peak_error": metrics.peak_error(a, source),
                     "rms_radius": metrics.rms_radius(a, source),
                     "gini": metrics.gini(a)}
    return out


def summarize(per_event):
    """Mean and standard error over events of every metric: {group: {metric: {mean, sem}}}."""
    summary = {}
    for group, group_scores in per_event[0].items():
        summary[group] = {}
        for metric in group_scores:
            values = torch.tensor([event_scores[group][metric] for event_scores in per_event], dtype=torch.float64)
            summary[group][metric] = (values.mean().item(), (values.std() / math.sqrt(len(values))).item())
    return summary


def run(num_events, seed, out_dir):
    torch.use_deterministic_algorithms(True)
    out_dir.mkdir(parents=True, exist_ok=True)
    events = sample_events(num_events, seed)
    results = {"config": {"n": 64, "leads": LEADS, "noise_levels": NOISE_LEVELS, "n_samples": N_SAMPLES,
                          "reg": REG, "num_events": num_events, "seed": seed},
               "summary": {}, "per_event": {}}
    for lead in LEADS:
        model = Nowcast(lead=lead)
        by_level = {}
        for level in NOISE_LEVELS:
            per_event = [scores(model, event, *explain(model, event, level, seed + i))
                         for i, event in enumerate(events)]
            by_level[level] = summarize(per_event)
            results["per_event"][f"lead={lead},alpha={level}"] = per_event
            print(f"lead {lead:2d}  alpha {level:.2f}  " + "  ".join(
                f"{name} {by_level[level][group][metric][0]:5.2f}" for name, group, metric in (
                    ("displacement", "perturbed", "displacement"),
                    ("SG/WG centroid", "SmoothGrad", "centroid_error"),
                    ("/", "WassersteinGrad", "centroid_error"),
                    ("peak", "SmoothGrad", "peak_error"),
                    ("/", "WassersteinGrad", "peak_error"),
                    ("rms", "SmoothGrad", "rms_radius"),
                    ("/", "WassersteinGrad", "rms_radius"))))
        results["summary"][str(lead)] = _by_metric(by_level)
    (out_dir / "metrics.json").write_text(json.dumps(results, indent=1))
    plots.plot_metrics(results["summary"], out_dir / "metrics_vs_noise.png")

    model = Nowcast(lead=LEADS[-1])
    clean, gradients, _ = explain(model, DEMO_EVENT, FIGURE_LEVEL, seed)
    motion = model.estimate_motion(model.observation(DEMO_EVENT)[None], model.observation(DEMO_EVENT, time=-1))
    forecast = model.extrapolate(model.observation(DEMO_EVENT)[None], motion)[0]
    plots.plot_setup(model, DEMO_EVENT, forecast, clean, gradients, FIGURE_LEVEL,
                     out_dir / "perturbed_gradients.png")
    maps = {level: explain(model, DEMO_EVENT, level, seed)[2] for level in COMPARISON_LEVELS}
    plots.plot_comparison(model, DEMO_EVENT, clean, maps, N_SAMPLES, out_dir / "smoothgrad_vs_wassersteingrad.png")
    print(f"Results written to {out_dir}/")


def _by_metric(by_level):
    """{alpha: {group: {metric: (mean, sem)}}} -> {group: {metric: {mean: [...], sem: [...]}}}."""
    levels = list(by_level)
    first = by_level[levels[0]]
    out = {"noise_levels": levels}
    for group in first:
        out[group] = {metric: {"mean": [by_level[a][group][metric][0] for a in levels],
                               "sem": [by_level[a][group][metric][1] for a in levels]}
                      for metric in first[group]}
    return out


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--events", type=int, default=24, help="number of random events")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", type=Path, default=Path("results"))
    args = parser.parse_args()
    run(args.events, args.seed, args.out)
