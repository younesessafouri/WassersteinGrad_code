"""Loading of the Py4Cast assets and experiment modes (see ``ExperimentRunner.run``)."""

import importlib
import json
import logging
import os
import sys
import time

import numpy as np
import torch
import yaml

from . import evaluation, plotting
from .displacement import aggregate_displacement, displacement_curves, transport_plan
from .explainers import BaseGrad, explainers_registry
from .inference import ForecastContext, batch_to_device, event_intensity, predict_step

logger = logging.getLogger(__name__)


def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)


def load_yaml(path):
    with open(path) as f:
        return yaml.safe_load(f)


def load_checkpoint(path):
    """Py4Cast Lightning checkpoint: weights, output feature/dim names and dtype."""
    # Checkpoints pickled with older mfai versions refer to the ``mfai.torch`` module.
    sys.modules.setdefault("mfai.torch", importlib.import_module("mfai.pytorch"))
    return torch.load(path, map_location="cpu", weights_only=False)


def build_dataloader(cfg_data):
    """Test split of a Py4Cast dataset, one sample per batch."""
    from py4cast.datasets import get_datasets

    if cfg_data.get("batch_size", 1) != 1:
        raise ValueError("The explainers expect batch_size: 1 in the dataset config.")
    _, _, infer_ds = get_datasets(
        cfg_data["dataset_name"],
        cfg_data["num_input_steps"],
        cfg_data["num_pred_steps_train"],
        cfg_data["num_pred_steps_val_test"],
        cfg_data["dataset_conf"],
    )
    dataloader = infer_ds.torch_dataloader(
        batch_size=1,
        num_workers=cfg_data["num_workers"],
        shuffle=False,
        prefetch_factor=cfg_data["prefetch_factor"],
        pin_memory=cfg_data["pin_memory"],
    )
    return dataloader, infer_ds


def build_model(cfg_model, cfg_data, dataset_info, checkpoint, weights_path=None):
    """Py4Cast model with the checkpoint weights, optionally replaced by a state dict."""
    from py4cast.models import build_model_from_settings

    statics = dataset_info.statics
    num_input_features = (
        cfg_data["num_input_steps"] * dataset_info.weather_dim
        + statics.grid_statics.dim_size("features")
        + dataset_info.forcing_dim
    )
    model, _ = build_model_from_settings(
        cfg_model["model_name"],
        num_input_features,
        dataset_info.weather_dim,
        cfg_model["settings_init_args"],
        statics.grid_shape,
    )
    state_dict = checkpoint["state_dict"]
    model.load_state_dict({k.replace("model.", ""): v for k, v in state_dict.items()}, strict=False)
    if weights_path:
        weights = torch.load(weights_path, map_location="cpu")
        model.load_state_dict({k.replace("model.", ""): v for k, v in weights.items()})
    return model.eval()


class ExperimentRunner:
    """Loads the Py4Cast assets referenced in the config and runs one mode."""

    def __init__(self, config_path, mode=None):
        self.cfg = load_yaml(config_path)
        self.mode = mode or self.cfg["mode"]
        self.seed = self.cfg.get("seed", 42)
        self.output_dir = self.cfg.get("output_dir", "outputs")
        self.run_hours = self.cfg.get("list_run_hour", list(range(24)))
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        cfg_data = load_yaml(self.cfg["dataset_path"])["data"]
        cfg_model = load_yaml(self.cfg["model_path"])["model"]
        checkpoint = load_checkpoint(self.cfg["ckpt_path"])
        self.dataloader, self.infer_ds = build_dataloader(cfg_data)
        dataset_info = self.infer_ds.dataset_info
        self.model = build_model(
            cfg_model, cfg_data, dataset_info, checkpoint, self.cfg.get("weights_path")
        ).to(self.device)
        self.ctx = ForecastContext(
            cfg_model=cfg_model,
            dataset_info=dataset_info,
            output_feature_names=checkpoint["output_feature_names"],
            output_dim_names=checkpoint["output_dim_names"],
            output_dtype=checkpoint["output_dtype"],
            input_name=self.cfg["input"],
            output_name=self.cfg["output"],
            target=self.cfg.get("target"),
            extent=self.cfg.get("extent") or dataset_info.domain_info.grid_limits,
        )

    def run(self):
        modes = {
            "infer": self.infer,
            "explain": self.explain,
            "compare": self.compare,
            "eval": self.evaluate,
            "time": self.benchmark,
            "centroid": self.centroid,
            "transport": self.transport,
            "precip": self.precip,
        }
        if self.mode not in modes:
            raise ValueError(f"Unknown mode {self.mode!r}, expected one of {', '.join(modes)}.")
        set_seed(self.seed)
        logger.info("Running mode %s", self.mode)
        return modes[self.mode]()

    # -- events and explainers -------------------------------------------------

    def _runtime(self, batch_idx):
        """Analysis time of a sample, or None if its hour is not in ``list_run_hour``."""
        date = self.infer_ds.sample_list[batch_idx].timestamps.datetime
        return date.strftime("%Y%m%d%H") if date.hour in self.run_hours else None

    def events(self, precip_threshold=None):
        """Yield ``(batch_idx, runtime, batch)`` for the selected events, on the device.

        An event is kept if its analysis hour is in ``list_run_hour`` and, when a
        threshold is given, if its mean ground truth of the explained output over
        the ROI at the first lead time reaches the threshold.
        """
        for batch_idx, batch in enumerate(self.dataloader):
            runtime = self._runtime(batch_idx)
            if runtime is None:
                continue
            if precip_threshold is not None and event_intensity(batch, self.ctx) < precip_threshold:
                continue
            yield batch_idx, runtime, batch_to_device(batch, self.device)

    def explainers(self):
        return {
            name: explainers_registry[name](self.model, **(params or {}))
            for name, params in self.cfg["explainers"].items()
        }

    def _fields(self, batch, prediction):
        """De-normalised input channel, and ground truth and forecast at the last lead time."""
        stats = self.ctx.dataset_info.stats

        def physical(x, name):
            x = x.detach().cpu()
            return (
                x * torch.as_tensor(stats[name]["std"]) + torch.as_tensor(stats[name]["mean"])
            ).numpy()

        input_idx = batch.inputs.feature_names_to_idx[self.ctx.input_name]
        output_idx = batch.outputs.feature_names_to_idx[self.ctx.output_name]
        return (
            physical(batch.inputs.tensor[0, 0, ..., input_idx], self.ctx.input_name),
            physical(batch.outputs.tensor[0, -1, ..., output_idx], self.ctx.output_name),
            prediction.tensor[0, -1, ..., output_idx].cpu().numpy(),
        )

    # -- modes -----------------------------------------------------------------

    def infer(self):
        """Forecast every event; GIFs are saved by Py4Cast if ``io_conf`` is set."""
        io_conf = self.ctx.cfg_model.get("io_conf")
        if io_conf:
            from py4cast.io.outputs import OutputSavingSettings, save_gifs

            with open(io_conf) as f:
                save_settings = OutputSavingSettings(**json.load(f))
        for _, runtime, batch in self.events():
            prediction, _ = predict_step(self.model, batch, self.ctx)
            logger.info("Forecast %s", runtime)
            if io_conf:
                for sample in prediction.iter_dim("batch"):
                    save_gifs(sample, runtime, self.infer_ds.grid, save_settings)

    def explain(self):
        """One figure per event and explainer."""
        for name, explainer in self.explainers().items():
            for _, runtime, batch in self.events():
                attribution, prediction = explainer.explain(batch, self.ctx)
                title = (
                    f"{name}: {self.ctx.output_name} (t+{batch.num_pred_steps}) "
                    f"w.r.t. {self.ctx.input_name}, {runtime}"
                )
                plotting.plot_attribution(
                    attribution,
                    *self._fields(batch, prediction),
                    self.ctx.extent,
                    self.ctx.target,
                    title,
                    os.path.join(self.output_dir, "explain", name, f"{runtime}.png"),
                )

    def compare(self):
        """Attribution maps of all explainers for the first event (Fig. 2)."""
        event = next(self.events(), None)
        if event is None:
            raise RuntimeError("No event selected.")
        _, runtime, batch = event
        attributions = {
            name: explainer.explain(batch, self.ctx)[0]
            for name, explainer in self.explainers().items()
        }
        plotting.plot_comparison(
            attributions,
            self.ctx.extent,
            self.ctx.target,
            os.path.join(self.output_dir, "compare", runtime),
        )

    def evaluate(self):
        """All metrics for all explainers (Table 1); scores saved to ``eval.json``."""
        cfg = self.cfg["eval"]
        threshold = cfg.get("precip_threshold")
        results = evaluation.evaluate(
            self.model,
            self.explainers(),
            cfg["metrics"],
            lambda: self.events(threshold),
            self.ctx,
            self.seed,
        )
        print(evaluation.format_results(results))
        os.makedirs(self.output_dir, exist_ok=True)
        with open(os.path.join(self.output_dir, "eval.json"), "w") as f:
            json.dump(results, f, indent=2)
        return results

    def benchmark(self):
        """Wall-time per event of each explainer, excluding data loading (Table 6)."""
        for name, explainer in self.explainers().items():
            times = []
            for _, _, batch in self.events():
                if torch.cuda.is_available():
                    torch.cuda.synchronize()
                start = time.perf_counter()
                explainer.explain(batch, self.ctx)
                if torch.cuda.is_available():
                    torch.cuda.synchronize()
                times.append(time.perf_counter() - start)
            logger.info(
                "%s: %.3f s ± %.3f s per event (n=%d)",
                name,
                np.mean(times),
                np.std(times),
                len(times),
            )

    def centroid(self):
        """Centroid and peak displacement of gradients under input noise (Fig. 4)."""
        cfg = self.cfg.get("centroid", {})
        max_events = cfg.get("max_events", 100)
        cache_dir = os.path.join(self.output_dir, "centroid", "cache")
        os.makedirs(cache_dir, exist_ok=True)
        results = []
        for batch_idx, runtime, batch in self.events(cfg.get("precip_threshold")):
            cache = os.path.join(cache_dir, f"event_{batch_idx:04d}.npz")
            if os.path.exists(cache) and not cfg.get("force_recompute", False):
                results.append(dict(np.load(cache)))
            else:
                result = displacement_curves(
                    self.model,
                    batch,
                    self.ctx,
                    sigma_step=cfg.get("sigma_step", 0.1),
                    num_noise_levels=cfg.get("num_noise_levels", 10),
                    num_monte_carlo=cfg.get("num_monte_carlo", 10),
                )
                np.savez(cache, **result)
                results.append(result)
            logger.info("Event %s done (%d/%d)", runtime, len(results), max_events)
            if len(results) == max_events:
                break
        if not results:
            raise RuntimeError("No event selected.")
        summary = aggregate_displacement(results)
        plotting.plot_displacement(
            summary, os.path.join(self.output_dir, "centroid", "displacement.pdf")
        )
        return summary

    def transport(self):
        """OT mass transport between clean and noisy gradients (Fig. 1c)."""
        cfg = self.cfg.get("transport", {})
        explainer = BaseGrad(self.model)
        for _, runtime, batch in self.events():
            clean, _ = explainer.gradient(batch, self.ctx)
            clean = clean.detach().cpu().numpy()
            noisy_gradients = explainer.perturbed_gradients(
                batch, self.ctx, cfg.get("std_perturbations", 0.2), cfg.get("num_perturbations", 1)
            )
            for k, noisy in enumerate(noisy_gradients):
                noisy = noisy.detach().cpu().numpy()
                plan, grid_shape = transport_plan(clean, noisy, scale=cfg.get("scale", 16))
                plotting.plot_transport(
                    clean,
                    noisy,
                    plan,
                    grid_shape,
                    self.ctx.extent,
                    self.ctx.target,
                    os.path.join(self.output_dir, "transport", f"{runtime}_{k}.pdf"),
                )

    def precip(self):
        """Rank the events by ROI intensity (see ``event_intensity``) to set thresholds."""
        records = []
        for batch_idx, batch in enumerate(self.dataloader):
            runtime = self._runtime(batch_idx)
            if runtime is not None:
                records.append((event_intensity(batch, self.ctx), runtime))
        records.sort(reverse=True)
        top_k = self.cfg.get("precip", {}).get("top_k", 100)
        print(f"{len(records)} events; top {top_k} by mean {self.ctx.output_name} over the ROI:")
        for rank, (value, runtime) in enumerate(records[:top_k], 1):
            print(f"{rank:4d}  {runtime}  {value:.6f}")
        return records


def run_experiment(config_path, mode=None):
    return ExperimentRunner(config_path, mode).run()
