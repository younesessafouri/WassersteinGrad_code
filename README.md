# Explanation of Dynamic Physical Field Predictions using WassersteinGrad: Application to Autoregressive Weather Forecasting

[![Project Page](https://img.shields.io/badge/Project-Website-blue)](https://younesessafouri.github.io/WassersteinGrad/)
[![arXiv](https://img.shields.io/badge/arXiv-2604.22580-b31b1b.svg)](https://arxiv.org/abs/2604.22580)
[![Branch](https://img.shields.io/badge/branch-py4cast-2ea44f)](#)
[![Companion Toy Experiments](https://img.shields.io/badge/companion-toy--experiments-orange)](https://github.com/younesessafouri/py4cast-xai/tree/toy-experiments)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

Code accompanying the paper "Explanation of Dynamic Physical Field Predictions using WassersteinGrad: Application to Autoregressive Weather Forecasting
" by **Younes Essafouri, Laure Raynaud, Luciano Drozda, and Laurent Risser**.
This branch contains the **Py4Cast-based implementation** used for the weather forecasting experiments in the paper.  
WassersteinGrad is a gradient-based explanation method designed for **dynamic physical fields**, where input perturbations often **displace** attribution maps rather than simply adding stationary noise.
 
> See also the [`toy-experiments`](https://github.com/younesessafouri/py4cast-xai/tree/toy-experiments) branch, which reproduces the key phenomenon of the paper in a small synthetic physical model.

---

## Overview
On dynamic physical fields, input noise does not add stationary noise to gradient
attribution maps: it displaces them. SmoothGrad, which averages the perturbed
gradients pixel by pixel, therefore blurs misaligned features. WassersteinGrad maps
each perturbed gradient to a probability measure on the grid and aggregates them
with their entropic Wasserstein barycenter (WGBary). The barycenter can also be used
as a spatial mask on the plain gradient to keep its sign (WGBary×Grad).

This repository contains the implementation used for the weather forecasting
experiments of the paper, built on [Py4Cast](https://github.com/meteofrance/py4cast).

## Links

- **Project page:** [younesessafouri.github.io/WassersteinGrad](https://younesessafouri.github.io/WassersteinGrad/)
- **Paper:** [arXiv:2604.22580](https://arxiv.org/abs/2604.22580)
- **Py4Cast:** [https://github.com/meteofrance/py4cast](https://github.com/meteofrance/py4cast)  
- **Companion branch:** [`toy-experiments`](https://github.com/younesessafouri/py4cast-xai/tree/toy-experiments)

## Installation

```bash
git clone https://github.com/younesessafouri/py4cast-xai.git
cd py4cast-xai
pip install -r requirements.txt
```

The forecasting experiments also need [Py4Cast](https://github.com/meteofrance/py4cast)
and [mfai](https://github.com/meteofrance/mfai), installed following the Py4Cast
instructions (Python ≥ 3.11). `wasserstein_barycenter` can be used without them.
The paper used PyTorch 2.5.1 with CUDA 12.1 and POT 0.9.5.

## Minimal usage

WassersteinGrad only changes how the gradients of the noisy inputs are aggregated,
so the aggregation step can be used with any model:

```python
from py4castxai.explainers import wasserstein_barycenter

# gradients: (N, H, W) tensor, gradients of the target at N noisy copies of the input
# gradient:  (H, W) tensor, gradient at the clean input
wg_bary = wasserstein_barycenter(gradients, reg=1e-3)  # WGBary
wg_bary_x_grad = gradient * wg_bary                    # WGBary×Grad
```

`reg` is the entropic regularisation λ. POT defines it on the grid rescaled to
[0, 1] × [0, 1], so its effect in pixels depends on the grid size; the paper uses
λ = 1e-3 on a 512 × 640 grid.

With the Py4Cast assets, the explainers return the attribution map and the forecast
of each event:

```python
from py4castxai.experiment import ExperimentRunner
from py4castxai.explainers import WassersteinGrad

runner = ExperimentRunner("configs/xai.yaml")
explainer = WassersteinGrad(runner.model, std_perturbations=0.2, num_perturbations=20, reg=1e-3)
for batch_idx, runtime, batch in runner.events():
    attribution, forecast = explainer.explain(batch, runner.ctx)
```

## Repository structure

```
main.py                    command-line entry point
configs/xai.yaml           experiment configuration, with the hyperparameters of the paper
py4castxai/
  explainers/
    wassersteingrad.py     WassersteinGrad (WGBary) and WassersteinGradMask (WGBary×Grad)
    smoothgrad.py          SmoothGrad
    basegrad.py            plain gradient
    vargrad.py             VarGrad
    integratedgrad.py      Integrated Gradients
    base.py                gradient and input perturbation code shared by the explainers
  metrics/                 Gini sparsity, ROAD faithfulness, local Lipschitz robustness
  inference.py             autoregressive Py4Cast rollout and gradients over the region of interest
  displacement.py          centroid/peak displacement and optimal transport of gradient mass
  evaluation.py            scores of every explainer with every metric
  plotting.py              figures
  experiment.py            loading of the Py4Cast assets and experiment modes
```

## Running the Py4Cast experiments

Set the paths of the Py4Cast dataset config, model config and checkpoint in
`configs/xai.yaml`, then run

```bash
python main.py configs/xai.yaml --mode <mode>
```

| Mode        | Computes                                                           | Paper               |
|-------------|--------------------------------------------------------------------|---------------------|
| `compare`   | attribution maps of all explainers for one event                   | Fig. 2              |
| `eval`      | sparsity, ROAD, LLE (ℓ2 and cosine): mean ± SEM over events        | Table 1, Tables 7–8 |
| `time`      | wall-time per event                                                | Table 6             |
| `centroid`  | centroid and peak displacement of the gradient against noise level | Fig. 4              |
| `transport` | optimal transport of gradient mass under input noise               | Fig. 1(c)           |
| `explain`   | input, truth, forecast and attribution maps for every event        |                     |
| `precip`    | events ranked by precipitation over the region of interest         |                     |
| `infer`     | forecasts only                                                     |                     |

In the paper, the forecasting model is a UNetRPP trained on TITAN, and the explained
quantity is the total precipitation (`aro_tp_0m`) over a box around Paris with
respect to the zonal wind at 250 hPa (`aro_u_250hpa`), on the 2023 test split. The
lead time is the number of prediction steps in the dataset config
(`num_pred_steps_val_test`: 1 for t+1, 5 for t+5). The evaluated events are selected
with `eval.precip_threshold`; the `precip` mode lists the events by decreasing
precipitation. Results are written to `outputs/`.

## Data and model availability

The experiments use Météo-France assets: the
[TITAN](https://huggingface.co/datasets/meteofrance/titan) dataset, derived from
AROME analyses, and a UNetRPP model trained with Py4Cast and validated by
meteorologists. Some of these data and model assets are restricted and cannot be
distributed with this repository, so the results of the paper cannot be reproduced
from it alone. The code documents the exact pipeline and can be run with other
Py4Cast datasets and models.

## Toy experiment

A fully reproducible experiment that does not depend on restricted assets is
available on the [`toy`](https://github.com/younesessafouri/py4cast-xai/tree/toy) branch.

## Citation

```bibtex
@article{essafouri2026wassersteingrad,
  title   = {Explanation of Dynamic Physical Field Predictions using {WassersteinGrad}:
             Application to Autoregressive Weather Forecasting},
  author  = {Essafouri, Younes and Raynaud, Laure and Drozda, Luciano and Risser, Laurent},
  journal = {arXiv preprint arXiv:2604.22580},
  year    = {2026}
}
```
