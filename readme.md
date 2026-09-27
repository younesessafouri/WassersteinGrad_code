# Py4cast-XAI

**Explainability & Evaluation Framework for Weather Forecasting Models**

---

##  Overview

**Py4cast-XAI** is a modular framework designed to generate and evaluate explainability maps (XAI) for numerical weather prediction models built using **PyTorch** and the **Py4cast** ecosystem.

It supports:

* **Inference** — Run model predictions on meteorological datasets.
*  **Explainability** — Generate attribution maps using different gradient-based explainers.
*  **Evaluation** — Quantitatively assess explanations with faithfulness, robustness, and complexity metrics.





---

## Configuration System

All experiments are controlled via **`config/xai.yaml`**, which links to every other configuration file:

---

## How to Run

### 1 Inference Mode

Run forward prediction only:

```bash
python py4castxai/main.py --config config/xai.yaml
```

Set in `xai.yaml`:

```yaml
mode: inference
```

---

### 2 Explain Mode

Generate and visualize attribution maps for selected explainers:

```bash
python py4castxai/main.py --config config/xai.yaml
```

Set in `xai.yaml`:

```yaml
mode: explain
```

---

### 3 Evaluation Mode

Compute and compare evaluation metrics (faithfulness, robustness, sparsity) across explainers:

```bash
python py4castxai/main.py --config config/xai.yaml
```

Set in `xai.yaml`:

```yaml
mode: eval
```

Each metric will produce if you choose to:

*  ROAD curves
*  Sparsity & robustness bar plots 

---

## Implemented Metrics

**Faithfulness** :
    - ROAD: Perturbs top-k regions and measures output degradation     

**Robustness** :
    - Local Lipschitz Estimate: Measures stability of attributions under input noise 


**Complexity**:
    - Sparsness (Gini): Quantifies concentration of attribution scores 

---

##  Explainability Methods

| Method         | Description                                                                    |
| -------------- | ------------------------------------------------------------------------------ |
| **BaseGrad**   | Raw gradient saliency                                                          |
| **Input×Grad** | Weighted gradient, highlights features contributing proportionally to output   |
| **SmoothGrad** | Smoothed gradient (averaged over noisy samples) for denoising attribution maps |

---

##  Adding New Components

To extend the framework:

* Add new **explainers** in `py4castxai/explainers/`
*  Add new **metrics** in the appropriate subfolder under `py4castxai/metrics/`
*  Register metrics in the `registry` dict for automatic loading

