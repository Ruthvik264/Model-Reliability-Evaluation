# Model Reliability & Confidence Interval Evaluation

A lightweight Python framework for evaluating machine learning classification performance using rigorous statistical bounds (**Wald vs. Wilson Score Intervals**), designed for small-batch test automation and telemetry validation.

### Overview
In production systems and automated test environments, evaluating a model or test suite using a single point-estimate accuracy (e.g., "96% accurate") is insufficient without understanding its variance and reliability bounds. This project implements a comparative framework analyzing standard normal approximations (**Wald intervals**) against robust score intervals (**Wilson intervals**) under small-sample constraints and boundary conditions.

### Key Features
- **Baseline Modeling:** Trains a `RandomForestClassifier` on structured tabular data.
- **Small-Batch Evaluation:** Computes accuracy and 95% confidence intervals on controlled sample sizes ($N=30$).
- **Comparative Analysis:** Generates interval shrinkage curves across sample sizes ($N=10$ to $100$).
- **Boundary Stress-Testing:** Demonstrates how standard Wald intervals fail near performance thresholds while Wilson scores remain constrained and reliable.

### Tech Stack
- **Python** (`pandas`, `numpy`, `scikit-learn`, `statsmodels`, `matplotlib`)

### Sample Results & Visualization
When running `evaluate.py`, the script outputs verification metrics and generates the interval width comparison visual:

```text
--- Results for N=30 ---
Point Estimate Accuracy: 70.00%
Wald Interval:   [0.5360, 0.8640]
Wilson Interval: [0.5212, 0.8334]

--- Boundary Test (29/30) ---
Wald Upper Bound:   1.0000 (Can exceed 1.0)
Wilson Upper Bound: 0.9941 (Bounded safely <= 1.0)
