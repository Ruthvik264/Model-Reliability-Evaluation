# Model Reliability & Confidence Interval Evaluation

A Python framework for evaluating machine learning classification performance using rigorous statistical bounds (**Wald vs. Wilson Score Intervals**), designed for small-batch test automation and telemetry validation.

### Overview
In production systems and automated test environments, evaluating a model or test suite using a single point-estimate accuracy (e.g., "96% accurate") is insufficient without understanding its variance and reliability bounds. This project implements a comparative framework analyzing standard normal approximations (**Wald intervals**) against robust score intervals (**Wilson intervals**) under small-sample constraints and boundary conditions.

### Key Features
- **Baseline Modeling:** Trains a `RandomForestClassifier` on structured tabular data.
- **Small-Batch Evaluation:** Computes accuracy and 95% confidence intervals on controlled sample sizes ($N=30$).
- **Comparative Analysis:** Generates interval shrinkage curves across sample sizes ($N=10$ to $100$).
- **Boundary Stress-Testing:** Demonstrates how standard Wald intervals fail near performance thresholds (exceeding valid bounds $>1.0$) while Wilson scores remain constrained and reliable.

### Tech Stack
- **Python** (`pandas`, `numpy`, `scikit-learn`, `statsmodels`, `matplotlib`)

### Usage
Run the main evaluation script to generate metrics and visualization plots:
```bash
python evaluate.py
