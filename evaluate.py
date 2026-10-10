import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from statsmodels.stats.proportion import proportion_confint

def run_evaluation():
    print("--- 1. Loading and Preparing Dataset ---")
    # Load UCI Heart Disease Dataset for binary classification
    url = "https://archive.ics.uci.edu/ml/machine-learning-databases/heart-disease/processed.cleveland.data"
    names = ['age', 'sex', 'cp', 'trestbps', 'chol', 'fbs', 'restecg', 'thalach', 'exang', 'oldpeak', 'slope', 'ca', 'thal', 'target']
    df = pd.read_csv(url, names=names, na_values="?")
    df = df.dropna()
    df['target'] = (df['target'] > 0).astype(int) # Binary classification: Disease vs No Disease

    X = df.drop('target', axis=1)
    y = df['target']
    X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.5, random_state=42)

    print("--- 2. Training Baseline Classifier ---")
    model = RandomForestClassifier(random_state=42)
    model.fit(X_train, y_train)

    # Small-Batch Evaluation (N = 30)
    N_small = 30
    X_test_small = X_temp.sample(N_small, random_state=7)
    y_test_small = y_temp.loc[X_test_small.index]

    predictions = model.predict(X_test_small)
    hits = np.sum(predictions == y_test_small)
    accuracy = hits / N_small

    # Calculate 95% Confidence Intervals
    wald_low, wald_high = proportion_confint(hits, N_small, method='normal')
    wilson_low, wilson_high = proportion_confint(hits, N_small, method='wilson')

    print(f"\n--- Results for Small Batch (N={N_small}) ---")
    print(f"Point Estimate Accuracy: {accuracy:.2%}")
    print(f"Wald Interval (Normal):   [{wald_low:.4f}, {wald_high:.4f}]")
    print(f"Wilson Interval (Score):  [{wilson_low:.4f}, {wilson_high:.4f}]")

    print("\n--- 3. Running Boundary Stress Test (29/30 Successes) ---")
    b_hits, b_n = 29, 30
    w_l, w_h = proportion_confint(b_hits, b_n, method='normal')
    wi_l, wi_h = proportion_confint(b_hits, b_n, method='wilson')
    print(f"Wald Upper Bound:   {w_h:.4f}  <-- Notice this can exceed 1.0 (invalid probability)")
    print(f"Wilson Upper Bound: {wi_h:.4f}  <-- Safely constrained <= 1.0")

    print("\n--- 4. Generating Interval Width Comparison Plot ---")
    n_sizes = np.arange(10, 101, 5)
    wald_widths = []
    wilson_widths = []

    for n in n_sizes:
        h = int(0.9 * n) # Simulating a steady 90% performance rate
        w_l, w_h = proportion_confint(h, n, method='normal')
        wi_l, wi_h = proportion_confint(h, n, method='wilson')
        wald_widths.append(w_h - w_l)
        wilson_widths.append(wi_h - wi_l)

    # Plotting
    plt.figure(figsize=(9, 5))
    plt.plot(n_sizes, wald_widths, 'o-', label='Wald Interval Width', color='#e74c3c', linewidth=2)
    plt.plot(n_sizes, wilson_widths, 's-', label='Wilson Interval Width', color='#3498db', linewidth=2)
    plt.title("Statistical Interval Width Comparison: Wald vs Wilson (95% CI)", fontsize=12, fontweight='bold')
    plt.xlabel("Sample Size (N)", fontsize=10)
    plt.ylabel("Width of Confidence Interval", fontsize=10)
    plt.legend(frameon=True)
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.tight_layout()
    plt.savefig("interval_comparison.png", dpi=300)
    print("Plot saved successfully as 'interval_comparison.png'!")
    plt.show()

if __name__ == "__main__":
    run_evaluation()
