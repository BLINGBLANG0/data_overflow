"""
SHAP Feature Importance & Explainability
Run this AFTER train.py to generate SHAP plots.
Install: pip install shap matplotlib
"""

import numpy as np
import pandas as pd
import joblib
import shap
import matplotlib.pyplot as plt

# ---------------------------------------------------------------------------
# 1. Load model artifacts & training data
# ---------------------------------------------------------------------------
MODEL_PATH = "model.pkl"
TRAIN_CSV = "train.csv"

artifacts = joblib.load(MODEL_PATH)
clf = artifacts["model"]
feature_cols = artifacts["feature_cols"]

# We need to re-create the feature matrix (import FE from train.py)
from train import feature_engineering, apply_label_encoders, apply_frequency_encoders

raw = pd.read_csv(TRAIN_CSV)
df = feature_engineering(raw)
target = df.pop("Purchased_Coverage_Bundle")
df.pop("User_ID")

# Apply same encodings
df = apply_label_encoders(df, artifacts["label_encoders"])
df = apply_frequency_encoders(df, artifacts["freq_maps"])
for c in artifacts["freq_cols"]:
    if c in df.columns:
        df.drop(columns=[c], inplace=True)

X = df[feature_cols].astype(np.float32)

# ---------------------------------------------------------------------------
# 2. SHAP analysis (use a sample for speed)
# ---------------------------------------------------------------------------
SAMPLE_SIZE = 2000  # increase for more accurate SHAP, decrease for speed
np.random.seed(42)
idx = np.random.choice(len(X), min(SAMPLE_SIZE, len(X)), replace=False)
X_sample = X.iloc[idx]

print("Computing SHAP values (this may take a few minutes) …")
explainer = shap.TreeExplainer(clf)
shap_values = explainer.shap_values(X_sample)

# ---------------------------------------------------------------------------
# 3. Plots
# ---------------------------------------------------------------------------

# 3a. Global feature importance (bar plot, averaged over all classes)
print("Generating summary bar plot …")
shap.summary_plot(shap_values, X_sample, plot_type="bar", max_display=25, show=False)
plt.tight_layout()
plt.savefig("shap_feature_importance_bar.png", dpi=150, bbox_inches="tight")
plt.close()
print("  → Saved shap_feature_importance_bar.png")

# 3b. Beeswarm plot for each class (top 15 features)
for cls in range(10):
    print(f"Generating beeswarm for class {cls} …")
    shap.summary_plot(shap_values[cls], X_sample, max_display=15, show=False)
    plt.title(f"SHAP — Class {cls}")
    plt.tight_layout()
    plt.savefig(f"shap_class_{cls}.png", dpi=150, bbox_inches="tight")
    plt.close()
print("  → Saved shap_class_0.png … shap_class_9.png")

# 3c. Mean absolute SHAP as a DataFrame
mean_abs_shap = np.mean([np.abs(sv) for sv in shap_values], axis=0).mean(axis=0)
importance_df = pd.DataFrame({
    "Feature": feature_cols,
    "Mean_Abs_SHAP": mean_abs_shap,
}).sort_values("Mean_Abs_SHAP", ascending=False)
importance_df.to_csv("shap_importance.csv", index=False)
print("\nTop 20 features by mean |SHAP|:")
print(importance_df.head(20).to_string(index=False))
print("\n  → Saved shap_importance.csv")

print("\n✓ SHAP analysis complete.")
