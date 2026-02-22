"""Quick confusion matrix to find what's eating class 3."""
import pandas as pd
import numpy as np
import joblib
import lightgbm as lgb
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import confusion_matrix, f1_score

from train import feature_engineering, apply_label_encoders, apply_frequency_encoders

artifacts = joblib.load("model.pkl")
feature_cols = artifacts["feature_cols"]
le_encoders = artifacts["label_encoders"]
freq_maps = artifacts["freq_maps"]
freq_cols = artifacts["freq_cols"]

raw = pd.read_csv("train.csv")
df = feature_engineering(raw)
target = df.pop("Purchased_Coverage_Bundle")
df.pop("User_ID")

df = apply_label_encoders(df, le_encoders)
df = apply_frequency_encoders(df, freq_maps)
for c in freq_cols:
    if c in df.columns:
        df.drop(columns=[c], inplace=True)

X = df[feature_cols].values.astype(np.float32)
y = target.values.astype(np.int32)

# OOF predictions
params = artifacts["model"].get_params()
params["verbose"] = -1
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
oof = np.zeros(len(y), dtype=np.int32)
for tr, va in skf.split(X, y):
    m = lgb.LGBMClassifier(**params)
    m.fit(X[tr], y[tr], eval_set=[(X[va], y[va])],
          callbacks=[lgb.early_stopping(50, verbose=False)])
    oof[va] = m.predict(X[va])

cm = confusion_matrix(y, oof, labels=list(range(10)))
print("\nConfusion Matrix (rows=true, cols=predicted):\n")
header = "     " + "".join(f"  P{i:d}  " for i in range(10))
print(header)
for i in range(10):
    row = f"T{i}  " + "".join(f"{cm[i,j]:6d} " for j in range(10))
    print(row)

print(f"\nClass 3 (true=3) is predicted as:")
for j in range(10):
    if cm[3, j] > 0:
        print(f"  Predicted {j}: {cm[3,j]:5d}  ({100*cm[3,j]/cm[3].sum():.1f}%)")
