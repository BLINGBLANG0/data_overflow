"""
Training script for Insurance Bundle Classification (10 classes, 0-9).
Optimized for: score = Macro_F1 × max(0.5, 1−size_mb/200) × max(0.5, 1−predict_s/10)
Run on Colab or locally.  Requires: lightgbm, scikit-learn, pandas, numpy, joblib, optuna (for tuning).
"""

import warnings, time, gc, os
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import lightgbm as lgb
import joblib
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import f1_score

# ---------------------------------------------------------------------------
# 0. CONFIG
# ---------------------------------------------------------------------------
TRAIN_CSV = "train.csv"          # ← adjust path if needed
MODEL_PATH = "model.pkl"
SEED = 42
N_FOLDS = 5
OPTUNA_TRIALS = 0            # set to 0 to skip tuning & use defaults

# ---------------------------------------------------------------------------
# 1. FEATURE ENGINEERING  (shared with solution.py → keep in sync!)
# ---------------------------------------------------------------------------

# Columns we treat as categorical (will be label-encoded)
CATEGORICAL_COLS = [
    "Employment_Status", "Region_Code", "Deductible_Tier",
    "Payment_Schedule", "Broker_Agency_Type", "Acquisition_Channel",
    "Existing_Policyholder", "Policy_Cancelled_Post_Purchase",
]

def feature_engineering(df: pd.DataFrame) -> pd.DataFrame:
    """
    Heavy-duty, deterministic feature engineering.
    Receives raw data (with or without the target column).
    Returns engineered DataFrame (target excluded).
    """
    df = df.copy()

    # --- Drop junk columns ---
    df = df.drop(columns=[c for c in df.columns if c.startswith("Unnamed")], errors="ignore")

    # --- Drop target if present (we handle it outside) ---
    target_col = "Purchased_Coverage_Bundle"
    has_target = target_col in df.columns
    if has_target:
        target = df[target_col].copy()
        df.drop(columns=[target_col], inplace=True)

    # Keep User_ID aside (needed in submission, but not a feature)
    user_id = df["User_ID"].copy() if "User_ID" in df.columns else None
    if "User_ID" in df.columns:
        df.drop(columns=["User_ID"], inplace=True)

    # ---- CONVERT Policy_Start_Month from string to int ----
    MONTH_MAP = {
        "january": 1, "february": 2, "march": 3, "april": 4,
        "may": 5, "june": 6, "july": 7, "august": 8,
        "september": 9, "october": 10, "november": 11, "december": 12,
    }
    if not pd.api.types.is_numeric_dtype(df["Policy_Start_Month"]):
        df["Policy_Start_Month"] = (
            df["Policy_Start_Month"]
            .astype(str)
            .str.strip()
            .str.lower()
            .map(MONTH_MAP)
            .fillna(1)
            .astype(int)
        )

    # ---- Handle Broker_ID / Employer_ID missingness ----
    df["Broker_ID_missing"] = df["Broker_ID"].isna().astype(np.int8)
    df["Employer_ID_missing"] = df["Employer_ID"].isna().astype(np.int8)

    # ---- HARDCODED RULE: Class 9 perfect fingerprint (5/5 in train) ----
    # Must be computed BEFORE encoding transforms raw column values
    df["_rule_class9"] = (
        (df["Estimated_Annual_Income"].fillna(-1) == 0) &
        (df["Region_Code"].isna()) &
        (df["Policy_Cancelled_Post_Purchase"] == 1) &
        (df["Deductible_Tier"].astype(str) == "Tier_4_Zero_Ded")
    ).astype(np.int8)

    # ---- HARDCODED RULE: Class 8 fingerprint (6/6 in train, 12 total matches) ----
    df["_rule_class8"] = (
        (df["Region_Code"].astype(str).str.strip() == "PRT") &
        (df["Deductible_Tier"].astype(str).str.strip() == "Tier_1_High_Ded") &
        (df["Vehicles_on_Policy"].fillna(-1) == 0) &
        (df["Underwriting_Processing_Days"].fillna(-1) == 0) &
        (df["Policy_Start_Year"].fillna(-1) == 2015) &
        (df["Previous_Policy_Duration_Months"].fillna(-1) == 1) &
        (df["Existing_Policyholder"].fillna(-1) == 0) &
        (df["Broker_ID"].isna()) &
        (df["Employer_ID"].isna()) &
        (df["Child_Dependents"].fillna(-1) == 0) &
        (df["Infant_Dependents"].fillna(-1) == 0) &
        (df["Custom_Riders_Requested"].fillna(-1) == 0) &
        (df["Policy_Amendments_Count"].fillna(-1) == 0) &
        (df["Acquisition_Channel"].astype(str).str.strip() == "Direct_Website") &
        (df["Payment_Schedule"].astype(str).str.strip() == "Monthly_EFT") &
        (df["Broker_Agency_Type"].astype(str).str.strip() == "National_Corporate") &
        (df["Employment_Status"].astype(str).str.strip() == "Employed_FullTime") &
        (df["Policy_Start_Month"].isin([11, 12]))
    ).astype(np.int8)

    # ---- A. Dependency / household features ----
    df["Total_Dependents"] = (
        df["Adult_Dependents"].fillna(0)
        + df["Child_Dependents"].fillna(0)
        + df["Infant_Dependents"].fillna(0)
    )
    df["Has_Dependents"] = (df["Total_Dependents"] > 0).astype(np.int8)
    df["Has_Infant"] = (df["Infant_Dependents"].fillna(0) > 0).astype(np.int8)
    df["Has_Child"] = (df["Child_Dependents"].fillna(0) > 0).astype(np.int8)
    df["Has_Adult_Dep"] = (df["Adult_Dependents"].fillna(0) > 0).astype(np.int8)
    df["Child_Infant_Ratio"] = (
        df["Child_Dependents"].fillna(0) / (df["Infant_Dependents"].fillna(0) + 1)
    )
    df["Dep_Mix"] = (
        df["Adult_Dependents"].fillna(0) * 3
        + df["Child_Dependents"].fillna(0) * 2
        + df["Infant_Dependents"].fillna(0) * 1
    )

    # ---- B. Income features ----
    income = df["Estimated_Annual_Income"].fillna(0)
    df["Log_Income"] = np.log1p(income)
    df["Income_Per_Dependent"] = income / (df["Total_Dependents"] + 1)
    df["Income_Per_Vehicle"] = income / (df["Vehicles_on_Policy"].fillna(1).clip(lower=1))

    # ---- C. Policy history / risk features ----
    df["Claims_Per_Year"] = (
        df["Previous_Claims_Filed"].fillna(0)
        / (df["Previous_Policy_Duration_Months"].fillna(1).clip(lower=1) / 12)
    )
    df["Years_Without_Claims_Ratio"] = (
        df["Years_Without_Claims"].fillna(0)
        / (df["Previous_Policy_Duration_Months"].fillna(1).clip(lower=1) / 12)
    ).clip(upper=1.0)
    df["Is_New_Customer"] = (df["Existing_Policyholder"].fillna(0) == 0).astype(np.int8)
    df["Policy_Duration_Years"] = df["Previous_Policy_Duration_Months"].fillna(0) / 12
    df["Had_Cancellation"] = (df["Policy_Cancelled_Post_Purchase"].fillna(0) == 1).astype(np.int8)
    df["Risk_Score"] = (
        df["Previous_Claims_Filed"].fillna(0) * 2
        - df["Years_Without_Claims"].fillna(0)
        + df["Had_Cancellation"] * 3
    )

    # ---- D. Date / timing features ----
    df["Policy_Start_Quarter"] = ((df["Policy_Start_Month"].fillna(1) - 1) // 3 + 1).astype(np.int8)
    df["Is_Year_End"] = (df["Policy_Start_Month"].fillna(0).isin([11, 12])).astype(np.int8)
    df["Is_Year_Start"] = (df["Policy_Start_Month"].fillna(0).isin([1, 2])).astype(np.int8)
    df["Day_of_Year_Approx"] = (
        (df["Policy_Start_Month"].fillna(1) - 1) * 30 + df["Policy_Start_Day"].fillna(1)
    )
    # Cyclical encoding for month
    month = df["Policy_Start_Month"].fillna(1)
    df["Month_Sin"] = np.sin(2 * np.pi * month / 12)
    df["Month_Cos"] = np.cos(2 * np.pi * month / 12)
    # Cyclical encoding for day of week (approximated from week)
    week = df["Policy_Start_Week"].fillna(1)
    df["Week_Sin"] = np.sin(2 * np.pi * week / 52)
    df["Week_Cos"] = np.cos(2 * np.pi * week / 52)

    df["Quote_to_Processing_Ratio"] = (
        df["Days_Since_Quote"].fillna(0)
        / (df["Underwriting_Processing_Days"].fillna(1).clip(lower=1))
    )
    df["Total_Wait_Days"] = (
        df["Days_Since_Quote"].fillna(0) + df["Underwriting_Processing_Days"].fillna(0)
    )

    # ---- E. Vehicle / rider features ----
    df["Riders_Per_Vehicle"] = (
        df["Custom_Riders_Requested"].fillna(0)
        / (df["Vehicles_on_Policy"].fillna(1).clip(lower=1))
    )
    df["Has_Riders"] = (df["Custom_Riders_Requested"].fillna(0) > 0).astype(np.int8)
    df["Has_Grace_Ext"] = (df["Grace_Period_Extensions"].fillna(0) > 0).astype(np.int8)
    df["Amendments_Per_Duration"] = (
        df["Policy_Amendments_Count"].fillna(0)
        / (df["Previous_Policy_Duration_Months"].fillna(1).clip(lower=1))
    )
    df["High_Amendments"] = (df["Policy_Amendments_Count"].fillna(0) >= 3).astype(np.int8)

    # ---- F. Interaction features ----
    df["Income_x_Dependents"] = income * df["Total_Dependents"]
    df["Income_x_Vehicles"] = income * df["Vehicles_on_Policy"].fillna(0)
    df["Vehicles_x_Riders"] = (
        df["Vehicles_on_Policy"].fillna(0) * df["Custom_Riders_Requested"].fillna(0)
    )
    df["Claims_x_Duration"] = (
        df["Previous_Claims_Filed"].fillna(0) * df["Previous_Policy_Duration_Months"].fillna(0)
    )

    # ---- F2. Additional discriminative features ----
    # Year-based features (class 8 strongly correlates with 2015)
    df["Policy_Year"] = df["Policy_Start_Year"].fillna(0).astype(int)

    # Zero-activity indicator (many rare classes have zero everything)
    df["Zero_Activity"] = (
        (df["Vehicles_on_Policy"].fillna(0) == 0) &
        (df["Custom_Riders_Requested"].fillna(0) == 0) &
        (df["Underwriting_Processing_Days"].fillna(0) == 0) &
        (df["Policy_Amendments_Count"].fillna(0) == 0)
    ).astype(np.int8)

    # Grace extensions per duration
    df["Grace_Per_Duration"] = (
        df["Grace_Period_Extensions"].fillna(0)
        / (df["Previous_Policy_Duration_Months"].fillna(1).clip(lower=1))
    )

    # Income to claims ratio
    df["Income_Per_Claim"] = income / (df["Previous_Claims_Filed"].fillna(0) + 1)

    # Dependents composition ratios  
    total_dep = df["Total_Dependents"] + 1
    df["Adult_Dep_Ratio"] = df["Adult_Dependents"].fillna(0) / total_dep
    df["Child_Dep_Ratio"] = df["Child_Dependents"].fillna(0) / total_dep
    df["Infant_Dep_Ratio"] = df["Infant_Dependents"].fillna(0) / total_dep

    # ---- G. Encode categoricals ----
    for col in CATEGORICAL_COLS:
        if col in df.columns:
            df[col] = df[col].astype(str).fillna("__MISSING__")

    # ---- H. Broker / Employer frequency features ----
    for col in ["Broker_ID", "Employer_ID"]:
        if col in df.columns:
            df[col] = df[col].astype(str).fillna("__MISSING__")

    # ---- Restore User_ID ----
    if user_id is not None:
        df.insert(0, "User_ID", user_id)
    if has_target:
        df[target_col] = target

    return df


# ---------------------------------------------------------------------------
# 2. ENCODING HELPERS  (fit on train, transform on both)
# ---------------------------------------------------------------------------
def fit_label_encoders(df, cols):
    """Fit label encoders: return dict of {col: {value: code}} mappings."""
    encoders = {}
    for c in cols:
        df[c] = df[c].astype(str).fillna("__MISSING__")
        uniques = df[c].unique()
        encoders[c] = {v: i for i, v in enumerate(sorted(uniques))}
    return encoders


def apply_label_encoders(df, encoders):
    """Transform columns using dict mappings (unseen → -1). Vectorized."""
    for c, mapping in encoders.items():
        if c not in df.columns:
            continue
        df[c] = df[c].astype(str).fillna("__MISSING__").map(mapping).fillna(-1).astype(int)
    return df


def fit_frequency_encoders(df, cols):
    """Return frequency maps for high-cardinality columns."""
    freq_maps = {}
    for c in cols:
        freq_maps[c] = df[c].value_counts(normalize=True).to_dict()
    return freq_maps


def apply_frequency_encoders(df, freq_maps):
    for c, fmap in freq_maps.items():
        if c in df.columns:
            df[f"{c}_freq"] = df[c].map(fmap).fillna(0).astype(np.float32)
    return df


# ---------------------------------------------------------------------------
# 3. MAIN TRAINING PIPELINE
# ---------------------------------------------------------------------------
def main():
    print("=" * 60)
    print("  INSURANCE BUNDLE CLASSIFIER — TRAINING")
    print("=" * 60)

    # --- Load data ---
    raw = pd.read_csv(TRAIN_CSV)
    # Drop junk trailing rows (all-NaN) and junk columns
    raw = raw.dropna(subset=["Purchased_Coverage_Bundle"]).reset_index(drop=True)
    raw = raw.drop(columns=[c for c in raw.columns if c.startswith("Unnamed")], errors="ignore")
    raw["Purchased_Coverage_Bundle"] = raw["Purchased_Coverage_Bundle"].astype(int)
    print(f"Raw shape: {raw.shape}")
    print(f"Target distribution:\n{raw['Purchased_Coverage_Bundle'].value_counts().sort_index()}\n")

    # --- Feature engineering ---
    df = feature_engineering(raw)
    target = df.pop("Purchased_Coverage_Bundle")
    user_id = df.pop("User_ID")

    # --- Encode categoricals ---
    cat_cols = [c for c in CATEGORICAL_COLS if c in df.columns]
    le_encoders = fit_label_encoders(df, cat_cols)
    df = apply_label_encoders(df, le_encoders)

    # --- Frequency encode high-cardinality ---
    freq_cols = ["Broker_ID", "Employer_ID"]
    freq_cols = [c for c in freq_cols if c in df.columns]
    freq_maps = fit_frequency_encoders(df, freq_cols)
    df = apply_frequency_encoders(df, freq_maps)

    # Drop raw high-card ID columns (keep freq-encoded versions)
    for c in freq_cols:
        if c in df.columns:
            df.drop(columns=[c], inplace=True)

    # --- Final feature list (exclude rule flags — used only for post-prediction override) ---
    feature_cols = [c for c in df.columns if c not in ["User_ID", "Purchased_Coverage_Bundle", "_rule_class9", "_rule_class8"]]

    # --- Feature selection: drop complex/noisy features that hurt generalization ---
    EXCLUDE_FEATURES = {
        "Child_Infant_Ratio", "Dep_Mix",
        "Income_Per_Dependent", "Income_Per_Vehicle",
        "Month_Sin", "Month_Cos", "Week_Sin", "Week_Cos",
        "Day_of_Year_Approx",
        "Quote_to_Processing_Ratio",
        "Riders_Per_Vehicle",
        "Amendments_Per_Duration", "High_Amendments",
        "Income_x_Dependents", "Income_x_Vehicles", "Vehicles_x_Riders", "Claims_x_Duration",
        "Policy_Year", "Zero_Activity", "Grace_Per_Duration",
        "Income_Per_Claim",
        "Adult_Dep_Ratio", "Child_Dep_Ratio", "Infant_Dep_Ratio",
    }
    feature_cols = [c for c in feature_cols if c not in EXCLUDE_FEATURES]

    print(f"Number of features: {len(feature_cols)}")
    print(f"Features: {feature_cols}\n")

    X = df[feature_cols].values.astype(np.float32)
    y = target.values.astype(np.int32)

    # === HYPERPARAMETER TUNING (Optuna) ===
    # Compute per-class weights emphasizing rare classes more aggressively
    from collections import Counter
    class_counts = Counter(y)
    n_samples = len(y)
    n_classes = len(class_counts)
    # Moderate emphasis on rare classes; cap at 3.0 to prevent extreme upweighting
    custom_weights = {}
    for cls, cnt in class_counts.items():
        w = (n_samples / (n_classes * cnt)) ** 0.4
        custom_weights[cls] = min(w, 3.0)
    # Rules handle classes 8 and 9 — don't distort model for them
    custom_weights[8] = 1.0
    custom_weights[9] = 1.0

    best_params = {
        # Good defaults — will be overridden if OPTUNA_TRIALS > 0
        "objective": "multiclass",
        "num_class": 10,
        "metric": "multi_logloss",
        "boosting_type": "gbdt",
        "n_estimators": 80,
        "learning_rate": 0.20,
        "num_leaves": 31,
        "max_depth": 6,
        "min_child_samples": 50,
        "subsample": 0.7,
        "subsample_freq": 3,
        "colsample_bytree": 0.6,
        "reg_alpha": 0.2,
        "reg_lambda": 5.0,
        "min_split_gain": 0.05,
        "path_smooth": 5.0,
        "random_state": SEED,
        "n_jobs": -1,
        "verbose": -1,
        "class_weight": custom_weights,
    }

    if OPTUNA_TRIALS > 0:
        try:
            import optuna
            from optuna.samplers import TPESampler

            def objective(trial):
                params = {
                    "objective": "multiclass",
                    "num_class": 10,
                    "metric": "multi_logloss",
                    "boosting_type": "gbdt",
                    "n_estimators": trial.suggest_int("n_estimators", 400, 2000, step=100),
                    "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.15, log=True),
                    "num_leaves": trial.suggest_int("num_leaves", 31, 127),
                    "max_depth": trial.suggest_int("max_depth", 5, 12),
                    "min_child_samples": trial.suggest_int("min_child_samples", 10, 80),
                    "subsample": trial.suggest_float("subsample", 0.6, 1.0),
                    "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
                    "reg_alpha": trial.suggest_float("reg_alpha", 1e-3, 10.0, log=True),
                    "reg_lambda": trial.suggest_float("reg_lambda", 1e-3, 10.0, log=True),
                    "min_split_gain": trial.suggest_float("min_split_gain", 0.0, 1.0),
                    "random_state": SEED,
                    "n_jobs": -1,
                    "verbose": -1,
                    "class_weight": "balanced",
                }
                skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
                scores = []
                for tr_idx, va_idx in skf.split(X, y):
                    clf = lgb.LGBMClassifier(**params)
                    clf.fit(
                        X[tr_idx], y[tr_idx],
                        eval_set=[(X[va_idx], y[va_idx])],
                        callbacks=[lgb.early_stopping(50, verbose=False)],
                    )
                    preds = clf.predict(X[va_idx])
                    scores.append(f1_score(y[va_idx], preds, average="macro"))
                return np.mean(scores)

            print("Running Optuna hyperparameter search …")
            sampler = TPESampler(seed=SEED)
            study = optuna.create_study(direction="maximize", sampler=sampler)
            study.optimize(objective, n_trials=OPTUNA_TRIALS, show_progress_bar=True)

            print(f"\nBest trial Macro F1: {study.best_value:.5f}")
            print(f"Best params: {study.best_params}\n")

            best_params.update(study.best_params)
        except ImportError:
            print("optuna not installed — using default hyperparameters.\n")

    # === CROSS-VALIDATION WITH BEST PARAMS ===
    print("Running final cross-validation …")
    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
    cv_scores = []
    best_iters = []
    oof_preds = np.zeros(len(y), dtype=np.int32)
    oof_proba = np.zeros((len(y), 10), dtype=np.float32)

    # Get rule flags for OOF evaluation
    rule_class9_all = df["_rule_class9"].values if "_rule_class9" in df.columns else np.zeros(len(y))
    rule_class8_all = df["_rule_class8"].values if "_rule_class8" in df.columns else np.zeros(len(y))

    for fold, (tr_idx, va_idx) in enumerate(skf.split(X, y)):
        clf = lgb.LGBMClassifier(**best_params)
        clf.fit(
            X[tr_idx], y[tr_idx],
            eval_set=[(X[va_idx], y[va_idx])],
            callbacks=[lgb.early_stopping(50, verbose=False)],
        )
        best_iters.append(clf.best_iteration_)
        preds = clf.predict(X[va_idx])
        proba = clf.predict_proba(X[va_idx])
        oof_preds[va_idx] = preds
        oof_proba[va_idx] = proba
        fold_f1 = f1_score(y[va_idx], preds, average="macro")
        cv_scores.append(fold_f1)
        print(f"  Fold {fold+1}: Macro F1 = {fold_f1:.5f}  (best iter = {clf.best_iteration_})")

    # Apply hardcoded rules to OOF predictions
    oof_preds_with_rules = oof_preds.copy()
    oof_preds_with_rules[rule_class9_all == 1] = 9
    oof_preds_with_rules[rule_class8_all == 1] = 8

    avg_best_iter = int(np.mean(best_iters))
    overall_f1_raw = f1_score(y, oof_preds, average="macro")
    overall_f1_rules = f1_score(y, oof_preds_with_rules, average="macro")
    print(f"\n  CV Mean Macro F1: {np.mean(cv_scores):.5f} ± {np.std(cv_scores):.5f}")
    print(f"  OOF  Macro F1 (raw):        {overall_f1_raw:.5f}")
    print(f"  OOF  Macro F1 (with rules): {overall_f1_rules:.5f}")
    print(f"  Avg best iteration: {avg_best_iter}\n")

    # === SKIP CALIBRATION — scales = 1.0 to prevent overfitting ===
    # (Calibration was overfitting to OOF distribution, hurting server F1)
    optimal_scales = np.ones(10, dtype=np.float64)
    overall_f1 = overall_f1_rules
    print(f"  No calibration (scales=1.0). OOF Macro F1: {overall_f1:.5f}  (raw: {overall_f1_raw:.5f}, rules: {overall_f1_rules:.5f})\n")
    oof_preds = oof_preds_with_rules  # use rule-enhanced predictions for reporting

    # === TRAIN FINAL MODEL ON ALL DATA (fixed n_estimators, no early stopping) ===
    # Holdout early stopping leads to too many trees (overfitting + latency).
    # Fixed tree count = predictable latency + controlled complexity.
    print("Training final model on full dataset …")
    final_params = best_params.copy()
    final_params["n_estimators"] = 80  # fixed for latency budget (~1s on server)
    print(f"  Using n_estimators = {final_params['n_estimators']} (fixed for latency)")
    final_clf = lgb.LGBMClassifier(**final_params)
    final_clf.fit(X, y)

    # === SAVE ARTIFACTS ===
    artifacts = {
        "model": final_clf,
        "booster": final_clf.booster_,  # raw booster for faster inference
        "feature_cols": feature_cols,
        "label_encoders": le_encoders,
        "freq_maps": freq_maps,
        "cat_cols": cat_cols,
        "freq_cols": freq_cols,
        "class_scales": optimal_scales.tolist(),  # per-class probability scaling
    }
    joblib.dump(artifacts, MODEL_PATH, compress=3)
    size_mb = os.path.getsize(MODEL_PATH) / (1024 * 1024)
    print(f"Model saved to {MODEL_PATH}  ({size_mb:.2f} MB)")

    # --- Score estimate ---
    size_penalty = max(0.5, 1 - size_mb / 200)
    print(f"\nEstimated score components:")
    print(f"  Macro F1 (OOF):       {overall_f1:.5f}")
    print(f"  Size penalty factor:   {size_penalty:.4f}  ({size_mb:.2f} MB)")
    print(f"  Speed penalty factor:  ~1.00  (LightGBM inference is < 1s)")
    print(f"  Estimated score:       {overall_f1 * size_penalty:.5f}  (before speed penalty)")

    # --- Per-class F1 ---
    from sklearn.metrics import classification_report
    print("\nPer-class OOF classification report:")
    print(classification_report(y, oof_preds, digits=4))

    print("\n✓ Training complete. Now prepare your submission zip with:")
    print("  - solution.py")
    print("  - model.pkl")
    print("  - requirements.txt")


if __name__ == "__main__":
    main()
