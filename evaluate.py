"""
evaluate.py — Local evaluation script for Insurance Bundle Classification.
Run AFTER train.py to compute the exact hackathon score locally.
Tracks runs in a log file so you know when it's worth submitting.

Usage:
    python evaluate.py                     # uses train.csv head as proxy
    python evaluate.py --test test.csv     # uses actual test CSV (no labels, just timing)
"""

import os, sys, time, json, datetime
import numpy as np
import pandas as pd
import joblib
from sklearn.metrics import f1_score, classification_report

# ---------------------------------------------------------------------------
# CONFIG
# ---------------------------------------------------------------------------
MODEL_PATH = "model.pkl"
TRAIN_CSV = "train.csv"
LOG_FILE = "run_log.json"          # persistent tracking across runs
SIMULATE_TEST_SIZE = 15000         # approximate test set size for timing
MIN_IMPROVEMENT = 0.003            # minimum OOF F1 improvement to recommend submission
MAX_CV_STD = 0.015                 # maximum acceptable CV std dev
MIN_CLASS_F1 = 0.40                # flag classes below this

# ---------------------------------------------------------------------------
# HELPERS
# ---------------------------------------------------------------------------
def load_log():
    if os.path.exists(LOG_FILE):
        with open(LOG_FILE, "r") as f:
            return json.load(f)
    return {"runs": [], "best_oof_f1": 0.0, "submissions_used": 0, "submissions_total": 20}


def save_log(log):
    with open(LOG_FILE, "w") as f:
        json.dump(log, f, indent=2)


def color(text, code):
    """ANSI color for terminal output."""
    return f"\033[{code}m{text}\033[0m"


def green(text):  return color(text, "92")
def red(text):    return color(text, "91")
def yellow(text): return color(text, "93")
def bold(text):   return color(text, "1")
def cyan(text):   return color(text, "96")


# ---------------------------------------------------------------------------
# MAIN EVALUATION
# ---------------------------------------------------------------------------
def main():
    print(bold("=" * 64))
    print(bold("  INSURANCE BUNDLE CLASSIFIER — LOCAL EVALUATION"))
    print(bold("=" * 64))
    print()

    # --- Check files exist ---
    if not os.path.exists(MODEL_PATH):
        print(red(f"ERROR: {MODEL_PATH} not found. Run train.py first."))
        sys.exit(1)
    if not os.path.exists(TRAIN_CSV):
        print(red(f"ERROR: {TRAIN_CSV} not found."))
        sys.exit(1)
    if not os.path.exists("solution.py"):
        print(red("ERROR: solution.py not found."))
        sys.exit(1)

    # --- Import solution.py ---
    from solution import preprocess, load_model, predict

    # --- Load log ---
    log = load_log()
    run_number = len(log["runs"]) + 1

    print(cyan(f"Run #{run_number}"))
    print(f"Previous best OOF F1: {log['best_oof_f1']:.5f}")
    print(f"Submissions used: {log['submissions_used']}/{log['submissions_total']}")
    print()

    # =========================================================================
    # 1. MODEL SIZE
    # =========================================================================
    model_size_bytes = os.path.getsize(MODEL_PATH)
    model_size_mb = model_size_bytes / (1024 * 1024)
    size_penalty = max(0.5, 1 - model_size_mb / 200)

    print(bold("1. MODEL SIZE"))
    if model_size_mb < 20:
        print(green(f"   {model_size_mb:.2f} MB — Excellent (penalty factor: {size_penalty:.4f})"))
    elif model_size_mb < 50:
        print(yellow(f"   {model_size_mb:.2f} MB — OK (penalty factor: {size_penalty:.4f})"))
    else:
        print(red(f"   {model_size_mb:.2f} MB — Too large! (penalty factor: {size_penalty:.4f})"))
    print()

    # =========================================================================
    # 2. OOF MACRO F1 (re-run CV to get reliable OOF predictions)
    # =========================================================================
    print(bold("2. OOF MACRO F1 (Cross-Validation)"))
    from train import feature_engineering, apply_label_encoders, apply_frequency_encoders, CATEGORICAL_COLS
    from sklearn.model_selection import StratifiedKFold
    import lightgbm as lgb

    artifacts = joblib.load(MODEL_PATH)
    clf_model = artifacts["model"]
    feature_cols = artifacts["feature_cols"]
    le_encoders = artifacts["label_encoders"]
    freq_maps = artifacts["freq_maps"]
    freq_cols = artifacts["freq_cols"]

    raw = pd.read_csv(TRAIN_CSV)
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

    # Use the saved model's params for CV (reproduce training)
    params = clf_model.get_params()
    params["verbose"] = -1

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    cv_scores = []
    oof_preds = np.zeros(len(y), dtype=np.int32)

    for fold, (tr_idx, va_idx) in enumerate(skf.split(X, y)):
        clf = lgb.LGBMClassifier(**params)
        clf.fit(
            X[tr_idx], y[tr_idx],
            eval_set=[(X[va_idx], y[va_idx])],
            callbacks=[lgb.early_stopping(50, verbose=False)],
        )
        preds = clf.predict(X[va_idx])
        oof_preds[va_idx] = preds
        fold_f1 = f1_score(y[va_idx], preds, average="macro")
        cv_scores.append(fold_f1)
        print(f"   Fold {fold+1}: Macro F1 = {fold_f1:.5f}")

    oof_f1 = f1_score(y, oof_preds, average="macro")
    cv_std = np.std(cv_scores)
    cv_mean = np.mean(cv_scores)

    print(f"\n   CV Mean:  {cv_mean:.5f} ± {cv_std:.5f}")
    print(f"   OOF F1:   {oof_f1:.5f}")
    print()

    # --- Per-class F1 ---
    print(bold("3. PER-CLASS F1"))
    report = classification_report(y, oof_preds, output_dict=True, zero_division=0)
    weak_classes = []
    for cls in range(10):
        cls_f1 = report[str(cls)]["f1-score"]
        support = report[str(cls)]["support"]
        status = ""
        if cls_f1 < MIN_CLASS_F1:
            status = red(" ← WEAK!")
            weak_classes.append(cls)
        elif cls_f1 < 0.60:
            status = yellow(" ← could improve")
        print(f"   Class {cls}: F1={cls_f1:.4f}  (support={int(support)}){status}")
    print()

    # =========================================================================
    # 4. INFERENCE TIMING
    # =========================================================================
    print(bold("4. INFERENCE TIMING"))
    test_proxy = pd.read_csv(TRAIN_CSV).head(SIMULATE_TEST_SIZE)
    if "Purchased_Coverage_Bundle" in test_proxy.columns:
        test_proxy.drop(columns=["Purchased_Coverage_Bundle"], inplace=True)

    # Preprocess (not timed)
    df_pre = preprocess(test_proxy)

    # Load model (not timed)
    loaded_model = load_model()

    # TIMED: predict only
    times = []
    for _ in range(3):  # run 3 times, take worst
        start = time.perf_counter()
        # dict.copy() is shallow — numpy arrays are NOT copied (same as server behavior)
        result = predict(df_pre.copy() if isinstance(df_pre, pd.DataFrame) else {k: v for k, v in df_pre.items()}, loaded_model)
        elapsed = time.perf_counter() - start
        times.append(elapsed)

    predict_seconds = max(times)  # worst-case
    speed_penalty = max(0.5, 1 - predict_seconds / 10)

    if predict_seconds < 1.0:
        print(green(f"   {predict_seconds:.3f}s (worst of 3 runs) — Excellent (penalty: {speed_penalty:.4f})"))
    elif predict_seconds < 3.0:
        print(yellow(f"   {predict_seconds:.3f}s — OK (penalty: {speed_penalty:.4f})"))
    else:
        print(red(f"   {predict_seconds:.3f}s — Slow! (penalty: {speed_penalty:.4f})"))

    # Verify output format
    assert "User_ID" in result.columns, red("ERROR: User_ID missing from predict output!")
    assert "Purchased_Coverage_Bundle" in result.columns, red("ERROR: Purchased_Coverage_Bundle missing!")
    assert result["Purchased_Coverage_Bundle"].dtype in [np.int32, np.int64, int], \
        red(f"ERROR: Predictions are {result['Purchased_Coverage_Bundle'].dtype}, not int!")
    assert set(result["Purchased_Coverage_Bundle"].unique()).issubset(set(range(10))), \
        red("ERROR: Predictions contain values outside 0-9!")
    print(green("   Output format verified: User_ID + int predictions 0-9 ✓"))
    print()

    # =========================================================================
    # 5. FINAL SCORE
    # =========================================================================
    final_score = oof_f1 * size_penalty * speed_penalty

    print(bold("=" * 64))
    print(bold("  FINAL SCORE ESTIMATE"))
    print(bold("=" * 64))
    print(f"  Macro F1 (OOF):         {oof_f1:.5f}")
    print(f"  × Size penalty:         {size_penalty:.4f}  ({model_size_mb:.2f} MB)")
    print(f"  × Speed penalty:        {speed_penalty:.4f}  ({predict_seconds:.3f}s)")
    print(f"  ─────────────────────────────────")
    print(bold(f"  ESTIMATED SCORE:        {final_score:.5f}"))
    print()

    # =========================================================================
    # 6. SUBMIT RECOMMENDATION
    # =========================================================================
    improvement = oof_f1 - log["best_oof_f1"]
    submissions_left = log["submissions_total"] - log["submissions_used"]

    print(bold("  RECOMMENDATION"))
    print(f"  ─────────────────────────────────")

    issues = []
    if cv_std > MAX_CV_STD:
        issues.append(f"CV std too high ({cv_std:.4f} > {MAX_CV_STD})")
    if weak_classes:
        issues.append(f"Weak classes: {weak_classes} (F1 < {MIN_CLASS_F1})")
    if improvement < MIN_IMPROVEMENT and log["best_oof_f1"] > 0:
        issues.append(f"Improvement too small (+{improvement:.4f} < {MIN_IMPROVEMENT})")
    if submissions_left <= 3:
        issues.append(f"Only {submissions_left} submissions left — be very careful!")

    if log["best_oof_f1"] == 0:
        # First run ever
        print(green("  ✓ SUBMIT — This is your first run, establish a baseline!"))
        should_submit = True
    elif len(issues) == 0 and improvement >= MIN_IMPROVEMENT:
        print(green(f"  ✓ SUBMIT — Improved by +{improvement:.4f}, CV is stable!"))
        should_submit = True
    elif improvement >= MIN_IMPROVEMENT and len(issues) <= 1:
        print(yellow(f"  ? MAYBE — Improved by +{improvement:.4f} but has concerns:"))
        for issue in issues:
            print(yellow(f"    • {issue}"))
        should_submit = None
    else:
        print(red("  ✗ DO NOT SUBMIT — Save your submission. Issues:"))
        for issue in issues:
            print(red(f"    • {issue}"))
        if improvement > 0:
            print(f"    Improvement (+{improvement:.4f}) is too small to risk a submission.")
        elif improvement <= 0:
            print(f"    No improvement over previous best ({log['best_oof_f1']:.5f}).")
        should_submit = False

    print()

    # =========================================================================
    # 7. LOG THIS RUN
    # =========================================================================
    run_entry = {
        "run": run_number,
        "timestamp": datetime.datetime.now().isoformat(),
        "oof_f1": round(oof_f1, 5),
        "cv_mean": round(cv_mean, 5),
        "cv_std": round(cv_std, 5),
        "model_size_mb": round(model_size_mb, 2),
        "predict_seconds": round(predict_seconds, 3),
        "size_penalty": round(size_penalty, 4),
        "speed_penalty": round(speed_penalty, 4),
        "final_score": round(final_score, 5),
        "improvement": round(improvement, 5),
        "weak_classes": weak_classes,
    }
    log["runs"].append(run_entry)

    # Ask user if they submitted
    if should_submit is not False:
        ans = input("Did you submit this model? (y/n): ").strip().lower()
        if ans == "y":
            log["submissions_used"] += 1
            if oof_f1 > log["best_oof_f1"]:
                log["best_oof_f1"] = round(oof_f1, 5)
                print(green(f"  New best OOF F1 recorded: {oof_f1:.5f}"))
            run_entry["submitted"] = True
        else:
            run_entry["submitted"] = False
    else:
        run_entry["submitted"] = False

    save_log(log)
    print(f"\nRun logged to {LOG_FILE}")

    # =========================================================================
    # 8. HISTORY TABLE
    # =========================================================================
    print(bold("\n  RUN HISTORY"))
    print(f"  {'Run':<5} {'OOF F1':<10} {'CV Std':<9} {'Size MB':<9} {'Time(s)':<9} {'Score':<10} {'Submitted'}")
    print(f"  {'─'*5} {'─'*10} {'─'*9} {'─'*9} {'─'*9} {'─'*10} {'─'*9}")
    for r in log["runs"]:
        sub_mark = "✓" if r.get("submitted") else ""
        print(f"  {r['run']:<5} {r['oof_f1']:<10.5f} {r['cv_std']:<9.5f} {r['model_size_mb']:<9.2f} {r['predict_seconds']:<9.3f} {r['final_score']:<10.5f} {sub_mark}")

    print(f"\n  Submissions used: {log['submissions_used']}/{log['submissions_total']}")
    print(f"  Best OOF F1:      {log['best_oof_f1']:.5f}")
    print()


if __name__ == "__main__":
    main()
