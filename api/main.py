"""
Insurance Bundle Recommender — FastAPI Application.

Endpoints:
  GET  /health          → Health check & model info
  POST /predict         → Single policy prediction
  POST /predict/batch   → Batch prediction (multiple policies)
"""

import os
import sys
import time
import numpy as np
import pandas as pd
import joblib
from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse

# Add parent directory to path so we can import solution.py
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from api.schemas import (
    PolicyInput,
    PredictionResult,
    BatchPredictionRequest,
    BatchPredictionResponse,
    HealthResponse,
    BUNDLE_DESCRIPTIONS,
)

# ---------------------------------------------------------------------------
# Global state
# ---------------------------------------------------------------------------
MODEL_STATE = {
    "model": None,
    "artifacts": None,
    "feature_cols": None,
    "label_encoders": None,
    "freq_maps": None,
    "freq_cols": None,
    "model_size_mb": 0.0,
}

MODEL_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "model.pkl")
VERSION = "1.0.0"


def _load_model():
    """Load model artifacts into memory."""
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(f"Model not found at {MODEL_PATH}")

    artifacts = joblib.load(MODEL_PATH)
    MODEL_STATE["artifacts"] = artifacts
    MODEL_STATE["model"] = artifacts["model"]
    MODEL_STATE["feature_cols"] = artifacts["feature_cols"]
    MODEL_STATE["label_encoders"] = artifacts["label_encoders"]
    MODEL_STATE["freq_maps"] = artifacts["freq_maps"]
    MODEL_STATE["freq_cols"] = artifacts["freq_cols"]
    MODEL_STATE["model_size_mb"] = os.path.getsize(MODEL_PATH) / (1024 * 1024)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load model on startup."""
    _load_model()
    print(f"Model loaded ({MODEL_STATE['model_size_mb']:.2f} MB, "
          f"{len(MODEL_STATE['feature_cols'])} features)")
    yield
    MODEL_STATE["model"] = None


# ---------------------------------------------------------------------------
# App setup
# ---------------------------------------------------------------------------
app = FastAPI(
    title="Insurance Bundle Recommender",
    description=(
        "ML-powered API that predicts the optimal insurance coverage bundle "
        "(0-9) for policyholders based on their demographic, financial, and "
        "policy attributes. Built with LightGBM and FastAPI."
    ),
    version=VERSION,
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Serve frontend static files
FRONTEND_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "frontend")
if os.path.isdir(FRONTEND_DIR):
    app.mount("/static", StaticFiles(directory=FRONTEND_DIR), name="static")


# ---------------------------------------------------------------------------
# Feature engineering (mirrors solution.py exactly)
# ---------------------------------------------------------------------------
def _feature_engineering(df: pd.DataFrame) -> pd.DataFrame:
    """Identical to solution.py _feature_engineering."""
    # Import from solution.py to avoid duplication
    from solution import _feature_engineering as fe
    return fe(df)


def _preprocess_for_api(records: list[dict]) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[str]]:
    """
    Convert list of dicts → feature matrix + rule flags.
    Returns: (X, rule9, rule8, user_ids)
    """
    df = pd.DataFrame(records)
    df = _feature_engineering(df)

    user_ids = df["User_ID"].tolist()
    work = df.drop(columns=["User_ID"], errors="ignore")

    # Rule flags
    rule9 = work["_rule_class9"].values.copy() if "_rule_class9" in work.columns else np.zeros(len(work), dtype=np.int8)
    rule8 = work["_rule_class8"].values.copy() if "_rule_class8" in work.columns else np.zeros(len(work), dtype=np.int8)

    # Label encoding
    for c, mapping in MODEL_STATE["label_encoders"].items():
        if c not in work.columns:
            continue
        work[c] = work[c].astype(str).fillna("__MISSING__").map(mapping).fillna(-1).astype(int)

    # Frequency encoding
    for c, fmap in MODEL_STATE["freq_maps"].items():
        if c in work.columns:
            work[f"{c}_freq"] = work[c].map(fmap).fillna(0).astype(np.float32)

    # Drop raw ID columns
    for c in MODEL_STATE["freq_cols"]:
        if c in work.columns:
            work.drop(columns=[c], inplace=True)

    # Ensure all feature columns exist
    for c in MODEL_STATE["feature_cols"]:
        if c not in work.columns:
            work[c] = 0

    X = work[MODEL_STATE["feature_cols"]].values.astype(np.float32)
    return X, rule9, rule8, user_ids


def _predict_bundle(X, rule9, rule8) -> tuple[np.ndarray, np.ndarray]:
    """Run model inference. Returns (predictions, probabilities)."""
    proba = MODEL_STATE["model"].predict_proba(X)
    preds = np.argmax(proba, axis=1)

    # Hardcoded rules for rare classes
    preds[rule9 == 1] = 9
    preds[rule8 == 1] = 8

    return preds, proba


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------
@app.get("/", include_in_schema=False)
async def root():
    """Serve the frontend."""
    index_path = os.path.join(FRONTEND_DIR, "index.html")
    if os.path.exists(index_path):
        return FileResponse(index_path)
    return {"message": "Insurance Bundle Recommender API", "docs": "/docs"}


@app.get("/health", response_model=HealthResponse, tags=["System"])
async def health_check():
    """
    Health check endpoint. Returns model status, size, and version.
    """
    return HealthResponse(
        status="healthy" if MODEL_STATE["model"] is not None else "model_not_loaded",
        model_loaded=MODEL_STATE["model"] is not None,
        model_size_mb=round(MODEL_STATE["model_size_mb"], 2),
        num_features=len(MODEL_STATE["feature_cols"] or []),
        version=VERSION,
    )


@app.post("/predict", response_model=PredictionResult, tags=["Prediction"])
async def predict_single(policy: PolicyInput):
    """
    Predict the optimal insurance coverage bundle for a single policyholder.

    Returns the predicted bundle (0-9), confidence score, and per-class probabilities.
    """
    if MODEL_STATE["model"] is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    record = policy.model_dump()
    X, rule9, rule8, user_ids = _preprocess_for_api([record])
    preds, proba = _predict_bundle(X, rule9, rule8)

    pred_class = int(preds[0])
    confidence = float(np.max(proba[0]))

    return PredictionResult(
        User_ID=user_ids[0],
        predicted_bundle=pred_class,
        confidence=round(confidence, 4),
        probabilities={
            f"bundle_{i} ({BUNDLE_DESCRIPTIONS.get(i, 'Unknown')})": round(float(proba[0][i]), 4)
            for i in range(10)
        },
    )


@app.post("/predict/batch", response_model=BatchPredictionResponse, tags=["Prediction"])
async def predict_batch(request: BatchPredictionRequest):
    """
    Predict insurance bundles for multiple policyholders in a single request.

    Accepts up to 10,000 policies per batch. Returns predictions with
    confidence scores and probabilities for each.
    """
    if MODEL_STATE["model"] is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    if len(request.policies) > 10000:
        raise HTTPException(status_code=400, detail="Max 10,000 policies per batch")

    if len(request.policies) == 0:
        raise HTTPException(status_code=400, detail="At least 1 policy required")

    records = [p.model_dump() for p in request.policies]
    X, rule9, rule8, user_ids = _preprocess_for_api(records)
    preds, proba = _predict_bundle(X, rule9, rule8)

    predictions = []
    for i in range(len(preds)):
        predictions.append(
            PredictionResult(
                User_ID=user_ids[i],
                predicted_bundle=int(preds[i]),
                confidence=round(float(np.max(proba[i])), 4),
                probabilities={
                    f"bundle_{j}": round(float(proba[i][j]), 4)
                    for j in range(10)
                },
            )
        )

    return BatchPredictionResponse(
        predictions=predictions,
        count=len(predictions),
        model_version=VERSION,
    )
