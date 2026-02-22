# Insurance Bundle Recommender

ML-powered insurance coverage bundle prediction system built for the DataQuest Hackathon.

## Overview

Predicts the optimal insurance coverage bundle (0–9) for policyholders based on demographic, financial, and policy attributes using a LightGBM classifier with 47 engineered features.

| Metric | Value |
|--------|-------|
| Model | LightGBM (80 trees, 0.87 MB) |
| Features | 47 engineered from 28 raw columns |
| OOF Macro-F1 | 0.685 |
| Inference | ~0.09s local, ~2.4s server |

## Quick Start

### Option 1: Docker (recommended)

```bash
docker-compose up --build
# Open http://localhost:8000
```

### Option 2: Local

```bash
pip install -r api/requirements.txt
uvicorn api.main:app --reload
# Open http://localhost:8000
```

## Project Structure

```
├── api/                          # FastAPI backend
│   ├── main.py                   # App, endpoints, feature pipeline
│   ├── schemas.py                # Pydantic request/response models
│   └── requirements.txt          # Python dependencies
├── frontend/                     # Web UI
│   ├── index.html                # Single-page form interface
│   └── style.css                 # Styles
├── docs/                         # Documentation
│   └── technical_report.md       # Full technical report
├── tests/                        # API tests
│   └── test_api.py               # pytest test suite
├── .github/workflows/ci.yml      # CI/CD pipeline
├── Dockerfile                    # Container image
├── docker-compose.yml            # Container orchestration
├── train.py                      # Model training pipeline
├── solution.py                   # Competition submission code
├── evaluate.py                   # Local evaluation script
├── model.pkl                     # Trained model artifact
├── train.csv                     # Training data
└── test.csv                      # Test data
```

## API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/` | GET | Web UI |
| `/health` | GET | Model status & info |
| `/predict` | POST | Single prediction |
| `/predict/batch` | POST | Batch predictions (≤10k) |
| `/docs` | GET | Interactive Swagger docs |

## Example Request

```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "User_ID": "USR_001",
    "Employment_Status": "Employed_FullTime",
    "Estimated_Annual_Income": 45000,
    "Region_Code": "ABJ",
    "Adult_Dependents": 2,
    "Child_Dependents": 1,
    "Deductible_Tier": "Tier_2_Moderate_Ded"
  }'
```

## Running Tests

```bash
pip install pytest httpx
pytest tests/ -v
```

## Training

```bash
python train.py
# Outputs: model.pkl, run_log.json
```

## Technical Details

See [docs/technical_report.md](docs/technical_report.md) for:
- Architecture diagrams
- Feature engineering details
- Model selection rationale
- Optimization journey (12 runs)
- Failed approaches & lessons learned

---

## Target Variable Mapping

| ID    | Bundle Name            |
| :---- | :--------------------- |
| **0** | `Auto_Comprehensive`   |
| **1** | `Auto_Liability_Basic` |
| **2** | `Basic_Health`         |
| **3** | `Family_Comprehensive` |
| **4** | `Health_Dental_Vision` |
| **5** | `Home_Premium`         |
| **6** | `Home_Standard`        |
| **7** | `Premium_Health_Life`  |
| **8** | `Renter_Basic`         |
| **9** | `Renter_Premium`       |

---

## Columns

### Identifiers & Target

- `User_ID` — Unique customer identifier.
- `Purchased_Coverage_Bundle` — **[TARGET]** Integer 0–9 (see mapping above). Present in `train.csv` only.

### Demographics & Financials

- `Adult_Dependents` — Number of adults covered.
- `Child_Dependents` — Number of children covered.
- `Infant_Dependents` — Number of infants covered.
- `Estimated_Annual_Income` — Estimated yearly household income.
- `Employment_Status` — Working arrangement of the primary applicant.
- `Region_Code` — Anonymized geographic location.

### Customer History & Risk Profile

- `Existing_Policyholder` — Already has another active policy?
- `Previous_Claims_Filed` — Total prior claims.
- `Years_Without_Claims` — Consecutive claim-free years.
- `Previous_Policy_Duration_Months` — Duration of prior policy.
- `Policy_Cancelled_Post_Purchase` — History of early cancellations?

### Policy Details & Preferences

- `Deductible_Tier` — Chosen deductible level.
- `Payment_Schedule` — Premium payment frequency.
- `Vehicles_on_Policy` — Number of vehicles covered.
- `Custom_Riders_Requested` — Special coverage add-ons.
- `Grace_Period_Extensions` — Payment deadline extensions.

### Sales & Underwriting

- `Days_Since_Quote` — Days between quote and finalization.
- `Underwriting_Processing_Days` — Days for underwriting approval.
- `Policy_Amendments_Count` — Quote modifications before signing.
- `Acquisition_Channel` — How the policy was sold.
- `Broker_Agency_Type` — Scale of the brokerage firm.
- `Broker_ID` — Sales agent identifier.
- `Employer_ID` — Employer identifier.

### Timeline

- `Policy_Start_Year`, `Policy_Start_Month`, `Policy_Start_Week`, `Policy_Start_Day`

---

## Scoring

Your **final score** combines three factors:

```
score = macro_f1 × size_penalty × duration_penalty

where:
  macro_f1         = (1/N) Σ F1_i   (unweighted average over all N classes)
  size_penalty     = max(0.5, 1 − model_size_mb / 200)
  duration_penalty = max(0.5, 1 − predict_seconds / 10)
```

### Macro F1-Score

F1 is the harmonic mean of precision and recall. The **Macro** variant computes F1 per class independently, then takes the unweighted average — so every bundle matters equally, regardless of frequency.

### Size Penalty

Penalizes large model files. A 0 MB model scores 1.0 (no penalty); a 100 MB model scores 0.5. The penalty floors at 0.5 — no model, however large, loses more than half its F1.

### Latency Penalty

Penalizes slow `predict()` calls. 0 s → 1.0; 5 s → 0.5. Floors at 0.5. Only `predict()` is timed — `preprocess()` and `load_model()` run before the clock starts.

> **Tip:** Don't just chase F1. A lightweight, fast model with slightly lower F1 can outscore a bloated one.

---

## Submission Format

Your `.zip` must contain exactly **3 files** at the root (no nested folders):

| File               | Required | Notes                                                                  |
| :----------------- | :------: | :--------------------------------------------------------------------- |
| `solution.py`      |   yes    | Must export `preprocess`, `load_model`, `predict`.                     |
| `model.*`          |   yes    | Exactly one file starting with `model` (e.g. `model.pkl`, `model.pt`). |
| `requirements.txt` |   yes    | Extra pip packages. Can be empty but must exist.                       |

### What `predict()` must return

A **pandas DataFrame** with two columns:

```
User_ID,Purchased_Coverage_Bundle
USR_060868,7
USR_060869,2
USR_060870,4
...
```

Every `User_ID` from the test set must be present. The `Purchased_Coverage_Bundle` column must contain integer predictions (0–9).

---

## Pre-installed Packages

The evaluation container already has these installed — you don't need to list them in `requirements.txt`:

| Package      | Version     |
| :----------- | :---------- |
| numpy        | 1.26.4      |
| scipy        | 1.11.4      |
| pandas       | 2.1.4       |
| scikit-learn | 1.3.2       |
| xgboost      | 2.0.3       |
| lightgbm     | 4.6.0       |
| catboost     | 1.2.3       |
| torch        | 2.6.0 (CPU) |
| torchvision  | 0.21.0      |
| torchaudio   | 2.6.0       |
| tensorflow   | 2.14.0      |
| joblib       | 1.3.2       |

You **can** add extra packages via `requirements.txt`. You **cannot** override pre-installed versions.

---

## Limits

| Constraint           | Value  |
| :------------------- | :----- |
| Max upload size      | 50 MB  |
| Container memory     | 1 GB   |
| Container CPU        | 1 core |
| Execution timeout    | 120 s  |
| Submissions per team | 20     |

Good luck!
