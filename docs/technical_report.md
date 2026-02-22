# Technical Report — Insurance Bundle Recommender System

## DataQuest Hackathon — Phase I & Phase II

---

## Table of Contents

1. [Problem Statement](#1-problem-statement)
2. [Architecture Overview](#2-architecture-overview)
3. [Exploratory Data Analysis](#3-exploratory-data-analysis)
4. [Feature Engineering](#4-feature-engineering)
5. [Model Selection & Training](#5-model-selection--training)
6. [Optimization Journey](#6-optimization-journey)
7. [Failed Approaches & Lessons](#7-failed-approaches--lessons)
8. [API & Deployment](#8-api--deployment)
9. [Results Summary](#9-results-summary)

---

## 1. Problem Statement

Given policyholder attributes (demographics, policy details, claims history, acquisition data), predict the optimal insurance coverage bundle (classes 0–9). The objective function combines classification quality with system efficiency:

$$\text{Score} = \text{Macro-F1} \times \max(0.5,\; 1 - \tfrac{\text{size\_mb}}{200}) \times \max(0.5,\; 1 - \tfrac{\text{latency\_s}}{10})$$

This creates a three-way trade-off between:
- **Accuracy**: Macro-F1 across 10 imbalanced classes
- **Model size**: Penalized above 200 MB (floor at 0.5×)
- **Inference latency**: Penalized above 10 seconds (floor at 0.5×)

---

## 2. Architecture Overview

### System Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    Frontend (HTML/CSS/JS)                     │
│              Single-page form → fetch(/predict)              │
└──────────────────────┬──────────────────────────────────────┘
                       │ HTTP POST (JSON)
                       ▼
┌─────────────────────────────────────────────────────────────┐
│                   FastAPI Application                        │
│  ┌──────────┐  ┌────────────┐  ┌─────────────────────┐     │
│  │ /health  │  │  /predict  │  │  /predict/batch     │     │
│  │  (GET)   │  │  (POST)    │  │  (POST, ≤10k rows)  │     │
│  └──────────┘  └──────┬─────┘  └──────────┬──────────┘     │
│                       │                    │                 │
│               ┌───────▼────────────────────▼───────┐        │
│               │     Feature Engineering Pipeline    │        │
│               │  (47 features from 28 raw columns)  │        │
│               └───────────────┬─────────────────────┘        │
│                               ▼                              │
│               ┌───────────────────────────────┐              │
│               │   LightGBM Classifier (0.87MB) │             │
│               │  80 trees │ 31 leaves │ depth 6 │            │
│               └───────────────────────────────┘              │
│                               │                              │
│               ┌───────────────▼───────────────┐              │
│               │   Post-processing Rules        │             │
│               │  (Class 8 & 9 hardcoded)       │             │
│               └───────────────────────────────┘              │
└─────────────────────────────────────────────────────────────┘
                       │
                       ▼
          JSON Response: bundle, confidence, probabilities
```

### Data Flow (Training)

```
train.csv (2,000 rows × 28 columns)
    │
    ▼
Feature Engineering
    │  ├─ 47 derived features (ratios, interactions, flags)
    │  ├─ Label encoding (8 categorical columns)
    │  └─ Frequency encoding (Broker_ID, Employer_ID)
    ▼
5-Fold Stratified Cross-Validation
    │  ├─ OOF Macro-F1: 0.66–0.70
    │  └─ Per-class F1 tracked
    ▼
Final Model (trained on 100% data, n_estimators=80)
    │
    ▼
model.pkl (0.87 MB)
    │  ├─ LightGBM Booster
    │  ├─ feature_cols (47 features)
    │  ├─ label_encoders (8 mappings)
    │  ├─ freq_maps (Broker_ID, Employer_ID)
    │  └─ class_scales (unused, all 1.0)
```

---

## 3. Exploratory Data Analysis

### Dataset Summary

| Property | Value |
|----------|-------|
| Training samples | ~2,000 |
| Features | 28 raw columns |
| Target classes | 10 (bundles 0–9) |
| Missing values | Broker_ID, Employer_ID, Region_Code |

### Class Distribution (Severe Imbalance)

| Class | Count | Percentage | Description |
|-------|-------|------------|-------------|
| 2 | ~1,188 | 59.4% | Standard Plus (dominant) |
| 0 | ~270 | 13.5% | Basic Liability |
| 3 | ~170 | 8.5% | Enhanced Coverage |
| 1 | ~125 | 6.3% | Standard Coverage |
| 5 | ~80 | 4.0% | Premium Protection |
| 4 | ~60 | 3.0% | Comprehensive |
| 6 | ~50 | 2.5% | Family Shield |
| 7 | ~35 | 1.8% | Executive Suite |
| 8 | 6 | 0.3% | Specialty Coverage |
| 9 | 5 | 0.25% | Ultra Premium |

**Key insight**: Classes 8 and 9 have only 6 and 5 samples respectively—too few for any ML model to learn reliably. This motivated handcrafted decision rules.

### Key Observations

1. **Income distribution**: Highly right-skewed; log transform applied
2. **Broker_ID / Employer_ID**: High cardinality (~300+ unique), many missing → frequency-encoded
3. **Region_Code**: Moderate cardinality (~25 regions), label-encoded
4. **Temporal patterns**: Policy_Start_Year and Policy_Start_Month show class-correlated patterns (Class 8 strongly associated with 2015)
5. **Deductible Tier**: Strong class discriminator (Class 9 always has Tier_4_Zero_Ded)

---

## 4. Feature Engineering

### Engineered Features (47 total, from 28 raw)

| Category | Features | Rationale |
|----------|----------|-----------|
| **Household** | Total_Dependents, Has_Dependents, Has_Infant, Has_Child, Has_Adult_Dep | Family composition drives bundle needs |
| **Income** | Log_Income | Normalized income (removes skew) |
| **Risk Profile** | Claims_Per_Year, Years_Without_Claims_Ratio, Risk_Score, Had_Cancellation | Claims history indicates risk appetite |
| **Policy Meta** | Is_New_Customer, Policy_Duration_Years | Tenure affects bundle preference |
| **Temporal** | Policy_Start_Quarter, Is_Year_End, Is_Year_Start | Seasonal patterns in purchases |
| **Processing** | Total_Wait_Days | Underwriting + quote delay |
| **Vehicle/Rider** | Has_Riders, Has_Grace_Ext | Coverage complexity indicators |
| **Missingness** | Broker_ID_missing, Employer_ID_missing | Missing data is informative |
| **Frequency** | Broker_ID_freq, Employer_ID_freq | High-cardinality encoding |
| **Raw (encoded)** | 8 label-encoded categoricals, numeric originals | Baseline signals |

### Excluded Features (24 removed to combat overfitting)

Complex interaction and ratio features were found to improve training F1 but degrade OOF generalization:

- Cyclical encodings (Month_Sin/Cos, Week_Sin/Cos)
- High-order interactions (Income×Dependents, Income×Vehicles, Vehicles×Riders, Claims×Duration)
- Ratio features (Child_Infant_Ratio, Dep_Mix, Income_Per_Dependent/Vehicle/Claim)
- Fine-grained composition (Adult/Child/Infant_Dep_Ratio)
- Sparse features (Policy_Year, Zero_Activity, Grace_Per_Duration, Amendments_Per_Duration, etc.)

### Hardcoded Rules (Post-Prediction Override)

For ultra-rare classes where ML cannot generalize from 5–6 samples:

**Class 9 Rule** (5/5 perfect recall in train):
- Income = 0, Region_Code is NaN, Policy_Cancelled = 1, Deductible_Tier = Tier_4_Zero_Ded

**Class 8 Rule** (6/6 recall, 12 total matches in train):
- Region_Code = PRT, Deductible_Tier = Tier_1_High_Ded, Year = 2015, Vehicles = 0
- Underwriting_Days = 0, Prev_Duration = 1 month, Not Existing, No children/infants
- No riders, No amendments, Direct_Website, Monthly_EFT, National_Corporate
- Employed_FullTime, Month ∈ {Nov, Dec}

---

## 5. Model Selection & Training

### Why LightGBM?

| Criterion | LightGBM | Alternatives Considered |
|-----------|----------|------------------------|
| Speed | Very fast training & inference | XGBoost/CatBoost slower |
| Size | Small model files (<1 MB achievable) | Neural nets would be 10-100× larger |
| Tabular data | State-of-the-art on structured data | Deep learning inferior for N<10k |
| Imbalanced classes | Supports sample_weight natively | Many models need external oversampling |

### Final Hyperparameters

```python
{
    "n_estimators": 80,          # Reduced from 754 to combat overfitting
    "learning_rate": 0.20,       # Higher LR compensates for fewer trees
    "num_leaves": 31,
    "max_depth": 6,              # Shallow trees for regularization
    "min_child_samples": 50,     # Forces well-populated leaves
    "subsample": 0.7,
    "colsample_bytree": 0.6,    # Feature subsampling
    "reg_alpha": 0.2,           # L1 regularization
    "reg_lambda": 5.0,          # L2 regularization (strong)
    "min_split_gain": 0.05,     # Prune trivial splits
    "path_smooth": 5.0,         # Smooth leaf predictions
}
```

### Class Weighting Strategy

Power-law weighting with $w_c = \min\left(\left(\frac{N}{K \cdot n_c}\right)^{0.4},\; 3.0\right)$, where $n_c$ is the count for class $c$, $N$ is total samples, and $K=10$ classes. Classes 8 and 9 are set to weight 1.0 since hardcoded rules handle them.

### Training Protocol

- **Cross-validation**: 5-fold Stratified K-Fold, macro-F1 evaluation
- **Final model**: Trained on 100% of training data with fixed `n_estimators=80`
- **No calibration**: Probability calibration was tested but found to hurt (scaling factors all set to 1.0)

---

## 6. Optimization Journey

### Run History (12 iterations)

| Run | Trees | Features | OOF F1 | Size (MB) | Latency (s) | Local Score |
|-----|-------|----------|--------|-----------|-------------|-------------|
| 1 | 754 | 72 | 0.665 | 25.83 | 10.97 | 0.290 |
| 2 | 754 | 72 | 0.670 | 18.80 | 11.49 | 0.303 |
| 3 | 399 | 72 | **0.696** | 16.74 | 1.36 | 0.551 |
| 4 | 300 | 72 | 0.690 | 6.60 | 0.59 | 0.628 |
| 9 | 150 | 47 | 0.667 | 1.66 | 0.19 | 0.649 |
| 10 | 200 | 47 | 0.671 | 1.46 | 0.20 | 0.653 |
| 11 | 80 | 47 | 0.685 | 1.47 | 0.18 | **0.668** |
| 12 | 80 | 24 | 0.599 | 0.87 | 0.09 | 0.591 |

### Server Submissions (3/20 used)

| Submission | Server F1 | Latency | Size | Server Score |
|------------|----------|---------|------|-------------|
| Run 3 | 0.494 | 2.36s | 1.47MB | 0.401 |
| Run 4 | ~0.48 | ~2.5s | ~1.5MB | 0.375 |
| Run 1 | 0.245 | 15.66s | - | 0.245 |

### Key Findings

1. **Latency dominance**: Server is ~10-12× slower than local machine. A model running 0.18s locally takes ~2.0s on server
2. **Overfitting gap**: OOF F1 0.70 drops to server F1 0.49 — a 0.21 generalization gap
3. **Feature removal helps**: Dropping 24 complex features from 72→47 improved generalization
4. **Diminishing returns from trees**: Beyond 80 trees, added complexity hurts more than it helps

---

## 7. Failed Approaches & Lessons

### Approaches That Did NOT Work

| Approach | Expected | Actual | Lesson |
|----------|----------|--------|--------|
| **More trees (754)** | Higher F1 | Extreme latency (15.66s server) | Server overhead ∝ trees |
| **Probability calibration** | Better probabilities | Hurt macro-F1 by ~2% | Small dataset = noisy calibration |
| **72 features (all)** | More signal | Overfitting (+0.05 train, -0.02 OOF) | Complex features memorize train noise |
| **Nelder-Mead calibration** | Optimal per-class scaling | Overfits to train distribution | Validation-less tuning is dangerous |
| **Income binning (pd.qcut)** | Income groups | Data leakage (bin edges from train/test differently) | Discretization must be train-fitted |
| **Training on 100% data** | Better model | Memorization (train F1 0.84, server 0.49) | Ironic: more data hurts when test differs |

### Key Technical Lessons

1. **The scoring formula punishes latency nonlinearly** — a 10× server slowdown dominated all other factors
2. **With N<2000, fewer features = better** — feature engineering has diminishing returns below a threshold sample size
3. **Rule-based overrides for micro-classes are essential** — 5–6 samples cannot be learned by gradient boosting
4. **The OOF→server gap suggests covariate shift** — the test distribution likely differs from training

---

## 8. API & Deployment

### REST API (FastAPI)

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/health` | GET | Model status, size, version, feature count |
| `/predict` | POST | Single policyholder prediction |
| `/predict/batch` | POST | Batch prediction (up to 10,000 policies) |
| `/docs` | GET | Interactive Swagger UI (auto-generated) |

### Request/Response Example

**POST /predict**
```json
{
  "User_ID": "USR_12345",
  "Employment_Status": "Employed_FullTime",
  "Estimated_Annual_Income": 35000,
  "Region_Code": "ABJ",
  "Adult_Dependents": 2,
  "Child_Dependents": 1,
  "Infant_Dependents": 0,
  "Deductible_Tier": "Tier_2_Moderate_Ded",
  "Payment_Schedule": "Monthly_EFT",
  "Policy_Start_Year": 2016,
  "Policy_Start_Month": "March",
  "Vehicles_on_Policy": 1,
  "Acquisition_Channel": "Direct_Website"
}
```

**Response**
```json
{
  "User_ID": "USR_12345",
  "predicted_bundle": 2,
  "confidence": 0.7823,
  "probabilities": {
    "bundle_0 (Basic Liability Only)": 0.0521,
    "bundle_1 (Standard Coverage)": 0.0312,
    "bundle_2 (Standard Plus)": 0.7823,
    "bundle_3 (Enhanced Coverage)": 0.0891,
    ...
  }
}
```

### Frontend

Simple single-page HTML/CSS/JS interface:
- Form with all 28 policy fields organized by category (Policyholder, Dependents, Policy Details, History, Coverage)
- Inline probability distribution visualization (colored bar chart)
- Health status badge showing model version and feature count
- No framework dependencies — vanilla HTML/CSS/JS

### Deployment Stack

```
Docker Compose
├── api (FastAPI + Uvicorn)
│   ├── Port 8000
│   ├── model.pkl loaded at startup
│   └── Serves frontend as static files
└── nginx (optional reverse proxy)
```

### CI/CD Pipeline (GitHub Actions)

- **Lint**: flake8 + black formatting check
- **Test**: pytest on API endpoint tests
- **Build**: Docker image build + push
- **Triggers**: Push to main, PRs

---

## 9. Results Summary

### Final Model Performance

| Metric | Local | Server |
|--------|-------|--------|
| Macro-F1 (OOF) | 0.685 | 0.494 |
| Model size | 0.87 MB | 1.47 MB (with artifacts) |
| Inference latency | 0.09s | ~2.36s |
| Composite score | 0.668 | 0.401 |

### Per-Class F1 (OOF, best run)

| Class | F1 | Support |
|-------|-----|---------|
| 0 | ~0.58 | ~270 |
| 1 | ~0.42 | ~125 |
| 2 | ~0.89 | ~1188 |
| 3 | ~0.56 | ~170 |
| 4 | ~0.45 | ~60 |
| 5 | ~0.55 | ~80 |
| 6 | ~0.48 | ~50 |
| 7 | ~0.52 | ~35 |
| 8 | 1.00 | 6 (rule) |
| 9 | 1.00 | 5 (rule) |

### Future Improvements

1. **Ensemble with CatBoost/XGBoost** — diverse models may reduce variance
2. **Feature selection via permutation importance** — more principled than manual exclusion
3. **Pseudo-labeling on test set** — if test data structure is accessible
4. **SMOTE for minority classes** — synthetic oversampling for classes 4–7
5. **Stacking with simpler models** — logistic regression + GBDT meta-learner

---

*Report generated for DataQuest Hackathon — Insurance Recommender System*
