# California Housing Price Prediction — Production ML Pipeline

> End-to-end, production-grade regression pipeline on the California Housing dataset — log-transformed target, 12-feature engineering transformer, tuned XGBoost + LightGBM + Random Forest, and a Ridge-meta stacking ensemble achieving R² = 0.8679 and MAE = $26,080.

---

## GitHub Repository Description

> Production ML pipeline on California Housing — log-target transform, advanced geographic and ratio feature engineering (20 total features), RandomizedSearchCV-tuned XGBoost / LightGBM / Random Forest, stacking ensemble with Ridge meta-learner, 95% CI evaluation, and full baseline-to-tuned comparison.

---

## Notebooks

| Notebook | Description |
|---|---|
| `california_housing_pipeline.ipynb` | Baseline pipeline — LR, Decision Tree, Random Forest with GridSearchCV |
| `california_housing_production.ipynb` | Production pipeline — log target, advanced features, XGBoost, LightGBM, stacking ensemble |

---

## Table of Contents

- [Overview](#overview)
- [Dataset](#dataset)
- [What's New vs Baseline Pipeline](#whats-new-vs-baseline-pipeline)
- [Pipeline Steps](#pipeline-steps)
- [Feature Engineering](#feature-engineering)
- [Models & Tuning](#models--tuning)
- [Results](#results)
- [Project Structure](#project-structure)
- [Requirements](#requirements)
- [How to Run](#how-to-run)
- [Key Findings](#key-findings)

---

## Overview

This project extends the baseline California Housing pipeline into a **production-grade regression system**. It introduces a log-transformed target, an advanced custom feature transformer adding 12 engineered features (ratio + geographic + geo-cluster), and replaces GridSearchCV on a single Random Forest with `RandomizedSearchCV` across XGBoost, LightGBM, and Random Forest — combined into a stacking ensemble.

The final stacking ensemble achieves:

| Metric | Value | Real-dollar meaning |
|---|---|---|
| **Test RMSE** | 0.4150 ($100K) | ~$41,500 per prediction |
| **Test MAE** | 0.2608 ($100K) | ~$26,080 on average |
| **Test R²** | 0.8679 | 86.8% of variance explained |
| **95% CI on RMSE** | [0.3942, 0.4347] | Tight — not a lucky split |
| **Mean residual** | 0.0198 | Near-zero — no systematic bias |

This represents a **14.5% RMSE reduction** over the original baseline Random Forest (0.4851 → 0.4150).

---

## Dataset

| Property | Details |
|---|---|
| **Source** | `sklearn.datasets.fetch_california_housing` |
| **Origin** | 1990 US Census |
| **Rows** | 20,640 census block groups |
| **Features** | 8 original → 20 total (after engineering) |
| **Target** | `MedHouseVal` — median house value (in $100,000s) |
| **Missing values** | None |
| **Known artifact** | Hard cap at $500K (5.0) in original data |

### Original Features

| Feature | Description |
|---|---|
| `MedInc` | Median income in block group |
| `HouseAge` | Median house age |
| `AveRooms` | Average rooms per household |
| `AveBedrms` | Average bedrooms per household |
| `Population` | Block group population |
| `AveOccup` | Average household occupancy |
| `Latitude` | Block group latitude |
| `Longitude` | Block group longitude |

### Summary Statistics

| Statistic | MedInc | HouseAge | AveRooms | Population | AveOccup | MedHouseVal |
|---|---|---|---|---|---|---|
| mean | 3.871 | 28.639 | 5.429 | 1,425.5 | 3.071 | 2.069 |
| std | 1.900 | 12.586 | 2.474 | 1,132.5 | 10.386 | 1.154 |
| min | 0.500 | 1.0 | 0.846 | 3.0 | 0.692 | 0.150 |
| 50% | 3.535 | 29.0 | 5.229 | 1,166.0 | 2.818 | 1.797 |
| max | 15.000 | 52.0 | 141.909 | 35,682.0 | 1,243.333 | 5.000 |

---

## What's New vs Baseline Pipeline

| Aspect | Baseline | Production (This Notebook) |
|---|---|---|
| Target transform | Raw | **log1p** — reduces right skew and ceiling effect |
| Engineered features | 3 (ratio only) | **12** (ratio + geographic distances + geo-cluster) |
| Total features | 11 | **20** |
| Models | LR, Decision Tree, RF | LR, RF, **XGBoost, LightGBM** |
| Tuning method | GridSearchCV (RF only) | **RandomizedSearchCV** (RF + XGBoost + LightGBM) |
| Final model | Single RF | **Stacking Ensemble** (RF + XGB + LGBM → Ridge meta) |
| Best Test RMSE | 0.4851 | **0.4150** (−14.5%) |
| Best Test MAE | 0.3196 | **0.2608** (−18.4%) |
| Best R² | 0.8212 | **0.8679** |

---

## Pipeline Steps

```
1. Data Loading & EDA
        ↓
2. Target Distribution Analysis    (raw vs log1p — visualized side-by-side)
        ↓
3. Stratified Train/Test Split      (80/20, stratified by income category)
        |                            Train: 16,512 | Test: 4,128
        ↓
4. Log-transform target             (fit on train only → honest test eval in raw scale)
        ↓
5. Advanced Feature Engineering     (8 original → 20 features via AdvancedFeatureAdder)
        ↓
6. Preprocessing Pipeline           (Imputation → AdvancedFeatureAdder → StandardScaler)
        ↓
7. Baseline Evaluation              (LR, RF, XGBoost, LightGBM — default params, 5-fold CV)
        ↓
8. Hyperparameter Tuning            (RandomizedSearchCV — 60 iters XGB, 60 iters LGBM, 40 iters RF)
        ↓
9. Stacking Ensemble                (Tuned RF + XGB + LGBM → Ridge meta-learner, 5-fold CV)
        ↓
10. Feature Importance Analysis     (XGBoost Gain — all 20 features)
        ↓
11. Final Test Evaluation           (RMSE, MAE, R², 95% CI — all in raw $100K scale)
        ↓
12. Results Summary
```

---

## Feature Engineering

All features are added via a **custom `sklearn` transformer** (`AdvancedFeatureAdder`) implementing `BaseEstimator` + `TransformerMixin`. It is fit on training data only — no leakage.

### Ratio Features (5)

| Feature | Formula | Intuition |
|---|---|---|
| `rooms_per_hh` | `AveRooms / AveOccup` | Spaciousness per occupant |
| `bedrooms_per_room` | `AveBedrms / AveRooms` | Bedroom density |
| `pop_per_hh` | `Population / AveOccup` | Neighborhood crowding |
| `income_per_room` | `MedInc / AveRooms` | Income relative to room count |
| `income_per_hh` | `MedInc / AveOccup` | **Strongest new feature — Gini ≈ 0.30** |

### Geographic Distance Features (5)

| Feature | Computation | Intuition |
|---|---|---|
| `dist_coast` | Haversine distance to nearest coastline point | Coastal premium |
| `dist_SF` | Distance to San Francisco center | Bay Area proximity |
| `dist_LA` | Distance to Los Angeles center | LA metro proximity |
| `dist_SD` | Distance to San Diego center | SD metro proximity |
| `dist_SAC` | Distance to Sacramento center | Capital region proximity |

### Geographic Cluster Feature (1)

| Feature | Computation | Intuition |
|---|---|---|
| `geo_cluster` | KMeans (k=20) on Latitude/Longitude | Neighborhood archetype label |

> The KMeans model is fit on training coordinates only and applied to test — no leakage.

---

## Models & Tuning

### Baseline (default hyperparameters, 5-fold CV on log target)

| Model | CV RMSE (log) | CV RMSE ±std | Test RMSE ($100K) | Test MAE | Test R² |
|---|---|---|---|---|---|
| Linear Regression | 0.1901 | ±0.0016 | 0.6693 | 0.4427 | 0.6563 |
| Random Forest | 0.1371 | ±0.0016 | 0.4557 | 0.2864 | 0.8407 |
| XGBoost | 0.1307 | ±0.0018 | 0.4320 | 0.2773 | 0.8568 |
| LightGBM | 0.1321 | ±0.0016 | 0.4393 | 0.2833 | 0.8520 |

### Hyperparameter Tuning

`RandomizedSearchCV` with 5-fold CV on log-transformed training target.

**XGBoost** (60 iterations):

```python
param_dist = {
    'n_estimators':     randint(300, 1000),
    'learning_rate':    uniform(0.01, 0.15),
    'max_depth':        randint(3, 9),
    'min_child_weight': randint(1, 7),
    'subsample':        uniform(0.6, 0.4),
    'colsample_bytree': uniform(0.5, 0.5),
    'gamma':            uniform(0, 0.5),
    'reg_alpha':        uniform(0, 1),
    'reg_lambda':       uniform(0.5, 2.0),
}
```

Best params: `n_estimators=700, learning_rate=0.084, max_depth=6, min_child_weight=3,`
`subsample=0.804, colsample_bytree=0.588, gamma=0.009, reg_alpha=0.308, reg_lambda=1.585`
Best CV RMSE (log): **0.1279**

**LightGBM** (60 iterations):

```python
param_dist = {
    'n_estimators':       randint(300, 1000),
    'learning_rate':      uniform(0.01, 0.15),
    'max_depth':          randint(3, 9),
    'num_leaves':         randint(20, 150),
    'min_child_samples':  randint(5, 50),
    'subsample':          uniform(0.6, 0.4),
    'colsample_bytree':   uniform(0.5, 0.5),
    'reg_alpha':          uniform(0, 1),
    'reg_lambda':         uniform(0, 2),
}
```

Best params: `n_estimators=707, learning_rate=0.052, max_depth=8, num_leaves=136,`
`min_child_samples=16, subsample=0.785, colsample_bytree=0.523, reg_alpha=0.277, reg_lambda=0.376`
Best CV RMSE (log): **0.1267**

**Random Forest** (40 iterations):

```python
param_dist = {
    'n_estimators':      randint(200, 600),
    'max_features':      randint(4, 14),
    'max_depth':         [None, 10, 20, 30, 40],
    'min_samples_leaf':  randint(1, 10),
    'min_samples_split': randint(2, 15),
    'bootstrap':         [True, False],
}
```

Best params: `bootstrap=False, max_depth=20, max_features=6, min_samples_leaf=1,`
`min_samples_split=12, n_estimators=460`
Best CV RMSE (log): **0.1319**

### Stacking Ensemble

```python
StackingRegressor(
    estimators=[('rf', best_rf), ('xgb', best_xgb), ('lgbm', best_lgbm)],
    final_estimator=Ridge(alpha=1.0),
    cv=5,
    passthrough=False,
)
```

Meta-learner: `Ridge(alpha=1.0)` trained on out-of-fold predictions from the 3 base learners.

---

## Results

### Final Test Set Performance (all metrics in raw $100K scale)

| Model | Test RMSE ($100K) | Test MAE ($100K) | Test R² | 95% CI RMSE |
|---|---|---|---|---|
| Random Forest (Tuned) | 0.4395 | 0.2768 | 0.8518 | [0.4183, 0.4596] |
| XGBoost (Tuned) | 0.4228 | 0.2677 | 0.8629 | [0.4023, 0.4423] |
| LightGBM (Tuned) | 0.4187 | 0.2632 | 0.8655 | [0.3977, 0.4388] |
| **Stacking Ensemble** | **0.4150** | **0.2608** | **0.8679** | **[0.3942, 0.4347]** |

### Improvement Summary

| Metric | Original Baseline RF | Production Stacking | Reduction |
|---|---|---|---|
| Test RMSE | 0.4851 ($100K) | **0.4150** ($100K) | **−14.5%** |
| Test MAE | 0.3196 ($100K) | **0.2608** ($100K) | **−18.4%** |
| Test R² | 0.8212 | **0.8679** | **+4.6 pts** |

### Residual Diagnostics (Stacking Ensemble)

| Metric | Value | Interpretation |
|---|---|---|
| Mean residual | 0.0198 | Near-zero — no systematic bias |
| Std residual | 0.4145 | Spread consistent with RMSE |
| Residual shape | Approximately normal, centered at 0 | Model is well-calibrated |

---

## Feature Importance (XGBoost Tuned — Gain)

| Rank | Feature | Importance (Gain) | Type |
|---|---|---|---|
| 1 | `income_per_hh` | ~0.304 | **Engineered** |
| 2 | `income_per_room` | ~0.149 | **Engineered** |
| 3 | `dist_coast` | ~0.130 | **Engineered** |
| 4 | `rooms_per_hh` | ~0.065 | **Engineered** |
| 5 | `MedInc` | ~0.060 | Original |
| 6 | `dist_nearest_metro` | ~0.039 | **Engineered** |
| 7 | `dist_SAC` | ~0.036 | **Engineered** |
| 8 | `dist_SF` | ~0.035 | **Engineered** |
| 9 | `Latitude` | ~0.028 | Original |
| 10 | `geo_cluster` | ~0.024 | **Engineered** |

> **8 of the top 10 features are engineered.** The raw `MedInc` that dominated the baseline pipeline now ranks 5th behind four derived features — validating the advanced transformer design.

---

## Project Structure

```
california-housing-production/
│
├── california_housing_production.ipynb   # Main notebook (full pipeline)
├── README.md                             # Project documentation
└── requirements.txt                      # Python dependencies
```

---

## Requirements

```
numpy
pandas
matplotlib
scikit-learn
scipy
xgboost
lightgbm
```

Install all dependencies:

```bash
pip install -r requirements.txt
```

Or install directly:

```bash
pip install numpy pandas matplotlib scikit-learn scipy xgboost lightgbm
```

---

## How to Run

**Option 1 — Jupyter Notebook (local):**

```bash
git clone https://github.com/abhinab44/california-housing-production.git
cd california-housing-production
jupyter notebook california_housing_production.ipynb
```

> Dataset loads automatically via `sklearn.datasets.fetch_california_housing` — no manual download required.

**Option 2 — Google Colab:**

[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/)

```python
# In Colab, install additional libraries first:
!pip install xgboost lightgbm
```

---

## Key Findings

- **Engineered features dominate** — `income_per_hh` (Gain ≈ 0.30) replaces raw `MedInc` as the most predictive signal, and 8 of the top 10 features by XGBoost Gain are engineered. This validates the `AdvancedFeatureAdder` design comprehensively.
- **Log-transforming the target** reduces the skew from the $500K hard cap and stabilizes gradient-based model training, contributing meaningfully to the ~14.5% RMSE improvement over the raw-target baseline.
- **Geographic distance features matter** — `dist_coast`, `dist_SF`, `dist_SAC` all rank in the top 8, confirming that proximity to coast and major metro areas captures price signal that raw Latitude/Longitude alone cannot express cleanly.
- **Stacking consistently beats individual tuned models** — the Ridge meta-learner effectively learns to blend the base learner predictions, adding incremental gains (0.4150 vs 0.4187 for the best single model LightGBM) with minimal computational overhead.
- **LightGBM edges XGBoost** at baseline and after tuning on this dataset, likely due to its leaf-wise tree growth and better handling of high-cardinality geographic features.
- **Residuals are well-calibrated** — mean residual of 0.0198 indicates no systematic over- or under-prediction across the range, and the residual distribution is approximately normal and centered at zero.
- **The $500K census cap** creates a visible vertical band in the actual-vs-predicted plot and a rightward fan in the residual plot at high predicted values — this is a known data artifact, not a model failure.
- **95% CI [0.3942, 0.4347]** — the narrow interval confirms results are stable and not driven by favorable test-set sampling.

---

## Concepts Demonstrated

- Log-transform of skewed target with inverse-transform for honest evaluation
- Custom `sklearn` transformer with ratio, geographic, and clustering features (`AdvancedFeatureAdder`)
- Haversine distance computation for geographic feature engineering
- KMeans geo-clustering as a supervised feature (fit on train only)
- `RandomizedSearchCV` for efficient large hyperparameter space exploration
- `StackingRegressor` with tuned base learners and Ridge meta-estimator
- 5-fold cross-validation on log target; final evaluation in raw scale
- Statistical 95% confidence intervals on test RMSE

---

## License

This project is open-source under the [MIT License](LICENSE).

---

*Built with Python 3.10 · scikit-learn · XGBoost · LightGBM · pandas · NumPy · Matplotlib · SciPy*
