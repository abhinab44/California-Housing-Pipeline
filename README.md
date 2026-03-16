# California Housing Price Prediction — End-to-End ML Pipeline

> Predict California median house values using a fully reproducible scikit-learn pipeline — from raw census data to tuned Random Forest with confidence-interval evaluation.

---

## GitHub Repository Description

> End-to-end ML pipeline on the California Housing dataset — EDA, stratified splitting, custom feature engineering, sklearn preprocessing pipelines, cross-validated Linear Regression / Decision Tree / Random Forest, GridSearchCV tuning, feature importance analysis, and confidence-interval test evaluation.

---

## Table of Contents

- [Overview](#overview)
- [Dataset](#dataset)
- [Pipeline Steps](#pipeline-steps)
- [Feature Engineering](#feature-engineering)
- [Models](#models)
- [Results](#results)
- [Project Structure](#project-structure)
- [Requirements](#requirements)
- [How to Run](#how-to-run)
- [Key Findings](#key-findings)

---

## Overview

This project implements a complete, production-style machine learning pipeline to predict median house values across California census block groups. It demonstrates best practices in data preprocessing, custom transformer design, model comparison via cross-validation, and hyperparameter tuning — all built with `scikit-learn`'s `Pipeline` and `ColumnTransformer` API.

---

## Dataset

| Property | Details |
|---|---|
| **Source** | `sklearn.datasets.fetch_california_housing` |
| **Origin** | 1990 US Census |
| **Rows** | 20,640 census block groups |
| **Features** | 8 original + 3 engineered |
| **Target** | `MedHouseVal` — Median house value (in $100,000s) |

### Original Features

| Feature | Description |
|---|---|
| `MedInc` | Median income in block group |
| `HouseAge` | Median house age |
| `AveRooms` | Average number of rooms per household |
| `AveBedrms` | Average number of bedrooms per household |
| `Population` | Block group population |
| `AveOccup` | Average household occupancy |
| `Latitude` | Block group latitude |
| `Longitude` | Block group longitude |

---

## Pipeline Steps

```
1. Data Loading & EDA
        ↓
2. Stratified Train/Test Split  (stratified by income category)
        ↓
3. Feature Engineering          (3 derived features via custom transformer)
        ↓
4. Preprocessing Pipeline       (Imputation → Custom Transformer → Standard Scaling)
        ↓
5. Model Training & Cross-Validation  (Linear Regression, Decision Tree, Random Forest)
        ↓
6. Hyperparameter Tuning        (GridSearchCV on Random Forest)
        ↓
7. Feature Importance Analysis
        ↓
8. Final Test Evaluation + 95% Confidence Interval
        ↓
9. Results Summary
```

---

## Feature Engineering

Three derived features are added via a **custom `sklearn` transformer** (`BaseEstimator` + `TransformerMixin`):

| Engineered Feature | Formula | Intuition |
|---|---|---|
| `rooms_per_hh` | `AveRooms / AveOccup` | Spaciousness per household member |
| `bedrooms_per_room` | `AveBedrms / AveRooms` | Bedroom density ratio |
| `population_per_hh` | `Population / AveOccup` | Crowding per household |

These features were shown to contribute meaningfully to Random Forest predictions.

---

## Models

Three regression models are trained and evaluated with **10-fold cross-validation**:

| Model | Notes |
|---|---|
| **Linear Regression** | Baseline model; assumes linear feature relationships |
| **Decision Tree Regressor** | Prone to overfitting; diagnosed via CV vs train RMSE gap |
| **Random Forest Regressor** | Ensemble method; best performer after tuning |

**Hyperparameter Tuning** is applied to Random Forest using `GridSearchCV`:

```python
param_grid = [
    {'n_estimators': [100, 200], 'max_features': [6, 8, 10]},
    {'bootstrap': [False], 'n_estimators': [100], 'max_features': [6, 8]}
]
```

---

## Results

| Model | CV RMSE | Test RMSE | Test MAE |
|---|---|---|---|
| Linear Regression | ~1.0 | — | 0.4811 |
| Decision Tree | ~1.0 (overfits) | — | 0.4731 |
| Random Forest | — | — | — |
| **Random Forest (Tuned)** | **Best** | **Lowest** | **0.3196** |

- **Best model:** Random Forest after GridSearchCV tuning
- **Most important feature:** `MedInc` (median income) — by a significant margin
- **95% confidence interval** computed on test RMSE for generalization bounds

---

## Project Structure

```
california-housing-pipeline/
│
├── california_housing_pipeline.ipynb   # Main notebook (full pipeline)
├── README.md                           # Project documentation
└── requirements.txt                    # Python dependencies
```

---

## Requirements

```
numpy
pandas
matplotlib
scikit-learn
scipy
```

Install all dependencies:

```bash
pip install -r requirements.txt
```

Or install directly:

```bash
pip install numpy pandas matplotlib scikit-learn scipy
```

---

## How to Run

**Option 1 — Jupyter Notebook (local):**

```bash
git clone https://github.com/abhinab44/california-housing-pipeline.git
cd california-housing-pipeline
jupyter notebook california_housing_pipeline.ipynb
```

**Option 2 — Google Colab:**

Click the badge below to open directly in Colab:

[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/)

> The dataset is fetched automatically via `sklearn.datasets.fetch_california_housing` — no manual download required.

---

## Key Findings

- **Random Forest (Tuned)** achieves the lowest test RMSE — confirming ensemble methods outperform linear and single-tree baselines on this dataset.
- **`MedInc` (Median Income)** is the dominant predictor of housing prices, consistent with real-world domain knowledge.
- **Engineered features** (`bedrooms_per_room`, `rooms_per_hh`) contribute meaningfully, validating the custom transformer design.
- **Decision Tree overfits severely** — training RMSE approaches zero while CV RMSE is high, demonstrating why cross-validation is essential over single train/test splits.
- **Residual analysis** reveals systematic underestimation at the $500K cap in the original dataset — an expected artifact of target value clipping.
- The **95% confidence interval** on test RMSE quantifies the expected range of generalization error.

---

## Concepts Demonstrated

- Stratified sampling for representative train/test splits
- Custom `sklearn` transformers using `BaseEstimator` & `TransformerMixin`
- `Pipeline` and `ColumnTransformer` for clean, leak-free preprocessing
- Cross-validation for honest model comparison
- `GridSearchCV` for systematic hyperparameter search
- Feature importance extraction from tree-based ensembles
- Statistical confidence intervals on evaluation metrics

---

## License

This project is open-source under the [MIT License](LICENSE).

---

*Built with Python 3.10 · scikit-learn · pandas · NumPy · Matplotlib*
