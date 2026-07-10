# ForecastIQ

**Multi-Product Demand Forecasting & Inventory Optimization Engine**

ForecastIQ is an end-to-end machine learning project for forecasting retail demand across multiple store-product time series and converting forecasts into inventory decisions.

The project focuses on leakage-safe time-series feature engineering, chronological validation, forecasting baseline comparison, and scalable tabular ML models.

> **Current status:** Forecasting pipeline under active development. Data ingestion, validation, leakage-safe feature engineering, temporal splitting, baseline models, Ridge Regression, and Random Forest experiments are complete.

---

## Problem Statement

Retail inventory planning involves two competing risks:

* **Understocking** → stockouts, lost sales, poor service levels
* **Overstocking** → holding costs, dead stock, capital lock-in

Static reorder rules often fail when demand changes due to:

* weekly seasonality
* product-specific demand patterns
* store-level variation
* calendar effects
* demand volatility

ForecastIQ aims to estimate future demand and use those forecasts to support inventory decisions such as:

* safety stock
* reorder point
* stockout risk score
* recommended order quantity

---

## Current System Pipeline

```text
Historical Retail Sales
        ↓
Data Ingestion
        ↓
Schema Normalization
        ↓
Data Validation
        ↓
Leakage-Safe Feature Engineering
        ↓
Chronological 70/15/15 Split
        ↓
Forecasting Baselines
        ↓
Machine Learning Models
        ↓
MAE / RMSE / WAPE Evaluation
        ↓
Best Model Selection
        ↓
Multi-Step Demand Forecasting
        ↓
Inventory Decision Engine
```

---

## Dataset

ForecastIQ currently uses the **Store Item Demand Forecasting** dataset.

### Raw schema

| Column  | Description        |
| ------- | ------------------ |
| `date`  | Observation date   |
| `store` | Store identifier   |
| `item`  | Product identifier |
| `sales` | Units sold         |

### Normalized schema

| Raw Column | ForecastIQ Column |
| ---------- | ----------------- |
| `date`     | `date`            |
| `store`    | `store_id`        |
| `item`     | `product_id`      |
| `sales`    | `sales`           |

### Dataset characteristics

* **913,000 observations**
* **10 stores**
* **50 products**
* **500 store-product demand series**
* Daily observations
* Date range: **2013-01-01 to 2017-12-31**

Each `(store_id, product_id)` pair is treated as an independent demand series during lag and rolling-feature construction.

---

## Leakage-Safe Feature Engineering

ForecastIQ generates time-series predictors independently for every store-product series.

### Calendar features

* `day_of_week`
* `day_of_month`
* `week_of_year`
* `month`
* `quarter`
* `is_weekend`

### Lag features

* `lag_1`
* `lag_7`
* `lag_14`
* `lag_28`

### Rolling mean features

* `rolling_mean_7`
* `rolling_mean_14`
* `rolling_mean_28`

### Rolling standard deviation features

* `rolling_std_7`
* `rolling_std_14`

### Leakage prevention

Rolling statistics are shifted before aggregation:

```python
series.shift(1).rolling(window=7).mean()
```

This ensures the current target value never contributes to its own predictor features.

Feature engineering is also grouped by:

```text
(store_id, product_id)
```

preventing historical values from crossing demand-series boundaries.

---

## Time-Aware Data Splitting

Random train-test splitting is intentionally prohibited.

ForecastIQ uses chronological date boundaries:

```text
Oldest 70% of dates
        ↓
Training

Next 15% of dates
        ↓
Validation

Newest 15% of dates
        ↓
Test
```

The split operates on **unique dates**, ensuring all observations from the same date remain in exactly one partition.

This prevents:

* future-data leakage
* date overlap
* unrealistic random validation
* partial splitting of panel observations from the same date

---

## Forecasting Models

### 1. Last Value Baseline

Predicts current demand using the previous observation:

```text
Forecast(t) = Demand(t - 1)
```

### 2. Seasonal Naive Baseline

Uses demand from seven days earlier:

```text
Forecast(t) = Demand(t - 7)
```

### 3. Ridge Regression

Linear ML baseline using:

* one-hot encoded store identifiers
* one-hot encoded product identifiers
* standardized numeric features
* calendar features
* lag features
* rolling statistics

### 4. Random Forest Regressor

Nonlinear tree ensemble designed to capture interactions among:

* recent demand
* weekly seasonality
* longer lag structure
* rolling demand levels
* demand volatility
* store effects
* product effects

---

## Experimental Results

All reported metrics below are generated from actual validation experiments.

| Model             |        MAE |       RMSE |       WAPE |
| ----------------- | ---------: | ---------: | ---------: |
| Last Value        |    10.7357 |    14.7017 |     19.63% |
| Seasonal Naive    |     8.5910 |    11.2863 |     15.71% |
| Ridge Regression  |     6.7119 |     8.7766 |     12.38% |
| **Random Forest** | **6.1260** | **7.9869** | **11.30%** |

### Current best model

**Random Forest**

Current validation WAPE:

```text
11.30%
```

### Observations

The Seasonal Naive model substantially outperforms the Last Value baseline, indicating meaningful weekly demand structure.

Ridge Regression improves WAPE from:

```text
15.71% → 12.38%
```

This demonstrates that engineered calendar, lag, rolling, store, and product features contain predictive signal.

Random Forest further improves WAPE to:

```text
11.30%
```

indicating that nonlinear interactions provide additional forecasting value.

---

## Evaluation Metrics

### Mean Absolute Error — MAE

Measures average absolute forecast error.

```text
MAE = mean(|Actual - Predicted|)
```

### Root Mean Squared Error — RMSE

Penalizes larger forecasting errors more strongly.

```text
RMSE = sqrt(mean((Actual - Predicted)²))
```

### Weighted Absolute Percentage Error — WAPE

Primary business-oriented forecasting metric.

```text
WAPE = Σ|Actual - Predicted| / Σ|Actual|
```

Lower values indicate better forecasting performance.

---

## Testing

ForecastIQ includes automated tests for critical time-series invariants.

Current test coverage verifies:

* lag features use historical values only
* rolling means exclude the current target
* lag features do not cross store-product boundaries
* temporal splits contain no date overlap
* chronological ordering is preserved
* expected split ratios are maintained
* observations from the same date stay together
* invalid split ratios are rejected
* baseline forecasts use correct historical values
* baseline forecasts do not cross series boundaries
* MAE calculation
* RMSE calculation
* WAPE calculation
* zero-demand WAPE edge cases

Run all tests:

```bash
pytest -v
```

Current status:

```text
16 passed
```

---

## Project Structure

```text
ForecastIQ/
│
├── data/
│   ├── raw/
│   └── processed/
│
├── scripts/
│   ├── run_baselines.py
│   ├── run_ridge.py
│   └── run_random_forest.py
│
├── src/
│   ├── data/
│   │   ├── loader.py
│   │   └── validator.py
│   │
│   ├── features/
│   │   └── build_features.py
│   │
│   ├── models/
│   │   ├── baselines.py
│   │   ├── evaluate.py
│   │   └── train.py
│   │
│   └── splitting/
│       └── temporal_split.py
│
├── tests/
│   ├── test_baselines.py
│   ├── test_evaluate.py
│   ├── test_features.py
│   └── test_temporal_split.py
│
├── .gitignore
├── pytest.ini
├── requirements.txt
└── README.md
```

---

## Installation

### 1. Clone the repository

```bash
git clone <your-repository-url>
cd ForecastIQ
```

### 2. Create a virtual environment

```bash
python -m venv venv
```

### 3. Activate the environment

#### Windows PowerShell

```powershell
.\venv\Scripts\Activate.ps1
```

#### Linux / macOS

```bash
source venv/bin/activate
```

### 4. Install dependencies

```bash
pip install -r requirements.txt
```

---

## Running Experiments

### Run baseline experiments

```bash
python -m scripts.run_baselines
```

### Run Ridge Regression

```bash
python -m scripts.run_ridge
```

### Run Random Forest

```bash
python -m scripts.run_random_forest
```

### Run tests

```bash
pytest -v
```

---

## Roadmap

### Completed

* [x] Multi-product sales ingestion
* [x] Schema normalization
* [x] Data validation
* [x] Missing-value detection
* [x] Duplicate-observation detection
* [x] Leakage-safe calendar features
* [x] Lag features
* [x] Rolling mean features
* [x] Rolling volatility features
* [x] Chronological 70/15/15 split
* [x] Last Value baseline
* [x] Seasonal Naive baseline
* [x] MAE evaluation
* [x] RMSE evaluation
* [x] WAPE evaluation
* [x] Ridge Regression
* [x] Random Forest
* [x] Automated tests for core temporal invariants

### Next

* [ ] Boosted-tree forecasting model
* [ ] Walk-forward validation
* [ ] Final model selection
* [ ] Test-set evaluation
* [ ] Multi-step recursive forecasting
* [ ] Feature importance
* [ ] Model persistence
* [ ] Safety stock calculation
* [ ] Reorder point calculation
* [ ] Stockout risk scoring
* [ ] Recommended order quantity
* [ ] FastAPI inference endpoints
* [ ] Streamlit dashboard

---

## Planned Inventory Decision Engine

ForecastIQ will convert future demand forecasts into operational inventory recommendations.

### Expected lead-time demand

```text
Expected Lead-Time Demand
=
Average Forecast Demand × Lead Time
```

### Safety stock

```text
Safety Stock
=
Z-score × Demand Standard Deviation × sqrt(Lead Time)
```

### Reorder point

```text
Reorder Point
=
Expected Lead-Time Demand + Safety Stock
```

### Reorder decision

```text
current_inventory <= reorder_point
    → REORDER_NOW

current_inventory <= reorder_point × 1.20
    → MONITOR

otherwise
    → SUFFICIENT_STOCK
```

### Recommended quantity

```text
Recommended Quantity
=
Forecast Horizon Demand
+
Safety Stock
-
Current Inventory
```

The final quantity will be clamped to a minimum of zero.

---

## Planned Stockout Risk Engine

The MVP will expose a bounded **risk score**:

```text
0.00 → 1.00
```

Risk categories:

| Score     | Level  |
| --------- | ------ |
| 0.00–0.30 | LOW    |
| 0.31–0.60 | MEDIUM |
| 0.61–1.00 | HIGH   |

The score will be documented explicitly as a **heuristic risk score**, not a calibrated probability of stockout.

---

## Tech Stack

* Python 3.12
* pandas
* NumPy
* scikit-learn
* pytest

Planned additions:

* boosted-tree library
* FastAPI
* Streamlit
* model persistence tooling

---

## Engineering Principles

ForecastIQ is built around the following constraints:

* no random splitting for time-series evaluation
* no current-target leakage into rolling features
* no cross-series lag contamination
* validation-driven model comparison
* test-set isolation until final evaluation
* reproducible model configuration
* business-oriented forecast evaluation
* explicit distinction between heuristic risk scores and calibrated probabilities

---

## License

This project is intended for educational, portfolio, and research purposes.
