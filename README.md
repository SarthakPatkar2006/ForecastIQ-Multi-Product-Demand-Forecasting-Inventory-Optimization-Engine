# ForecastIQ

> End-to-End Multi-Product Demand Forecasting & Inventory Optimization Engine using XGBoost, Time-Series Forecasting, Recursive Prediction, and FastAPI.

## Overview

ForecastIQ is an end-to-end machine learning system for demand forecasting and inventory optimization. It predicts future product demand across multiple stores using a leakage-safe time-series forecasting pipeline and converts forecasts into actionable inventory recommendations.

The project follows production-style machine learning practices including chronological data splitting, walk-forward validation, recursive forecasting, model persistence, API deployment, and automated testing.

---

# Features

- Leakage-safe time-series feature engineering
- Temporal train / validation / test split
- Walk-forward cross-validation
- Fold-local temporal early stopping
- Recursive multi-step forecasting
- XGBoost forecasting model
- Inventory optimization engine
- FastAPI REST API
- Model persistence
- Feature importance analysis
- Comprehensive unit testing
- Production-style project structure

---

# Dataset

**Kaggle Store Item Demand Forecasting Challenge**

Dataset Statistics:

- 913,000 historical observations
- 10 Stores
- 50 Products
- 500 Independent Demand Series
- Daily Sales
- Date Range:
  - 2013-01-01
  - 2017-12-31

---

# Project Architecture

```
Historical Sales Data
        │
        ▼
Feature Engineering
        │
        ▼
Temporal Train / Validation / Test Split
        │
        ▼
Walk-Forward Validation
        │
        ▼
Temporal Early Stopping
        │
        ▼
Final XGBoost Model
        │
        ▼
Recursive Multi-Step Forecasting
        │
        ▼
Inventory Optimization
        │
        ▼
FastAPI Inference API
```

---

# Project Structure

```
ForecastIQ/

├── api/
│   ├── inventory.py
│   ├── main.py
│   ├── predictor.py
│   ├── schemas.py
│   └── utils.py
│
├── artifacts/
│   └── forecastiq_xgboost.json
│
├── data/
│   └── raw/
│
├── outputs/
│   ├── feature_importance.csv
│   ├── forecast_7_day.csv
│   └── inventory_recommendations.csv
│
├── scripts/
│   ├── create_inventory_template.py
│   ├── generate_inventory_recommendations.py
│   ├── run_final_evaluation.py
│   ├── run_recursive_evaluation.py
│   ├── run_walk_forward.py
│   └── train_final_model.py
│
├── src/
│   ├── backtesting/
│   ├── data/
│   ├── features/
│   ├── forecasting/
│   ├── inventory/
│   ├── models/
│   └── splitting/
│
├── tests/
│
├── requirements.txt
├── README.md
└── pytest.ini
```

---

# Feature Engineering

ForecastIQ generates leakage-safe time-series features using only historical observations.

### Calendar Features

- Day of Week
- Day of Month
- Week of Year
- Month
- Quarter
- Weekend Indicator

### Lag Features

- Lag-1
- Lag-7
- Lag-14
- Lag-28

### Rolling Statistics

- Rolling Mean (7)
- Rolling Mean (14)
- Rolling Mean (28)
- Rolling Std (7)
- Rolling Std (14)

### Entity Features

- Store ID
- Product ID

---

# Models

## Baseline Models

- Last Value Forecast
- Seasonal Naive Forecast
- Ridge Regression
- Random Forest

## Final Model

- XGBoost Regressor

---

# Evaluation Strategy

ForecastIQ follows strict chronological evaluation.

Evaluation includes:

- Temporal Validation Split
- Walk-Forward Cross Validation
- Fold-local Temporal Early Stopping
- Final Untouched Holdout Test
- Genuine Recursive Multi-Step Evaluation

Metrics:

- MAE
- RMSE
- WAPE

---

# Final Results

| Evaluation | WAPE |
|------------|------|
| Validation | **10.84%** |
| Walk-Forward Mean | **10.73%** |
| Final Holdout Test | **10.05%** |
| Recursive 7-Day Evaluation | **9.98%** |

---

# Inventory Optimization

ForecastIQ transforms demand forecasts into inventory recommendations.

Outputs include:

- Expected Lead-Time Demand
- Safety Stock
- Reorder Point
- Reorder Status
- Recommended Order Quantity
- Stockout Risk Score
- Stockout Risk Level

---

# REST API

Start the API

```bash
uvicorn api.main:app --reload
```

Open Swagger UI

```
http://127.0.0.1:8000/docs
```

Available Endpoints

```
GET /

GET /health

POST /forecast

POST /inventory
```

---

# Example Forecast Response

```json
{
  "store_id": 1,
  "product_id": 1,
  "horizon": 7,
  "forecast": [
    13.09,
    15.10,
    15.11,
    15.89,
    16.95,
    18.93,
    20.03
  ]
}
```

---

# Example Inventory Response

```json
{
  "expected_lead_time_demand": 129,
  "safety_stock": 22.19,
  "reorder_point": 151.19,
  "reorder_status": "REORDER_NOW",
  "recommended_order_quantity": 51.19,
  "stockout_risk_score": 0.3386,
  "stockout_risk_level": "MEDIUM"
}
```

---

# Testing

Run all tests

```bash
pytest -v
```

Current Status

```
58 / 58 Tests Passed
```

Test Coverage

- Feature Engineering
- Temporal Splitting
- Walk-Forward Validation
- Recursive Forecasting
- Recursive Backtesting
- Inventory Optimization
- Evaluation Metrics
- Model Persistence

---

# Technologies Used

### Programming

- Python

### Machine Learning

- XGBoost
- Scikit-learn

### Data Processing

- Pandas
- NumPy

### API

- FastAPI

### Testing

- Pytest

---

# Future Improvements

- CSV Upload Forecasting
- Batch Forecasting
- Docker Support
- CI/CD Pipeline
- Cloud Deployment
- Interactive Dashboard

---

# License

This project is released under the MIT License.
