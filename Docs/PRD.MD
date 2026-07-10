PRD — ForecastIQ
Multi-Product Demand Forecasting & Inventory Optimization Engine

Version: 1.0
Build target: 12 focused hours
Schedule: 6 days × 2 hours/day
Primary goal: Resume-worthy and interview-defensible AI/ML project
Project type: Time-series ML + decision optimization + lightweight deployment

1. Product Vision

ForecastIQ is an ML system that predicts future demand for multiple retail products and converts those forecasts into actionable inventory decisions.

The system answers:

“How much demand should we expect, what is the risk of stockout, and when/how much should we reorder?”

Core pipeline:

Historical Retail Sales
        ↓
Data Validation
        ↓
Leakage-Safe Feature Engineering
        ↓
Time-Aware Model Training
        ↓
Multi-Step Demand Forecast
        ↓
Forecast Evaluation
        ↓
Inventory Decision Engine
        ↓
┌─────────────────────────────┐
│ Safety Stock                │
│ Reorder Point               │
│ Stockout Risk               │
│ Recommended Order Quantity  │
└─────────────────────────────┘
        ↓
Dashboard + FastAPI
2. Problem Statement

Retail businesses face two opposing costs:

Too Little Inventory
        ↓
Stockouts
Lost Sales
Poor Service Level

versus:

Too Much Inventory
        ↓
Holding Costs
Dead Stock
Capital Lock-in

Static reorder rules fail when demand changes because of:

weekly seasonality
product-specific trends
promotions
demand volatility
calendar effects

ForecastIQ will use historical sales patterns to estimate future demand and produce inventory recommendations.

3. Target Users
Primary user

Inventory Manager

Needs to know:

expected demand
stockout risk
products requiring reorder
recommended reorder quantity
Secondary user

ML / Data Analyst

Needs to know:

model performance
baseline comparison
forecast errors
feature importance
validation results
4. Scope
P0 — Mandatory features

These determine whether the project is complete.

Multi-product sales ingestion
Data validation
Time-series feature engineering
Naive forecasting baselines
ML model comparison
Temporal train/validation/test split
Walk-forward validation
Demand forecasting
Inventory optimization
Stockout risk scoring
Model evaluation dashboard
P1 — Important if time permits
FastAPI prediction endpoint
Feature importance
Forecast intervals
Product-level error analysis
Model persistence
P2 — Explicitly out of scope

Do not build:

authentication
user registration
payment
MERN frontend
microservices
Kafka
Kubernetes
Docker
LSTM
Transformer forecasting
reinforcement learning
real-time streaming
multi-agent AI
LLM chatbot

These features would dilute the ML story and exceed the time budget.

5. Dataset Requirements

The dataset should contain:

Field	Required	Description
date	Yes	Observation date
product_id	Yes	Product identifier
sales	Yes	Units sold
store_id	Preferred	Store identifier
price	Optional	Product price
promotion	Optional	Promotion flag

Minimum acceptable structure:

date,product_id,sales
2025-01-01,P001,42
2025-01-01,P002,18
2025-01-02,P001,39
2025-01-02,P002,24
Dataset selection criterion

Prefer:

multiple products
chronological observations
sufficient historical depth
non-trivial seasonality
manageable local size

Do not spend more than 30 minutes selecting the dataset.

6. Functional Requirements
FR-1: Data Ingestion

System shall:

load historical sales data
parse timestamps
sort chronologically
validate required columns
detect missing values
detect duplicate observations

Input:

CSV dataset

Output:

Validated DataFrame
FR-2: Time-Series Feature Engineering

System shall generate calendar features:

day_of_week
day_of_month
week_of_year
month
quarter
is_weekend

Lag features:

lag_1
lag_7
lag_14
lag_28

Rolling features:

rolling_mean_7
rolling_mean_14
rolling_mean_28

rolling_std_7
rolling_std_14

Optional:

expanding_mean
trend_index
Critical leakage requirement

Rolling statistics must use historical data only.

Required pattern:

df.groupby("product_id")["sales"] \
  .shift(1) \
  .rolling(7) \
  .mean()

Current target values must never enter predictor features.

7. Forecasting Models
Model 0 — Naive Last Value
Tomorrow Forecast
=
Today's Demand

Purpose:

minimum baseline
Model 1 — Seasonal Naive
Today's Forecast
=
Demand 7 Days Ago

Purpose:

weekly-seasonality baseline
Model 2 — Ridge Regression

Purpose:

interpretable linear baseline
evaluate whether engineered features provide signal
Model 3 — Random Forest Regressor

Purpose:

nonlinear interactions
robust tree baseline
Model 4 — XGBoost or LightGBM

Purpose:

primary candidate model
nonlinear tabular forecasting
Final model selection

Select based on:

validation WAPE
validation MAE
stability across temporal folds

Do not select based on training score.

8. Data Splitting Strategy

Random splitting is prohibited.

Incorrect:

train_test_split(
    X,
    y,
    shuffle=True
)

Required:

Oldest 70%
    ↓
Training

Next 15%
    ↓
Validation

Newest 15%
    ↓
Test

Example:

Jan ───────── Aug | Sep ─ Oct | Nov ─ Dec
     TRAIN          VALID       TEST
9. Walk-Forward Validation

The system shall support expanding-window evaluation.

Fold 1

Train:
████████

Validation:
        ██
Fold 2

Train:
██████████

Validation:
          ██
Fold 3

Train:
████████████

Validation:
            ██

Purpose:

simulate real deployment
prevent future-data leakage
test temporal robustness
10. Evaluation Metrics
MAE

Primary interpretation:

Average absolute forecasting error.

RMSE

Purpose:

Penalizes large errors more strongly.

WAPE

Primary business metric:

WAPE
=
Sum Absolute Errors
/
Sum Actual Demand

Required model comparison:

Model	MAE	RMSE	WAPE
Last Value	Actual	Actual	Actual
Seasonal Naive	Actual	Actual	Actual
Ridge	Actual	Actual	Actual
Random Forest	Actual	Actual	Actual
Boosted Trees	Actual	Actual	Actual

All values must come from actual experiments.

11. Forecasting Requirements

User shall select:

Product ID
Forecast Horizon

Supported horizon for MVP:

7 days

Optional:

14 days
28 days

Output:

{
  "product_id": "P102",
  "forecast_horizon": 7,
  "predictions": [
    {
      "date": "2026-07-11",
      "predicted_demand": 28
    },
    {
      "date": "2026-07-12",
      "predicted_demand": 31
    }
  ]
}
12. Inventory Optimization Engine

This is a mandatory differentiator.

Inputs
forecasted demand
current inventory
lead time
demand variability
service level
12.1 Expected Lead-Time Demand
Expected Lead-Time Demand
=
Average Forecast Demand
×
Lead Time

Example:

Average Daily Forecast = 20
Lead Time = 5 days

Lead-Time Demand = 100
12.2 Safety Stock

MVP approach:

Safety Stock
=
Z-score
×
Demand Standard Deviation
×
sqrt(Lead Time)

Supported service levels:

Service level	Approx. Z
90%	1.28
95%	1.65
99%	2.33
12.3 Reorder Point
Reorder Point
=
Expected Lead-Time Demand
+
Safety Stock
12.4 Reorder Decision
if current_inventory <= reorder_point:
    REORDER_NOW

elif current_inventory <= reorder_point * 1.20:
    MONITOR

else:
    SUFFICIENT_STOCK
12.5 Recommended Order Quantity

MVP:

Recommended Quantity
=
Forecasted Horizon Demand
+
Safety Stock
-
Current Inventory

Clamp:

minimum = 0
13. Stockout Risk Engine

Input:

Current Inventory
Forecast Demand
Demand Uncertainty
Lead Time

Output:

0.00 → 1.00

Example:

{
  "stockout_risk": 0.81,
  "risk_level": "HIGH"
}

Risk categories:

Score	Level
0.00–0.30	LOW
0.31–0.60	MEDIUM
0.61–1.00	HIGH

Important technical requirement:

If the MVP uses a heuristic risk score rather than a calibrated probability model, the UI and README must call it a risk score, not a “probability of stockout.”

That distinction matters in interviews.

14. API Requirements

Framework:

FastAPI
Endpoint 1
GET /health

Response:

{
  "status": "healthy"
}
Endpoint 2
POST /forecast

Request:

{
  "product_id": "P102",
  "horizon": 7
}

Response:

{
  "product_id": "P102",
  "model": "xgboost",
  "forecast": []
}
Endpoint 3
POST /inventory/recommend

Request:

{
  "product_id": "P102",
  "current_inventory": 91,
  "lead_time_days": 5,
  "service_level": 0.95
}

Response:

{
  "reorder_point": 137,
  "safety_stock": 32,
  "risk_score": 0.81,
  "decision": "REORDER_NOW",
  "recommended_quantity": 125
}
Endpoint 4
GET /metrics

Response:

{
  "models": [
    {
      "name": "seasonal_naive",
      "mae": 0,
      "rmse": 0,
      "wape": 0
    }
  ]
}
15. Dashboard Requirements

Use:

Streamlit

Not React.

Screen 1 — Executive Overview

Display:

total products
high-risk products
average forecast error
products requiring reorder
Screen 2 — Product Forecast

User selects:

Product ID

Display:

Historical Demand
        +
Future Forecast
Screen 3 — Inventory Decision

Display:

Current Inventory
Forecast Demand
Safety Stock
Reorder Point
Risk Score
Recommended Quantity
Screen 4 — Model Performance

Display:

Model comparison table
MAE
RMSE
WAPE
16. Recommended Folder Structure
forecastiq/
│
├── data/
│   ├── raw/
│   └── processed/
│
├── notebooks/
│   └── 01_eda.ipynb
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
│   │   ├── train.py
│   │   ├── evaluate.py
│   │   └── predict.py
│   │
│   ├── validation/
│   │   └── walk_forward.py
│   │
│   └── inventory/
│       ├── safety_stock.py
│       ├── reorder.py
│       └── risk.py
│
├── api/
│   ├── main.py
│   └── schemas.py
│
├── dashboard/
│   └── app.py
│
├── models/
│   └── best_model.pkl
│
├── reports/
│   └── metrics.json
│
├── tests/
│   ├── test_features.py
│   └── test_inventory.py
│
├── requirements.txt
├── README.md
└── .gitignore
17. Non-Functional Requirements
Performance

MVP target:

Single-product forecast:
< 2 seconds
Reproducibility

Must define:

random_state = 42

where applicable.

Reliability

System must handle:

unknown product ID
missing inventory value
negative inventory
invalid horizon
insufficient history
Explainability

At minimum:

Feature Importance

Optional:

SHAP

SHAP is P1, not mandatory.