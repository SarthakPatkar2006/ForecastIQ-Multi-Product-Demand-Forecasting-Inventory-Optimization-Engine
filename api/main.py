from fastapi import FastAPI, HTTPException
import pandas as pd

from api.inventory import generate_inventory_recommendation
from api.predictor import predictor
from api.schemas import (
    ForecastRequest,
    ForecastResponse,
    InventoryRequest,
    InventoryResponse,
)
from api.utils import (
    validate_history,
)

from src.data.loader import load_sales_data
from src.forecasting.recursive import (
    recursive_forecast,
)

from src.models.train import (
    FEATURE_COLUMNS,
)


DATA_PATH = "data/raw/train.csv"


app = FastAPI(
    title="ForecastIQ API",
    version="1.0.0",
    description="Demand Forecasting & Inventory Optimization API",
)


@app.get("/")
def root():
    return {
        "project": "ForecastIQ",
        "status": "running",
    }


@app.get("/health")
def health():
    return {
        "status": "healthy",
    }


@app.post(
    "/forecast",
    response_model=ForecastResponse,
)
def forecast(
    request: ForecastRequest,
):
    history = load_sales_data(DATA_PATH)

    history = (
        history[
            (history["store_id"] == request.store_id)
            & (
                history["product_id"]
                == request.product_id
            )
        ]
        .sort_values("date")
        .tail(request.history_days)
        .copy()
    )

    if history.empty:
        raise HTTPException(
            status_code=404,
            detail="Series not found.",
        )

    try:
        validate_history(history)

        forecast_df = recursive_forecast(
            model=predictor.model,
            history_df=history,
            feature_columns=FEATURE_COLUMNS,
            horizon=request.horizon,
        )

        forecast = (
            forecast_df["forecast"]
            .tolist()
        )

    except Exception as exc:
        raise HTTPException(
            status_code=400,
            detail=str(exc),
        )

    return ForecastResponse(
        store_id=request.store_id,
        product_id=request.product_id,
        horizon=request.horizon,
        forecast=forecast,
    )


@app.post(
    "/inventory",
    response_model=InventoryResponse,
)
def inventory(
    request: InventoryRequest,
):
    try:
        return generate_inventory_recommendation(
            request
        )
    except Exception as exc:
        raise HTTPException(
            status_code=400,
            detail=str(exc),
        )