from __future__ import annotations

import numpy as np
import pandas as pd


SERIES_COLUMNS = [
    "store_id",
    "product_id",
]


def _build_future_feature_row(
    series_history: pd.DataFrame,
    forecast_date: pd.Timestamp,
) -> dict[str, float | int | pd.Timestamp]:
    """
    Build one future feature row using only information
    available before forecast_date.
    """
    history = (
        series_history
        .sort_values("date")
        .reset_index(drop=True)
    )

    if len(history) < 28:
        raise ValueError(
            "At least 28 historical observations are required "
            "for recursive forecasting."
        )

    sales_history = history["sales"].astype(float)

    iso_calendar = forecast_date.isocalendar()

    return {
        "date": forecast_date,
        "store_id": int(history["store_id"].iloc[-1]),
        "product_id": int(history["product_id"].iloc[-1]),
        "day_of_week": int(forecast_date.dayofweek),
        "day_of_month": int(forecast_date.day),
        "week_of_year": int(iso_calendar.week),
        "month": int(forecast_date.month),
        "quarter": int(forecast_date.quarter),
        "is_weekend": int(
            forecast_date.dayofweek >= 5
        ),
        "lag_1": float(
            sales_history.iloc[-1]
        ),
        "lag_7": float(
            sales_history.iloc[-7]
        ),
        "lag_14": float(
            sales_history.iloc[-14]
        ),
        "lag_28": float(
            sales_history.iloc[-28]
        ),
        "rolling_mean_7": float(
            sales_history.iloc[-7:].mean()
        ),
        "rolling_mean_14": float(
            sales_history.iloc[-14:].mean()
        ),
        "rolling_mean_28": float(
            sales_history.iloc[-28:].mean()
        ),
        "rolling_std_7": float(
            sales_history.iloc[-7:].std()
        ),
        "rolling_std_14": float(
            sales_history.iloc[-14:].std()
        ),
    }


def recursive_forecast(
    model,
    history_df: pd.DataFrame,
    feature_columns: list[str],
    horizon: int = 7,
) -> pd.DataFrame:
    """
    Generate recursive multi-step forecasts for every
    store-product demand series.

    Each predicted value is appended to history and may
    contribute to later lag and rolling features.
    """
    if horizon <= 0:
        raise ValueError(
            "horizon must be positive."
        )

    required_columns = {
        "date",
        "store_id",
        "product_id",
        "sales",
    }

    missing_columns = (
        required_columns - set(history_df.columns)
    )

    if missing_columns:
        raise ValueError(
            "Missing required columns: "
            + ", ".join(sorted(missing_columns))
        )

    if history_df.empty:
        raise ValueError(
            "history_df cannot be empty."
        )

    history = (
        history_df[
            [
                "date",
                "store_id",
                "product_id",
                "sales",
            ]
        ]
        .copy()
        .sort_values(
            SERIES_COLUMNS + ["date"],
            kind="stable",
        )
        .reset_index(drop=True)
    )

    forecast_frames: list[pd.DataFrame] = []

    grouped = history.groupby(
        SERIES_COLUMNS,
        sort=False,
    )

    for (
        store_id,
        product_id,
    ), series_df in grouped:
        series_history = (
            series_df.copy()
            .sort_values("date")
            .reset_index(drop=True)
        )

        if len(series_history) < 28:
            raise ValueError(
                f"Series ({store_id}, {product_id}) "
                "has fewer than 28 observations."
            )

        last_date = pd.Timestamp(
            series_history["date"].max()
        )

        series_forecasts = []

        for step in range(1, horizon + 1):
            forecast_date = (
                last_date
                + pd.Timedelta(days=step)
            )

            feature_row = _build_future_feature_row(
                series_history=series_history,
                forecast_date=forecast_date,
            )

            X_future = pd.DataFrame(
                [feature_row]
            )[feature_columns]

            prediction = float(
                model.predict(X_future)[0]
            )

            prediction = float(
                np.clip(
                    prediction,
                    a_min=0,
                    a_max=None,
                )
            )

            series_forecasts.append(
                {
                    "date": forecast_date,
                    "store_id": store_id,
                    "product_id": product_id,
                    "forecast": prediction,
                    "horizon_step": step,
                }
            )

            predicted_history_row = pd.DataFrame(
                [
                    {
                        "date": forecast_date,
                        "store_id": store_id,
                        "product_id": product_id,
                        "sales": prediction,
                    }
                ]
            )

            series_history = pd.concat(
                [
                    series_history,
                    predicted_history_row,
                ],
                ignore_index=True,
            )

        forecast_frames.append(
            pd.DataFrame(series_forecasts)
        )

    return (
        pd.concat(
            forecast_frames,
            ignore_index=True,
        )
        .sort_values(
            [
                "store_id",
                "product_id",
                "date",
            ]
        )
        .reset_index(drop=True)
    )