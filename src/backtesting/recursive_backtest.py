from __future__ import annotations

import pandas as pd

from src.forecasting.recursive import recursive_forecast
from src.models.evaluate import evaluate_predictions


KEY_COLUMNS = [
    "date",
    "store_id",
    "product_id",
]


def run_recursive_backtest(
    model,
    history_df: pd.DataFrame,
    actual_df: pd.DataFrame,
    feature_columns: list[str],
    horizon: int = 7,
) -> tuple[pd.DataFrame, dict[str, float]]:
    """
    Evaluate genuine recursive multi-step forecasting.

    Forecasts are generated only from history_df.
    Later forecast steps consume earlier predictions,
    never actual future target values.
    """
    if horizon <= 0:
        raise ValueError(
            "horizon must be positive."
        )

    if history_df.empty:
        raise ValueError(
            "history_df cannot be empty."
        )

    if actual_df.empty:
        raise ValueError(
            "actual_df cannot be empty."
        )

    history_end = pd.Timestamp(
        history_df["date"].max()
    )

    expected_start = (
        history_end
        + pd.Timedelta(days=1)
    )

    expected_end = (
        history_end
        + pd.Timedelta(days=horizon)
    )

    actual_window = (
        actual_df[
            actual_df["date"].between(
                expected_start,
                expected_end,
            )
        ]
        [
            [
                "date",
                "store_id",
                "product_id",
                "sales",
            ]
        ]
        .copy()
    )

    if actual_window.empty:
        raise ValueError(
            "No actual observations found for "
            "the recursive evaluation window."
        )

    forecasts = recursive_forecast(
        model=model,
        history_df=history_df,
        feature_columns=feature_columns,
        horizon=horizon,
    )

    evaluated = forecasts.merge(
        actual_window,
        on=KEY_COLUMNS,
        how="inner",
        validate="one_to_one",
    )

    if len(evaluated) != len(forecasts):
        raise ValueError(
            "Actual evaluation data does not fully cover "
            "all recursive forecasts."
        )

    evaluated = evaluated.rename(
        columns={
            "sales": "actual",
        }
    )

    metrics = evaluate_predictions(
        evaluated["actual"],
        evaluated["forecast"],
    )

    return evaluated, metrics