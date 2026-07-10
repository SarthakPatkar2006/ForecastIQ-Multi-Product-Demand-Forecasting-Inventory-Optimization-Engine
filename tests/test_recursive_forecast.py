import numpy as np
import pandas as pd
import pytest

from src.forecasting.recursive import (
    recursive_forecast,
)


FEATURE_COLUMNS = [
    "store_id",
    "product_id",
    "day_of_week",
    "day_of_month",
    "week_of_year",
    "month",
    "quarter",
    "is_weekend",
    "lag_1",
    "lag_7",
    "lag_14",
    "lag_28",
    "rolling_mean_7",
    "rolling_mean_14",
    "rolling_mean_28",
    "rolling_std_7",
    "rolling_std_14",
]


class LagOneModel:
    """
    Deterministic fake model:
    prediction = lag_1 + 1
    """

    def predict(self, X):
        return (
            X["lag_1"]
            .to_numpy(dtype=float)
            + 1.0
        )


class NegativeModel:
    def predict(self, X):
        return np.array([-5.0])


def make_history(
    store_id=1,
    product_id=101,
):
    return pd.DataFrame(
        {
            "date": pd.date_range(
                "2025-01-01",
                periods=35,
                freq="D",
            ),
            "store_id": [store_id] * 35,
            "product_id": [product_id] * 35,
            "sales": list(range(1, 36)),
        }
    )


def test_recursive_forecast_returns_requested_horizon():
    history = make_history()

    result = recursive_forecast(
        model=LagOneModel(),
        history_df=history,
        feature_columns=FEATURE_COLUMNS,
        horizon=7,
    )

    assert len(result) == 7

    assert result["horizon_step"].tolist() == [
        1, 2, 3, 4, 5, 6, 7
    ]


def test_recursive_prediction_feeds_next_step():
    history = make_history()

    result = recursive_forecast(
        model=LagOneModel(),
        history_df=history,
        feature_columns=FEATURE_COLUMNS,
        horizon=3,
    )

    assert result["forecast"].tolist() == [
        36.0,
        37.0,
        38.0,
    ]


def test_forecast_dates_begin_after_history():
    history = make_history()

    result = recursive_forecast(
        model=LagOneModel(),
        history_df=history,
        feature_columns=FEATURE_COLUMNS,
        horizon=3,
    )

    expected_dates = pd.date_range(
        "2025-02-05",
        periods=3,
        freq="D",
    )

    assert result["date"].tolist() == list(
        expected_dates
    )


def test_series_are_forecast_independently():
    history = pd.concat(
        [
            make_history(
                store_id=1,
                product_id=101,
            ),
            make_history(
                store_id=2,
                product_id=202,
            ),
        ],
        ignore_index=True,
    )

    result = recursive_forecast(
        model=LagOneModel(),
        history_df=history,
        feature_columns=FEATURE_COLUMNS,
        horizon=2,
    )

    counts = (
        result.groupby(
            ["store_id", "product_id"]
        )
        .size()
        .tolist()
    )

    assert counts == [2, 2]


def test_negative_predictions_are_clipped_to_zero():
    history = make_history()

    result = recursive_forecast(
        model=NegativeModel(),
        history_df=history,
        feature_columns=FEATURE_COLUMNS,
        horizon=1,
    )

    assert result.iloc[0]["forecast"] == 0.0


def test_invalid_horizon_raises_error():
    history = make_history()

    with pytest.raises(ValueError):
        recursive_forecast(
            model=LagOneModel(),
            history_df=history,
            feature_columns=FEATURE_COLUMNS,
            horizon=0,
        )