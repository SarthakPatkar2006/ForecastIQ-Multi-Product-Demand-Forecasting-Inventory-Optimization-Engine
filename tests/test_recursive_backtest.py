import numpy as np
import pandas as pd
import pytest

from src.backtesting.recursive_backtest import (
    run_recursive_backtest,
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


class LagOnePlusOneModel:
    """
    prediction = lag_1 + 1

    This makes recursive feedback directly testable.
    """

    def predict(self, X):
        return (
            X["lag_1"]
            .to_numpy(dtype=float)
            + 1.0
        )


def make_history() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "date": pd.date_range(
                "2025-01-01",
                periods=35,
                freq="D",
            ),
            "store_id": [1] * 35,
            "product_id": [101] * 35,
            "sales": list(range(1, 36)),
        }
    )


def make_actuals() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "date": pd.date_range(
                "2025-02-05",
                periods=3,
                freq="D",
            ),
            "store_id": [1] * 3,
            "product_id": [101] * 3,
            "sales": [
                100.0,
                200.0,
                300.0,
            ],
        }
    )


def test_backtest_uses_recursive_predictions():
    history = make_history()
    actuals = make_actuals()

    evaluated, _ = run_recursive_backtest(
        model=LagOnePlusOneModel(),
        history_df=history,
        actual_df=actuals,
        feature_columns=FEATURE_COLUMNS,
        horizon=3,
    )

    assert evaluated["forecast"].tolist() == [
        36.0,
        37.0,
        38.0,
    ]


def test_actual_values_do_not_feed_later_predictions():
    history = make_history()
    actuals = make_actuals()

    evaluated, _ = run_recursive_backtest(
        model=LagOnePlusOneModel(),
        history_df=history,
        actual_df=actuals,
        feature_columns=FEATURE_COLUMNS,
        horizon=3,
    )

    # If actual day 1 = 100 leaked into day 2,
    # prediction would become 101 instead of 37.
    assert evaluated.iloc[1]["forecast"] == 37.0

    # If actual day 2 = 200 leaked into day 3,
    # prediction would become 201 instead of 38.
    assert evaluated.iloc[2]["forecast"] == 38.0


def test_backtest_returns_all_horizon_rows():
    history = make_history()
    actuals = make_actuals()

    evaluated, _ = run_recursive_backtest(
        model=LagOnePlusOneModel(),
        history_df=history,
        actual_df=actuals,
        feature_columns=FEATURE_COLUMNS,
        horizon=3,
    )

    assert len(evaluated) == 3


def test_backtest_returns_metrics():
    history = make_history()
    actuals = make_actuals()

    _, metrics = run_recursive_backtest(
        model=LagOnePlusOneModel(),
        history_df=history,
        actual_df=actuals,
        feature_columns=FEATURE_COLUMNS,
        horizon=3,
    )

    assert set(metrics) == {
        "mae",
        "rmse",
        "wape",
    }

    assert metrics["mae"] > 0
    assert metrics["rmse"] > 0
    assert metrics["wape"] > 0


def test_missing_actual_coverage_is_rejected():
    history = make_history()

    incomplete_actuals = pd.DataFrame(
        {
            "date": [
                pd.Timestamp("2025-02-05"),
            ],
            "store_id": [1],
            "product_id": [101],
            "sales": [100.0],
        }
    )

    with pytest.raises(ValueError):
        run_recursive_backtest(
            model=LagOnePlusOneModel(),
            history_df=history,
            actual_df=incomplete_actuals,
            feature_columns=FEATURE_COLUMNS,
            horizon=3,
        )


def test_invalid_horizon_is_rejected():
    with pytest.raises(ValueError):
        run_recursive_backtest(
            model=LagOnePlusOneModel(),
            history_df=make_history(),
            actual_df=make_actuals(),
            feature_columns=FEATURE_COLUMNS,
            horizon=0,
        )