import pandas as pd

from src.models.baselines import (
    add_last_value_forecast,
    add_seasonal_naive_forecast,
)


def make_series() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "date": pd.date_range(
                "2025-01-01",
                periods=10,
                freq="D",
            ),
            "store_id": [1] * 10,
            "product_id": [101] * 10,
            "sales": list(range(1, 11)),
        }
    )


def test_last_value_forecast():
    df = make_series()

    result = add_last_value_forecast(df)

    assert pd.isna(
        result.iloc[0]["last_value_forecast"]
    )

    assert (
        result.iloc[1]["last_value_forecast"]
        == 1
    )

    assert (
        result.iloc[9]["last_value_forecast"]
        == 9
    )


def test_seasonal_naive_forecast():
    df = make_series()

    result = add_seasonal_naive_forecast(
        df,
        season_length=7,
    )

    assert pd.isna(
        result.iloc[6]["seasonal_naive_forecast"]
    )

    assert (
        result.iloc[7]["seasonal_naive_forecast"]
        == 1
    )

    assert (
        result.iloc[9]["seasonal_naive_forecast"]
        == 3
    )


def test_baseline_does_not_cross_series():
    dates = pd.date_range(
        "2025-01-01",
        periods=10,
        freq="D",
    )

    df = pd.DataFrame(
        {
            "date": list(dates) + list(dates),
            "store_id": [1] * 10 + [2] * 10,
            "product_id": [101] * 20,
            "sales": (
                list(range(1, 11))
                + list(range(101, 111))
            ),
        }
    )

    result = add_last_value_forecast(df)

    second_series = result[
        result["store_id"] == 2
    ].reset_index(drop=True)

    assert pd.isna(
        second_series.iloc[0]["last_value_forecast"]
    )

    assert (
        second_series.iloc[1]["last_value_forecast"]
        == 101
    )