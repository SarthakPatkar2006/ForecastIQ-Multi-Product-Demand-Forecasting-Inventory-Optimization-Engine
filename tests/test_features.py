import pandas as pd
import pytest

from src.features.build_features import (
    build_features,
)


def test_lag_features_use_only_past_values():
    df = pd.DataFrame(
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

    featured = build_features(
        df,
        drop_na=False,
    )

    row = featured.iloc[28]

    assert row["sales"] == 29
    assert row["lag_1"] == 28
    assert row["lag_7"] == 22
    assert row["lag_14"] == 15
    assert row["lag_28"] == 1


def test_rolling_mean_excludes_current_target():
    df = pd.DataFrame(
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

    featured = build_features(
        df,
        drop_na=False,
    )

    row = featured.iloc[7]

    expected_mean = sum(
        range(1, 8)
    ) / 7

    assert row["sales"] == 8

    assert row["rolling_mean_7"] == pytest.approx(
        expected_mean
    )


def test_lags_do_not_cross_series_boundaries():
    dates = pd.date_range(
        "2025-01-01",
        periods=35,
        freq="D",
    )

    df = pd.DataFrame(
        {
            "date": list(dates) + list(dates),
            "store_id": [1] * 35 + [2] * 35,
            "product_id": [101] * 70,
            "sales": (
                list(range(1, 36))
                + list(range(101, 136))
            ),
        }
    )

    featured = build_features(
        df,
        drop_na=False,
    )

    second_series = featured[
        featured["store_id"] == 2
    ].reset_index(drop=True)

    assert pd.isna(
        second_series.iloc[0]["lag_1"]
    )

    assert second_series.iloc[1]["lag_1"] == 101