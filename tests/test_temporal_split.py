import pandas as pd
import pytest

from src.splitting.temporal_split import (
    temporal_train_val_test_split,
)


def make_panel_data() -> pd.DataFrame:
    dates = pd.date_range(
        "2025-01-01",
        periods=20,
        freq="D",
    )

    rows = []

    for date in dates:
        for store_id in [1, 2]:
            for product_id in [101, 102]:
                rows.append(
                    {
                        "date": date,
                        "store_id": store_id,
                        "product_id": product_id,
                        "sales": 10,
                    }
                )

    return pd.DataFrame(rows)


def test_temporal_split_has_no_date_overlap():
    df = make_panel_data()

    split = temporal_train_val_test_split(df)

    train_dates = set(split.train["date"])
    validation_dates = set(split.validation["date"])
    test_dates = set(split.test["date"])

    assert train_dates.isdisjoint(validation_dates)
    assert train_dates.isdisjoint(test_dates)
    assert validation_dates.isdisjoint(test_dates)


def test_temporal_split_preserves_chronology():
    df = make_panel_data()

    split = temporal_train_val_test_split(df)

    assert (
        split.train["date"].max()
        < split.validation["date"].min()
    )

    assert (
        split.validation["date"].max()
        < split.test["date"].min()
    )


def test_temporal_split_uses_expected_date_ratios():
    df = make_panel_data()

    split = temporal_train_val_test_split(df)

    assert split.train["date"].nunique() == 14
    assert split.validation["date"].nunique() == 3
    assert split.test["date"].nunique() == 3


def test_temporal_split_keeps_same_date_together():
    df = make_panel_data()

    split = temporal_train_val_test_split(df)

    original_rows_per_date = (
        df.groupby("date")
        .size()
    )

    for subset in [
        split.train,
        split.validation,
        split.test,
    ]:
        subset_rows_per_date = (
            subset.groupby("date")
            .size()
        )

        for date, count in subset_rows_per_date.items():
            assert count == original_rows_per_date.loc[date]


def test_invalid_ratios_raise_error():
    df = make_panel_data()

    with pytest.raises(ValueError):
        temporal_train_val_test_split(
            df,
            train_ratio=0.90,
            validation_ratio=0.20,
        )