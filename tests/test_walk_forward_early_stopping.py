import pandas as pd
import pytest

from scripts.run_walk_forward import (
    split_fold_training_for_early_stopping,
)


def make_fold_train() -> pd.DataFrame:
    dates = pd.date_range(
        "2025-01-01",
        periods=100,
        freq="D",
    )

    return pd.DataFrame(
        {
            "date": dates,
            "store_id": [1] * 100,
            "product_id": [101] * 100,
            "sales": list(range(100)),
        }
    )


def test_inner_split_preserves_chronology():
    fold_train = make_fold_train()

    fit_df, early_stop_df = (
        split_fold_training_for_early_stopping(
            fold_train,
            early_stop_ratio=0.15,
        )
    )

    assert (
        fit_df["date"].max()
        < early_stop_df["date"].min()
    )


def test_inner_split_has_no_date_overlap():
    fold_train = make_fold_train()

    fit_df, early_stop_df = (
        split_fold_training_for_early_stopping(
            fold_train,
            early_stop_ratio=0.15,
        )
    )

    fit_dates = set(
        fit_df["date"]
    )

    early_stop_dates = set(
        early_stop_df["date"]
    )

    assert fit_dates.isdisjoint(
        early_stop_dates
    )


def test_inner_split_keeps_same_date_together():
    base = make_fold_train()

    duplicated = pd.concat(
        [
            base,
            base.assign(
                store_id=2,
                product_id=202,
            ),
        ],
        ignore_index=True,
    )

    fit_df, early_stop_df = (
        split_fold_training_for_early_stopping(
            duplicated,
            early_stop_ratio=0.15,
        )
    )

    assert set(
        fit_df["date"]
    ).isdisjoint(
        set(early_stop_df["date"])
    )


def test_inner_split_uses_latest_dates_for_early_stop():
    fold_train = make_fold_train()

    _, early_stop_df = (
        split_fold_training_for_early_stopping(
            fold_train,
            early_stop_ratio=0.15,
        )
    )

    assert (
        early_stop_df["date"].max()
        == fold_train["date"].max()
    )


def test_invalid_inner_split_ratio_rejected():
    fold_train = make_fold_train()

    with pytest.raises(ValueError):
        split_fold_training_for_early_stopping(
            fold_train,
            early_stop_ratio=1.0,
        )