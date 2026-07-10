import pandas as pd
import pytest

from src.validation.walk_forward import (
    generate_walk_forward_folds,
)


def make_panel_data() -> pd.DataFrame:
    dates = pd.date_range(
        "2025-01-01",
        periods=100,
        freq="D",
    )

    rows = []

    for date in dates:
        for store_id in [1, 2]:
            rows.append(
                {
                    "date": date,
                    "store_id": store_id,
                    "product_id": 101,
                    "sales": 10,
                }
            )

    return pd.DataFrame(rows)


def test_generates_expected_number_of_folds():
    df = make_panel_data()

    folds = generate_walk_forward_folds(
        df,
        initial_train_ratio=0.55,
        validation_ratio=0.10,
        n_folds=3,
    )

    assert len(folds) == 3


def test_training_window_expands():
    df = make_panel_data()

    folds = generate_walk_forward_folds(df)

    train_sizes = [
        fold.train["date"].nunique()
        for fold in folds
    ]

    assert train_sizes == [55, 65, 75]


def test_validation_windows_have_expected_size():
    df = make_panel_data()

    folds = generate_walk_forward_folds(df)

    validation_sizes = [
        fold.validation["date"].nunique()
        for fold in folds
    ]

    assert validation_sizes == [10, 10, 10]


def test_train_precedes_validation():
    df = make_panel_data()

    folds = generate_walk_forward_folds(df)

    for fold in folds:
        assert (
            fold.train["date"].max()
            < fold.validation["date"].min()
        )


def test_no_train_validation_overlap():
    df = make_panel_data()

    folds = generate_walk_forward_folds(df)

    for fold in folds:
        train_dates = set(
            fold.train["date"]
        )

        validation_dates = set(
            fold.validation["date"]
        )

        assert train_dates.isdisjoint(
            validation_dates
        )


def test_validation_moves_forward():
    df = make_panel_data()

    folds = generate_walk_forward_folds(df)

    validation_starts = [
        fold.validation["date"].min()
        for fold in folds
    ]

    assert validation_starts == sorted(
        validation_starts
    )

    assert len(set(validation_starts)) == 3


def test_invalid_configuration_raises_error():
    df = make_panel_data()

    with pytest.raises(ValueError):
        generate_walk_forward_folds(
            df,
            initial_train_ratio=0.80,
            validation_ratio=0.10,
            n_folds=3,
        )