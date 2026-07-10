from dataclasses import dataclass

import pandas as pd


@dataclass
class WalkForwardFold:
    fold_number: int
    train: pd.DataFrame
    validation: pd.DataFrame


def generate_walk_forward_folds(
    df: pd.DataFrame,
    initial_train_ratio: float = 0.55,
    validation_ratio: float = 0.10,
    n_folds: int = 3,
) -> list[WalkForwardFold]:
    """
    Generate expanding-window temporal validation folds.

    Example:
        Fold 1: Train 55%, Validate next 10%
        Fold 2: Train 65%, Validate next 10%
        Fold 3: Train 75%, Validate next 10%

    Splitting is performed on unique chronological dates.
    """
    if df.empty:
        raise ValueError(
            "Cannot generate folds from an empty DataFrame."
        )

    if "date" not in df.columns:
        raise ValueError(
            "DataFrame must contain a 'date' column."
        )

    if df["date"].isna().any():
        raise ValueError(
            "Date column contains missing values."
        )

    if not 0 < initial_train_ratio < 1:
        raise ValueError(
            "initial_train_ratio must be between 0 and 1."
        )

    if not 0 < validation_ratio < 1:
        raise ValueError(
            "validation_ratio must be between 0 and 1."
        )

    if n_folds <= 0:
        raise ValueError(
            "n_folds must be positive."
        )

    total_required_ratio = (
        initial_train_ratio
        + n_folds * validation_ratio
    )

    if total_required_ratio > 1:
        raise ValueError(
            "Walk-forward configuration exceeds "
            "available date range."
        )

    unique_dates = (
        pd.Series(df["date"].unique())
        .sort_values()
        .reset_index(drop=True)
    )

    n_dates = len(unique_dates)

    if n_dates < n_folds + 1:
        raise ValueError(
            "Insufficient unique dates for requested folds."
        )

    initial_train_size = int(
        n_dates * initial_train_ratio
    )

    validation_size = int(
        n_dates * validation_ratio
    )

    if initial_train_size < 1:
        raise ValueError(
            "Initial training window is empty."
        )

    if validation_size < 1:
        raise ValueError(
            "Validation window is empty."
        )

    folds: list[WalkForwardFold] = []

    for fold_index in range(n_folds):
        train_end = (
            initial_train_size
            + fold_index * validation_size
        )

        validation_start = train_end

        validation_end = (
            validation_start
            + validation_size
        )

        train_dates = unique_dates.iloc[
            :train_end
        ]

        validation_dates = unique_dates.iloc[
            validation_start:validation_end
        ]

        if validation_dates.empty:
            raise ValueError(
                f"Fold {fold_index + 1} has "
                "an empty validation window."
            )

        train_df = df[
            df["date"].isin(train_dates)
        ].copy()

        validation_df = df[
            df["date"].isin(validation_dates)
        ].copy()

        folds.append(
            WalkForwardFold(
                fold_number=fold_index + 1,
                train=train_df.reset_index(drop=True),
                validation=validation_df.reset_index(
                    drop=True
                ),
            )
        )

    return folds