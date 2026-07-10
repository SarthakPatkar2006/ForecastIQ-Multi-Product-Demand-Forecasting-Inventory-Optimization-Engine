from dataclasses import dataclass

import pandas as pd


@dataclass
class TemporalSplit:
    train: pd.DataFrame
    validation: pd.DataFrame
    test: pd.DataFrame


def temporal_train_val_test_split(
    df: pd.DataFrame,
    train_ratio: float = 0.70,
    validation_ratio: float = 0.15,
) -> TemporalSplit:
    """
    Split time-series data chronologically using unique dates.

    Oldest dates:
        training set

    Middle dates:
        validation set

    Newest dates:
        test set

    No random shuffling is performed.
    """
    if df.empty:
        raise ValueError("Cannot split an empty DataFrame.")

    if "date" not in df.columns:
        raise ValueError(
            "DataFrame must contain a 'date' column."
        )

    if not 0 < train_ratio < 1:
        raise ValueError(
            "train_ratio must be between 0 and 1."
        )

    if not 0 < validation_ratio < 1:
        raise ValueError(
            "validation_ratio must be between 0 and 1."
        )

    if train_ratio + validation_ratio >= 1:
        raise ValueError(
            "train_ratio + validation_ratio must be less than 1."
        )

    if df["date"].isna().any():
        raise ValueError(
            "Date column contains missing values."
        )

    unique_dates = (
        pd.Series(df["date"].unique())
        .sort_values()
        .reset_index(drop=True)
    )

    number_of_dates = len(unique_dates)

    if number_of_dates < 3:
        raise ValueError(
            "At least 3 unique dates are required."
        )

    train_end_index = int(
        number_of_dates * train_ratio
    )

    validation_end_index = int(
        number_of_dates
        * (train_ratio + validation_ratio)
    )

    # Ensure every split gets at least one date.
    train_end_index = max(
        1,
        min(train_end_index, number_of_dates - 2),
    )

    validation_end_index = max(
        train_end_index + 1,
        min(
            validation_end_index,
            number_of_dates - 1,
        ),
    )

    train_dates = unique_dates.iloc[
        :train_end_index
    ]

    validation_dates = unique_dates.iloc[
        train_end_index:validation_end_index
    ]

    test_dates = unique_dates.iloc[
        validation_end_index:
    ]

    train = df[
        df["date"].isin(train_dates)
    ].copy()

    validation = df[
        df["date"].isin(validation_dates)
    ].copy()

    test = df[
        df["date"].isin(test_dates)
    ].copy()

    return TemporalSplit(
        train=train.reset_index(drop=True),
        validation=validation.reset_index(drop=True),
        test=test.reset_index(drop=True),
    )