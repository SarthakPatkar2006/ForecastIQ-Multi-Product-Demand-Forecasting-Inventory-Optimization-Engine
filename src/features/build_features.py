import numpy as np
import pandas as pd


SERIES_COLUMNS = [
    "store_id",
    "product_id",
]

LAG_WINDOWS = [
    1,
    7,
    14,
    28,
]

ROLLING_MEAN_WINDOWS = [
    7,
    14,
    28,
]

ROLLING_STD_WINDOWS = [
    7,
    14,
]


def add_calendar_features(
    df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Add deterministic calendar-based features.
    """
    result = df.copy()

    iso_calendar = result["date"].dt.isocalendar()

    result["day_of_week"] = (
        result["date"].dt.dayofweek
    )

    result["day_of_month"] = (
        result["date"].dt.day
    )

    result["week_of_year"] = (
        iso_calendar.week.astype("int16")
    )

    result["month"] = (
        result["date"].dt.month
    )

    result["quarter"] = (
        result["date"].dt.quarter
    )

    result["is_weekend"] = (
        result["day_of_week"] >= 5
    ).astype("int8")

    return result


def add_lag_features(
    df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Add historical demand lag features independently
    for each store-product demand series.
    """
    result = df.copy()

    grouped_sales = result.groupby(
        SERIES_COLUMNS,
        sort=False,
    )["sales"]

    for lag in LAG_WINDOWS:
        result[f"lag_{lag}"] = (
            grouped_sales.shift(lag)
        )

    return result


def add_rolling_features(
    df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Add leakage-safe rolling statistics.

    shift(1) ensures today's sales never contribute
    to today's predictor features.
    """
    result = df.copy()

    for window in ROLLING_MEAN_WINDOWS:
        result[f"rolling_mean_{window}"] = (
            result.groupby(
                SERIES_COLUMNS,
                sort=False,
            )["sales"]
            .transform(
                lambda series: (
                    series
                    .shift(1)
                    .rolling(
                        window=window,
                        min_periods=window,
                    )
                    .mean()
                )
            )
        )

    for window in ROLLING_STD_WINDOWS:
        result[f"rolling_std_{window}"] = (
            result.groupby(
                SERIES_COLUMNS,
                sort=False,
            )["sales"]
            .transform(
                lambda series: (
                    series
                    .shift(1)
                    .rolling(
                        window=window,
                        min_periods=window,
                    )
                    .std()
                )
            )
        )

    return result


def build_features(
    df: pd.DataFrame,
    drop_na: bool = True,
) -> pd.DataFrame:
    """
    Build complete leakage-safe time-series features.
    """
    required_columns = {
        "date",
        "store_id",
        "product_id",
        "sales",
    }

    missing_columns = (
        required_columns - set(df.columns)
    )

    if missing_columns:
        raise ValueError(
            "Missing required columns: "
            + ", ".join(
                sorted(missing_columns)
            )
        )

    result = (
        df.copy()
        .sort_values(
            SERIES_COLUMNS + ["date"],
            kind="stable",
        )
        .reset_index(drop=True)
    )

    result = add_calendar_features(result)
    result = add_lag_features(result)
    result = add_rolling_features(result)

    result = result.replace(
        [np.inf, -np.inf],
        np.nan,
    )

    if drop_na:
        generated_history_features = [
            column
            for column in result.columns
            if (
                column.startswith("lag_")
                or column.startswith("rolling_")
            )
        ]

        result = result.dropna(
            subset=generated_history_features
        )

    return result.reset_index(drop=True)