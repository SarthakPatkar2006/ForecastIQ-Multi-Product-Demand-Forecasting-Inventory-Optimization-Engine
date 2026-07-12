import pandas as pd


SERIES_COLUMNS = [
    "store_id",
    "product_id",
]


def add_last_value_forecast(
    df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Naive one-step baseline.

    Forecast for time t:
        demand observed at t - 1
    """
    result = (
        df.copy()
        .sort_values(
            SERIES_COLUMNS + ["date"],
            kind="stable",
        )
        .reset_index(drop=True)
    )

    result["last_value_forecast"] = (
        result.groupby(
            SERIES_COLUMNS,
            sort=False,
        )["sales"]
        .shift(1)
    )

    return result


def add_seasonal_naive_forecast(
    df: pd.DataFrame,
    season_length: int = 7,
) -> pd.DataFrame:
    """
    Seasonal naive baseline.

    Forecast for time t:
        demand observed at t - season_length
    """
    if season_length <= 0:
        raise ValueError(
            "season_length must be positive."
        )

    result = (
        df.copy()
        .sort_values(
            SERIES_COLUMNS + ["date"],
            kind="stable",
        )
        .reset_index(drop=True)
    )

    result["seasonal_naive_forecast"] = (
        result.groupby(
            SERIES_COLUMNS,
            sort=False,
        )["sales"]
        .shift(season_length)
    )

    return result