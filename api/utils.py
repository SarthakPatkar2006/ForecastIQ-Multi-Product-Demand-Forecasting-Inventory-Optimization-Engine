import pandas as pd

from src.features.build_features import build_features
from src.models.train import FEATURE_COLUMNS


def prepare_features(
    history: pd.DataFrame,
) -> pd.DataFrame:
    """
    Convert raw historical sales into the
    latest feature row required by XGBoost.
    """

    featured = build_features(
        history,
        drop_na=True,
    )

    if featured.empty:
        raise ValueError(
            "Not enough history to build features."
        )

    latest = (
        featured
        .sort_values("date")
        .tail(1)
    )

    return latest[FEATURE_COLUMNS].copy()


def validate_history(
    history: pd.DataFrame,
    minimum_days: int = 28,
) -> None:
    """
    Basic validation for inference history.
    """

    required = {
        "date",
        "store_id",
        "product_id",
        "sales",
    }

    missing = required - set(history.columns)

    if missing:
        raise ValueError(
            "Missing columns: "
            + ", ".join(sorted(missing))
        )

    if len(history) < minimum_days:
        raise ValueError(
            f"Need at least {minimum_days} historical rows."
        )

    if history["sales"].isna().any():
        raise ValueError(
            "Sales column contains missing values."
        )

    history.sort_values(
        "date",
        inplace=True,
    )