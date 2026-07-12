from __future__ import annotations

import pandas as pd

from xgboost import XGBRegressor


def get_feature_importance(
    model: XGBRegressor,
    feature_columns: list[str],
) -> pd.DataFrame:
    """
    Extract normalized XGBoost feature importances.

    Returns features sorted from highest to lowest importance.
    """
    importances = model.feature_importances_

    if len(importances) != len(feature_columns):
        raise ValueError(
            "Feature importance count does not match "
            "the number of feature columns."
        )

    result = pd.DataFrame(
        {
            "feature": feature_columns,
            "importance": importances,
        }
    )

    return (
        result
        .sort_values(
            "importance",
            ascending=False,
            kind="stable",
        )
        .reset_index(drop=True)
    )
    