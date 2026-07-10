import numpy as np
import pandas as pd

from xgboost import XGBRegressor

from src.models.explain import (
    get_feature_importance,
)


def test_feature_importance_returns_all_features():
    X = pd.DataFrame(
        {
            "lag_1": [1, 2, 3, 4, 5, 6],
            "lag_7": [6, 5, 4, 3, 2, 1],
        }
    )

    y = np.array(
        [2, 4, 6, 8, 10, 12],
        dtype=float,
    )

    model = XGBRegressor(
        n_estimators=10,
        max_depth=2,
        random_state=42,
    )

    model.fit(X, y)

    result = get_feature_importance(
        model,
        ["lag_1", "lag_7"],
    )

    assert set(result["feature"]) == {
        "lag_1",
        "lag_7",
    }

    assert len(result) == 2


def test_feature_importance_is_sorted_descending():
    X = pd.DataFrame(
        {
            "lag_1": [1, 2, 3, 4, 5, 6],
            "lag_7": [6, 5, 4, 3, 2, 1],
        }
    )

    y = np.array(
        [2, 4, 6, 8, 10, 12],
        dtype=float,
    )

    model = XGBRegressor(
        n_estimators=10,
        max_depth=2,
        random_state=42,
    )

    model.fit(X, y)

    result = get_feature_importance(
        model,
        ["lag_1", "lag_7"],
    )

    assert result["importance"].is_monotonic_decreasing