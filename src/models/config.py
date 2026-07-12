from __future__ import annotations

from xgboost import XGBRegressor


# Development-only tree count selected from clean walk-forward
# Fold best iterations:
#   Fold 1 = 415
#   Fold 2 = 266
#   Fold 3 = 449
#
# We intentionally freeze the median value to avoid
# selecting a tree count using future information.
FINAL_N_ESTIMATORS = 415


XGBOOST_PARAMS = {
    "objective": "reg:squarederror",
    "n_estimators": FINAL_N_ESTIMATORS,
    "learning_rate": 0.05,
    "max_depth": 8,
    "min_child_weight": 5,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "reg_alpha": 0.1,
    "reg_lambda": 1.0,
    "tree_method": "hist",
    "eval_metric": "rmse",
    "random_state": 42,
    "n_jobs": -1,
}


def build_xgboost_model(**kwargs) -> XGBRegressor:
    """
    Create a standardized XGBoost model used
    throughout the project.
    """

    params = XGBOOST_PARAMS.copy()
    params.update(kwargs)

    return XGBRegressor(**params)