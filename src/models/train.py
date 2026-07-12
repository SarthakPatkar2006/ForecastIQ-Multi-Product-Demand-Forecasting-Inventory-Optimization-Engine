import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.ensemble import RandomForestRegressor
from src.models.config import build_xgboost_model as _build_xgboost_model
TARGET_COLUMN = "sales"

CATEGORICAL_FEATURES = [
    "store_id",
    "product_id",
]

NUMERIC_FEATURES = [
    "day_of_week",
    "day_of_month",
    "week_of_year",
    "month",
    "quarter",
    "is_weekend",
    "lag_1",
    "lag_7",
    "lag_14",
    "lag_28",
    "rolling_mean_7",
    "rolling_mean_14",
    "rolling_mean_28",
    "rolling_std_7",
    "rolling_std_14",
]

FEATURE_COLUMNS = (
    CATEGORICAL_FEATURES
    + NUMERIC_FEATURES
)


def build_ridge_pipeline(
    alpha: float = 1.0,
) -> Pipeline:
    """
    Build a leakage-safe Ridge regression pipeline.

    Preprocessing is fitted only when pipeline.fit()
    is called on the training set.
    """
    preprocessor = ColumnTransformer(
        transformers=[
            (
                "categorical",
                OneHotEncoder(
                    handle_unknown="ignore",
                ),
                CATEGORICAL_FEATURES,
            ),
            (
                "numeric",
                StandardScaler(),
                NUMERIC_FEATURES,
            ),
        ]
    )

    model = Ridge(
        alpha=alpha,
    )

    return Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("model", model),
        ]
    )


def split_features_target(
    df: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.Series]:
    missing_columns = (
        set(FEATURE_COLUMNS + [TARGET_COLUMN])
        - set(df.columns)
    )

    if missing_columns:
        raise ValueError(
            "Missing model columns: "
            + ", ".join(sorted(missing_columns))
        )

    X = df[FEATURE_COLUMNS].copy()
    y = df[TARGET_COLUMN].copy()

    return X, y
def build_random_forest_pipeline(
    n_estimators: int = 100,
    max_depth: int | None = 18,
    min_samples_leaf: int = 2,
) -> Pipeline:
    """
    Build a Random Forest forecasting pipeline.

    Uses one-hot encoding for categorical identifiers.
    Scaling is unnecessary for tree-based models.
    """
    preprocessor = ColumnTransformer(
        transformers=[
            (
                "categorical",
                OneHotEncoder(
                    handle_unknown="ignore",
                ),
                CATEGORICAL_FEATURES,
            ),
            (
                "numeric",
                "passthrough",
                NUMERIC_FEATURES,
            ),
        ]
    )

    model = RandomForestRegressor(
        n_estimators=n_estimators,
        max_depth=max_depth,
        min_samples_leaf=min_samples_leaf,
        max_features="sqrt",
        n_jobs=-1,
        random_state=42,
    )

    return Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("model", model),
        ]
    )

def build_xgboost_model():
    """
    Backward-compatible wrapper around the centralized
    XGBoost configuration.
    """
    return _build_xgboost_model(
        n_estimators=1500,
        early_stopping_rounds=75,
    )
    """
    Build the primary boosted-tree forecasting candidate.

    Configuration is intentionally controlled for:
    - ~629k training rows
    - CPU training
    - temporal validation
    - early stopping
    """
    return XGBRegressor(
        objective="reg:squarederror",
        n_estimators=1500,
        learning_rate=0.05,
        max_depth=8,
        min_child_weight=5,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_alpha=0.1,
        reg_lambda=1.0,
        tree_method="hist",
        eval_metric="rmse",
        early_stopping_rounds=75,
        n_jobs=-1,
        random_state=42,
    )