import numpy as np

from xgboost import XGBRegressor

from src.data.loader import load_sales_data
from src.features.build_features import build_features
from src.models.evaluate import evaluate_predictions
from src.models.train import FEATURE_COLUMNS
from src.splitting.temporal_split import (
    temporal_train_val_test_split,
)


DATA_PATH = "data/raw/train.csv"

FINAL_N_ESTIMATORS = 409


def build_final_model() -> XGBRegressor:
    """
    Build final XGBoost model.

    Hyperparameters are frozen before test evaluation.
    Tree count is based on the previously selected
    validation best iteration (~408).
    """
    return XGBRegressor(
        objective="reg:squarederror",
        n_estimators=FINAL_N_ESTIMATORS,
        learning_rate=0.05,
        max_depth=8,
        min_child_weight=5,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_alpha=0.1,
        reg_lambda=1.0,
        tree_method="hist",
        eval_metric="rmse",
        n_jobs=-1,
        random_state=42,
    )


def main() -> None:
    print("Loading dataset...")

    df = load_sales_data(DATA_PATH)

    print(f"Raw rows: {len(df):,}")

    print("\nBuilding leakage-safe features...")

    featured_df = build_features(
        df,
        drop_na=True,
    )

    print(
        f"Rows after feature generation: "
        f"{len(featured_df):,}"
    )

    print("\nCreating frozen temporal partitions...")

    split = temporal_train_val_test_split(
        featured_df,
        train_ratio=0.70,
        validation_ratio=0.15,
    )

    development_df = (
        featured_df[
            featured_df["date"]
            <= split.validation["date"].max()
        ]
        .copy()
        .reset_index(drop=True)
    )

    test_df = split.test.copy()

    print(
        f"Development rows: "
        f"{len(development_df):,}"
    )

    print(
        f"Test rows:        "
        f"{len(test_df):,}"
    )

    print(
        "\nDevelopment range: "
        f"{development_df['date'].min().date()} "
        f"to {development_df['date'].max().date()}"
    )

    print(
        "Test range:        "
        f"{test_df['date'].min().date()} "
        f"to {test_df['date'].max().date()}"
    )

    assert (
        development_df["date"].max()
        < test_df["date"].min()
    ), "Development and test periods overlap."

    X_development = development_df[
        FEATURE_COLUMNS
    ]

    y_development = development_df[
        "sales"
    ]

    X_test = test_df[
        FEATURE_COLUMNS
    ]

    y_test = test_df[
        "sales"
    ]

    print(
        "\nTraining final XGBoost model "
        "on complete development history..."
    )

    model = build_final_model()

    model.fit(
        X_development,
        y_development,
        verbose=False,
    )

    print("Running one-time test prediction...")

    predictions = model.predict(
        X_test
    )

    predictions = np.clip(
        predictions,
        a_min=0,
        a_max=None,
    )

    metrics = evaluate_predictions(
        y_test,
        predictions,
    )

    print("\n" + "=" * 50)
    print("FINAL UNTOUCHED TEST RESULTS")
    print("=" * 50)

    print(f"MAE:  {metrics['mae']:.4f}")
    print(f"RMSE: {metrics['rmse']:.4f}")
    print(f"WAPE: {metrics['wape']:.4f}")
    print(
        f"WAPE%: "
        f"{metrics['wape'] * 100:.2f}%"
    )

    print("\n" + "=" * 50)
    print("REFERENCE RESULTS")
    print("=" * 50)

    print("Original validation WAPE: 10.84%")
    print("Walk-forward mean WAPE:   10.72%")
    print(
        f"Final test WAPE:          "
        f"{metrics['wape'] * 100:.2f}%"
    )


if __name__ == "__main__":
    main()