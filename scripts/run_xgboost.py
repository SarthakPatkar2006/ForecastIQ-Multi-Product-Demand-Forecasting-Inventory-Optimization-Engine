import numpy as np

from src.data.loader import load_sales_data
from src.features.build_features import build_features
from src.models.evaluate import evaluate_predictions
from src.models.train import (
    FEATURE_COLUMNS,
    build_xgboost_model,
)
from src.splitting.temporal_split import (
    temporal_train_val_test_split,
)


DATA_PATH = "data/raw/train.csv"


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

    print("\nCreating temporal split...")

    split = temporal_train_val_test_split(
        featured_df,
        train_ratio=0.70,
        validation_ratio=0.15,
    )

    train_df = split.train
    validation_df = split.validation

    print(f"Train rows:      {len(train_df):,}")
    print(
        f"Validation rows: "
        f"{len(validation_df):,}"
    )
    print(f"Test rows:       {len(split.test):,}")

    X_train = train_df[
        FEATURE_COLUMNS
    ].copy()

    y_train = train_df[
        "sales"
    ].copy()

    X_validation = validation_df[
        FEATURE_COLUMNS
    ].copy()

    y_validation = validation_df[
        "sales"
    ].copy()

    print("\nBuilding XGBoost model...")

    model = build_xgboost_model()

    print("Training XGBoost with early stopping...")

    model.fit(
        X_train,
        y_train,
        eval_set=[
            (X_validation, y_validation),
        ],
        verbose=50,
    )

    print("\nPredicting validation set...")

    predictions = model.predict(
        X_validation
    )

    predictions = np.clip(
        predictions,
        a_min=0,
        a_max=None,
    )

    metrics = evaluate_predictions(
        y_validation,
        predictions,
    )

    print("\n=== XGBOOST ===")
    print(f"MAE:  {metrics['mae']:.4f}")
    print(f"RMSE: {metrics['rmse']:.4f}")
    print(f"WAPE: {metrics['wape']:.4f}")
    print(
        f"WAPE%: "
        f"{metrics['wape'] * 100:.2f}%"
    )

    print("\n=== TRAINING INFO ===")

    if hasattr(model, "best_iteration"):
        print(
            f"Best iteration: "
            f"{model.best_iteration}"
        )

    if hasattr(model, "best_score"):
        print(
            f"Best validation RMSE: "
            f"{model.best_score}"
        )

    print("\n=== CURRENT COMPARISON ===")
    print("Last Value WAPE:     19.63%")
    print("Seasonal Naive WAPE: 15.71%")
    print("Ridge WAPE:          12.38%")
    print("Random Forest WAPE:  11.30%")
    print(
        f"XGBoost WAPE:        "
        f"{metrics['wape'] * 100:.2f}%"
    )


if __name__ == "__main__":
    main()