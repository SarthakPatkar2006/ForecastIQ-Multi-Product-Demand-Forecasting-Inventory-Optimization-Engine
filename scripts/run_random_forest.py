from src.data.loader import load_sales_data
from src.features.build_features import build_features
from src.models.evaluate import evaluate_predictions
from src.models.train import (
    build_random_forest_pipeline,
    split_features_target,
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

    X_train, y_train = split_features_target(
        train_df
    )

    X_validation, y_validation = (
        split_features_target(
            validation_df
        )
    )

    print("\nBuilding Random Forest pipeline...")

    pipeline = build_random_forest_pipeline(
        n_estimators=100,
        max_depth=18,
        min_samples_leaf=2,
    )

    print("Training Random Forest...")

    pipeline.fit(
        X_train,
        y_train,
    )

    print("Predicting validation set...")

    predictions = pipeline.predict(
        X_validation
    )

    predictions = predictions.clip(min=0)

    metrics = evaluate_predictions(
        y_validation,
        predictions,
    )

    print("\n=== RANDOM FOREST ===")
    print(f"MAE:  {metrics['mae']:.4f}")
    print(f"RMSE: {metrics['rmse']:.4f}")
    print(f"WAPE: {metrics['wape']:.4f}")
    print(
        f"WAPE%: "
        f"{metrics['wape'] * 100:.2f}%"
    )

    print("\n=== CURRENT COMPARISON ===")
    print("Seasonal Naive WAPE: 15.71%")
    print("Ridge WAPE:          12.38%")
    print(
        f"Random Forest WAPE:  "
        f"{metrics['wape'] * 100:.2f}%"
    )


if __name__ == "__main__":
    main()