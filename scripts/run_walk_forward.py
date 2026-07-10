import numpy as np

from xgboost import XGBRegressor

from src.data.loader import load_sales_data
from src.features.build_features import build_features
from src.models.evaluate import evaluate_predictions
from src.models.train import FEATURE_COLUMNS
from src.splitting.temporal_split import (
    temporal_train_val_test_split,
)
from src.validation.walk_forward import (
    generate_walk_forward_folds,
)


DATA_PATH = "data/raw/train.csv"

BEST_N_ESTIMATORS = 409


def build_fold_model() -> XGBRegressor:
    """
    Build fixed-capacity XGBoost model for temporal
    fold stability evaluation.

    409 estimators comes from the previously observed
    best validation iteration (~408).
    """
    return XGBRegressor(
        objective="reg:squarederror",
        n_estimators=BEST_N_ESTIMATORS,
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

    print("\nReserving final test partition...")

    outer_split = temporal_train_val_test_split(
        featured_df,
        train_ratio=0.70,
        validation_ratio=0.15,
    )

    development_df = (
        np.nan
    )

    # Train + validation become development history.
    # Final test remains untouched.
    development_df = (
        featured_df[
            featured_df["date"]
            <= outer_split.validation["date"].max()
        ]
        .copy()
        .reset_index(drop=True)
    )

    final_test = outer_split.test

    print(
        f"Development rows: "
        f"{len(development_df):,}"
    )

    print(
        f"Reserved test rows: "
        f"{len(final_test):,}"
    )

    print(
        "Reserved test range: "
        f"{final_test['date'].min().date()} "
        f"to {final_test['date'].max().date()}"
    )

    print("\nGenerating walk-forward folds...")

    folds = generate_walk_forward_folds(
        development_df,
        initial_train_ratio=0.55,
        validation_ratio=0.10,
        n_folds=3,
    )

    fold_results = []

    for fold in folds:
        print(
            f"\n{'=' * 50}"
        )

        print(
            f"FOLD {fold.fold_number}"
        )

        print(
            f"{'=' * 50}"
        )

        train_df = fold.train
        validation_df = fold.validation

        print(
            f"Train range: "
            f"{train_df['date'].min().date()} "
            f"to {train_df['date'].max().date()}"
        )

        print(
            f"Validation range: "
            f"{validation_df['date'].min().date()} "
            f"to {validation_df['date'].max().date()}"
        )

        print(
            f"Train rows: "
            f"{len(train_df):,}"
        )

        print(
            f"Validation rows: "
            f"{len(validation_df):,}"
        )

        X_train = train_df[
            FEATURE_COLUMNS
        ]

        y_train = train_df[
            "sales"
        ]

        X_validation = validation_df[
            FEATURE_COLUMNS
        ]

        y_validation = validation_df[
            "sales"
        ]

        model = build_fold_model()

        print("Training XGBoost...")

        model.fit(
            X_train,
            y_train,
            verbose=False,
        )

        print("Predicting...")

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

        fold_results.append(
            {
                "fold": fold.fold_number,
                **metrics,
            }
        )

        print(
            f"MAE:  {metrics['mae']:.4f}"
        )

        print(
            f"RMSE: {metrics['rmse']:.4f}"
        )

        print(
            f"WAPE: "
            f"{metrics['wape'] * 100:.2f}%"
        )

    wape_values = np.array(
        [
            result["wape"]
            for result in fold_results
        ]
    )

    mae_values = np.array(
        [
            result["mae"]
            for result in fold_results
        ]
    )

    rmse_values = np.array(
        [
            result["rmse"]
            for result in fold_results
        ]
    )

    print(
        f"\n{'=' * 50}"
    )

    print(
        "WALK-FORWARD SUMMARY"
    )

    print(
        f"{'=' * 50}"
    )

    for result in fold_results:
        print(
            f"Fold {result['fold']}: "
            f"MAE={result['mae']:.4f}, "
            f"RMSE={result['rmse']:.4f}, "
            f"WAPE={result['wape'] * 100:.2f}%"
        )

    print("\nAggregate stability:")

    print(
        f"Mean MAE:  "
        f"{mae_values.mean():.4f}"
    )

    print(
        f"Mean RMSE: "
        f"{rmse_values.mean():.4f}"
    )

    print(
        f"Mean WAPE: "
        f"{wape_values.mean() * 100:.2f}%"
    )

    print(
        f"WAPE Std:  "
        f"{wape_values.std(ddof=0) * 100:.2f}%"
    )

    print(
        f"Best WAPE: "
        f"{wape_values.min() * 100:.2f}%"
    )

    print(
        f"Worst WAPE: "
        f"{wape_values.max() * 100:.2f}%"
    )


if __name__ == "__main__":
    main()