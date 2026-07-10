from __future__ import annotations

import numpy as np

from src.data.loader import load_sales_data
from src.features.build_features import build_features
from src.models.evaluate import evaluate_predictions
from src.models.train import FEATURE_COLUMNS
from src.validation.walk_forward import generate_walk_forward_folds

from xgboost import XGBRegressor


DATA_PATH = "data/raw/train.csv"

N_FOLDS = 3


# Last 15% of each outer training history is used only
# for fold-local early stopping.
EARLY_STOP_RATIO = 0.15

MAX_N_ESTIMATORS = 2000
EARLY_STOPPING_ROUNDS = 75


def build_xgboost_model() -> XGBRegressor:
    """
    Build an unfitted XGBoost model.

    n_estimators is intentionally large because the
    effective tree count is selected independently
    inside each fold through temporal early stopping.
    """
    return XGBRegressor(
        objective="reg:squarederror",
        n_estimators=MAX_N_ESTIMATORS,
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


def split_fold_training_for_early_stopping(
    fold_train,
    early_stop_ratio: float = EARLY_STOP_RATIO,
):
    """
    Split an outer fold's training history chronologically.

    Earlier dates:
        model fitting

    Latest dates inside outer training history:
        fold-local early stopping

    The outer validation fold remains untouched.
    """
    if not 0 < early_stop_ratio < 1:
        raise ValueError(
            "early_stop_ratio must be between 0 and 1."
        )

    unique_dates = np.array(
        sorted(
            fold_train["date"].unique()
        )
    )

    if len(unique_dates) < 2:
        raise ValueError(
            "Fold training data must contain "
            "at least two unique dates."
        )

    split_index = int(
        len(unique_dates)
        * (1.0 - early_stop_ratio)
    )

    split_index = max(
        1,
        min(
            split_index,
            len(unique_dates) - 1,
        ),
    )

    fit_dates = unique_dates[
        :split_index
    ]

    early_stop_dates = unique_dates[
        split_index:
    ]

    fit_df = (
        fold_train[
            fold_train["date"].isin(
                fit_dates
            )
        ]
        .copy()
        .reset_index(drop=True)
    )

    early_stop_df = (
        fold_train[
            fold_train["date"].isin(
                early_stop_dates
            )
        ]
        .copy()
        .reset_index(drop=True)
    )

    if fit_df.empty:
        raise RuntimeError(
            "Fold fitting partition is empty."
        )

    if early_stop_df.empty:
        raise RuntimeError(
            "Fold early-stop partition is empty."
        )

    if (
        fit_df["date"].max()
        >= early_stop_df["date"].min()
    ):
        raise RuntimeError(
            "Fit and early-stop periods overlap "
            "or violate chronology."
        )

    return fit_df, early_stop_df


def main() -> None:
    print("Loading dataset...")

    df = load_sales_data(
        DATA_PATH
    )

    print(f"Raw rows: {len(df):,}")

    print(
        "\nBuilding leakage-safe features..."
    )

    featured_df = build_features(
        df,
        drop_na=True,
    )

    print(
        f"Rows after feature generation: "
        f"{len(featured_df):,}"
    )

    print(
        "\nReserving final test partition..."
    )

    unique_dates = np.array(
        sorted(
            featured_df["date"].unique()
        )
    )

    test_start_index = int(
        len(unique_dates) * 0.85
    )

    development_dates = unique_dates[
        :test_start_index
    ]

    test_dates = unique_dates[
        test_start_index:
    ]

    development_df = (
        featured_df[
            featured_df["date"].isin(
                development_dates
            )
        ]
        .copy()
        .reset_index(drop=True)
    )

    reserved_test_df = (
        featured_df[
            featured_df["date"].isin(
                test_dates
            )
        ]
        .copy()
        .reset_index(drop=True)
    )

    print(
        f"Development rows: "
        f"{len(development_df):,}"
    )

    print(
        f"Reserved test rows: "
        f"{len(reserved_test_df):,}"
    )

    print(
        f"Reserved test range: "
        f"{reserved_test_df['date'].min().date()} "
        f"to "
        f"{reserved_test_df['date'].max().date()}"
    )

    print(
        "\nGenerating walk-forward folds..."
    )

    folds = generate_walk_forward_folds(
        development_df,
        initial_train_ratio=0.55,
        validation_ratio=0.10,
        n_folds=N_FOLDS,
    )

    fold_results = []

    for fold in folds:
        fold_number = fold.fold_number
        fold_train = fold.train
        fold_validation = fold.validation

        

        print(
            f"Outer train range: "
            f"{fold_train['date'].min().date()} "
            f"to "
            f"{fold_train['date'].max().date()}"
        )

        print(
            f"Outer validation range: "
            f"{fold_validation['date'].min().date()} "
            f"to "
            f"{fold_validation['date'].max().date()}"
        )

        fit_df, early_stop_df = (
            split_fold_training_for_early_stopping(
                fold_train
            )
        )

        print(
            f"Fit range: "
            f"{fit_df['date'].min().date()} "
            f"to "
            f"{fit_df['date'].max().date()}"
        )

        print(
            f"Early-stop range: "
            f"{early_stop_df['date'].min().date()} "
            f"to "
            f"{early_stop_df['date'].max().date()}"
        )

        print(
            f"Fit rows: "
            f"{len(fit_df):,}"
        )

        print(
            f"Early-stop rows: "
            f"{len(early_stop_df):,}"
        )

        print(
            f"Outer validation rows: "
            f"{len(fold_validation):,}"
        )

        if (
            early_stop_df["date"].max()
            >= fold_validation["date"].min()
        ):
            raise RuntimeError(
                "Early-stop period overlaps "
                "outer validation."
            )

        X_fit = fit_df[
            FEATURE_COLUMNS
        ]

        y_fit = fit_df[
            "sales"
        ]

        X_early_stop = early_stop_df[
            FEATURE_COLUMNS
        ]

        y_early_stop = early_stop_df[
            "sales"
        ]

        X_validation = fold_validation[
            FEATURE_COLUMNS
        ]

        y_validation = fold_validation[
            "sales"
        ]

        print(
            "Training XGBoost with "
            "fold-local temporal early stopping..."
        )

        model = build_xgboost_model()

        model.set_params(
            early_stopping_rounds=(
                EARLY_STOPPING_ROUNDS
            )
        )

        model.fit(
            X_fit,
            y_fit,
            eval_set=[
                (
                    X_early_stop,
                    y_early_stop,
                )
            ],
            verbose=False,
        )

        print("Predicting outer validation...")

        predictions = model.predict(
            X_validation
        )

        metrics = evaluate_predictions(
            y_validation,
            predictions,
        )

        best_iteration = getattr(
            model,
            "best_iteration",
            None,
        )

        print(
            f"Best iteration: "
            f"{best_iteration}"
        )

        print(
            f"MAE:  "
            f"{metrics['mae']:.4f}"
        )

        print(
            f"RMSE: "
            f"{metrics['rmse']:.4f}"
        )

        print(
            f"WAPE: "
            f"{metrics['wape'] * 100:.2f}%"
        )

        fold_results.append(
            {
                "fold": fold_number,
                "mae": metrics["mae"],
                "rmse": metrics["rmse"],
                "wape": metrics["wape"],
                "best_iteration": (
                    best_iteration
                ),
            }
        )

    print("\n" + "=" * 58)
    print("CLEAN WALK-FORWARD SUMMARY")
    print("=" * 58)

    for result in fold_results:
        print(
            f"Fold {result['fold']}: "
            f"MAE={result['mae']:.4f}, "
            f"RMSE={result['rmse']:.4f}, "
            f"WAPE="
            f"{result['wape'] * 100:.2f}%, "
            f"BestIteration="
            f"{result['best_iteration']}"
        )

    wapes = np.array(
        [
            result["wape"]
            for result in fold_results
        ],
        dtype=float,
    )

    maes = np.array(
        [
            result["mae"]
            for result in fold_results
        ],
        dtype=float,
    )

    rmses = np.array(
        [
            result["rmse"]
            for result in fold_results
        ],
        dtype=float,
    )

    print("\nAggregate stability:")

    print(
        f"Mean MAE:  "
        f"{maes.mean():.4f}"
    )

    print(
        f"Mean RMSE: "
        f"{rmses.mean():.4f}"
    )

    print(
        f"Mean WAPE: "
        f"{wapes.mean() * 100:.2f}%"
    )

    print(
        f"WAPE Std:  "
        f"{wapes.std() * 100:.2f}%"
    )

    print(
        f"Best WAPE: "
        f"{wapes.min() * 100:.2f}%"
    )

    print(
        f"Worst WAPE: "
        f"{wapes.max() * 100:.2f}%"
    )

    print(
        "\nEach fold selected its tree count "
        "using only an internal temporal slice "
        "of that fold's training history."
    )


if __name__ == "__main__":
    main()