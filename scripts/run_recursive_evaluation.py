from __future__ import annotations

import pandas as pd
from xgboost import XGBRegressor

from src.backtesting.recursive_backtest import (
    run_recursive_backtest,
)
from src.data.loader import load_sales_data
from src.features.build_features import build_features
from src.models.evaluate import evaluate_predictions
from src.models.train import FEATURE_COLUMNS


DATA_PATH = "data/raw/train.csv"

HORIZON = 7

HISTORY_END_DATE = pd.Timestamp(
    "2017-04-05"
)

FINAL_N_ESTIMATORS = 409


def build_cutoff_model() -> XGBRegressor:
    """
    Build the same frozen XGBoost configuration used
    for the current forecasting system.

    Important:
    This model will be trained only on information
    available through HISTORY_END_DATE.
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
    print("Loading full labeled dataset...")

    df = load_sales_data(
        DATA_PATH
    )

    print(f"Rows: {len(df):,}")

    print(
        f"Date range: "
        f"{df['date'].min().date()} "
        f"to {df['date'].max().date()}"
    )

    print(
        "\nCreating strict historical cutoff..."
    )

    history_df = (
        df[
            df["date"] <= HISTORY_END_DATE
        ]
        .copy()
        .reset_index(drop=True)
    )

    future_actual_df = (
        df[
            df["date"] > HISTORY_END_DATE
        ]
        .copy()
        .reset_index(drop=True)
    )

    if history_df.empty:
        raise RuntimeError(
            "Historical data is empty."
        )

    if future_actual_df.empty:
        raise RuntimeError(
            "Future actual data is empty."
        )

    if (
        history_df["date"].max()
        >= future_actual_df["date"].min()
    ):
        raise RuntimeError(
            "Historical and future periods overlap."
        )

    print(
        f"History range: "
        f"{history_df['date'].min().date()} "
        f"to "
        f"{history_df['date'].max().date()}"
    )

    forecast_start = (
        HISTORY_END_DATE
        + pd.Timedelta(days=1)
    )

    forecast_end = (
        HISTORY_END_DATE
        + pd.Timedelta(days=HORIZON)
    )

    print(
        f"Recursive evaluation window: "
        f"{forecast_start.date()} "
        f"to {forecast_end.date()}"
    )

    series_count = (
        history_df[
            ["store_id", "product_id"]
        ]
        .drop_duplicates()
        .shape[0]
    )

    expected_rows = (
        series_count * HORIZON
    )

    print(
        f"Demand series: "
        f"{series_count:,}"
    )

    print(
        f"Expected evaluated rows: "
        f"{expected_rows:,}"
    )

    print(
        "\nBuilding training features from "
        "historical data only..."
    )

    training_df = build_features(
        history_df,
        drop_na=True,
    )

    if training_df.empty:
        raise RuntimeError(
            "Training feature data is empty."
        )

    if (
        training_df["date"].max()
        > HISTORY_END_DATE
    ):
        raise RuntimeError(
            "Training features cross the "
            "historical cutoff."
        )

    print(
        f"Training rows: "
        f"{len(training_df):,}"
    )

    print(
        f"Training feature range: "
        f"{training_df['date'].min().date()} "
        f"to "
        f"{training_df['date'].max().date()}"
    )

    X_train = training_df[
        FEATURE_COLUMNS
    ]

    y_train = training_df[
        "sales"
    ]

    print(
        "\nTraining cutoff-specific XGBoost..."
    )

    model = build_cutoff_model()

    model.fit(
        X_train,
        y_train,
        verbose=False,
    )

    print(
        "\nRunning genuine recursive "
        "7-day backtest..."
    )

    evaluated, metrics = run_recursive_backtest(
        model=model,
        history_df=history_df,
        actual_df=future_actual_df,
        feature_columns=FEATURE_COLUMNS,
        horizon=HORIZON,
    )

    if len(evaluated) != expected_rows:
        raise RuntimeError(
            "Unexpected evaluated row count: "
            f"expected {expected_rows:,}, "
            f"got {len(evaluated):,}."
        )

    print("\n" + "=" * 58)
    print(
        "GENUINE 7-DAY RECURSIVE HOLDOUT RESULTS"
    )
    print("=" * 58)

    print(
        f"Evaluated rows: "
        f"{len(evaluated):,}"
    )

    print(
        f"Forecast range: "
        f"{evaluated['date'].min().date()} "
        f"to "
        f"{evaluated['date'].max().date()}"
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
        f"{metrics['wape']:.4f}"
    )

    print(
        f"WAPE%: "
        f"{metrics['wape'] * 100:.2f}%"
    )

    print("\n" + "=" * 58)
    print("HORIZON-WISE ERROR")
    print("=" * 58)

    for step in range(
        1,
        HORIZON + 1,
    ):
        step_df = evaluated[
            evaluated["horizon_step"] == step
        ]

        if step_df.empty:
            raise RuntimeError(
                f"Missing horizon step {step}."
            )

        step_metrics = evaluate_predictions(
            step_df["actual"],
            step_df["forecast"],
        )

        print(
            f"Day +{step}: "
            f"MAE={step_metrics['mae']:.4f}, "
            f"RMSE={step_metrics['rmse']:.4f}, "
            f"WAPE="
            f"{step_metrics['wape'] * 100:.2f}%"
        )

    print("\n" + "=" * 58)
    print("PROTOCOL INTERPRETATION")
    print("=" * 58)

    print(
        "Previous 10.05% WAPE:"
    )

    print(
        "  Rolling observed-history holdout "
        "evaluation."
    )

    print(
        "\nCurrent recursive WAPE:"
    )

    print(
        "  Genuine 7-day multi-step evaluation "
        "where predictions feed later steps."
    )

    print(
        "\nUse the recursive result when claiming "
        "7-day deployment performance."
    )


if __name__ == "__main__":
    main()