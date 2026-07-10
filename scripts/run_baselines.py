from src.data.loader import load_sales_data
from src.models.baselines import (
    add_last_value_forecast,
    add_seasonal_naive_forecast,
)
from src.models.evaluate import evaluate_predictions
from src.splitting.temporal_split import (
    temporal_train_val_test_split,
)


DATA_PATH = "data/raw/train.csv"


def print_metrics(
    model_name: str,
    metrics: dict[str, float],
) -> None:
    print(f"\n=== {model_name} ===")
    print(f"MAE:  {metrics['mae']:.4f}")
    print(f"RMSE: {metrics['rmse']:.4f}")
    print(f"WAPE: {metrics['wape']:.4f}")
    print(f"WAPE%: {metrics['wape'] * 100:.2f}%")


def main() -> None:
    print("Loading dataset...")

    df = load_sales_data(DATA_PATH)

    print(f"Rows: {len(df):,}")
    print(
        f"Date range: "
        f"{df['date'].min().date()} "
        f"to {df['date'].max().date()}"
    )

    # Important:
    # Generate baseline predictions on full chronological
    # history first, then evaluate only validation dates.
    #
    # This preserves access to legitimate historical values
    # immediately before the validation boundary.
    print("\nGenerating baseline forecasts...")

    baseline_df = add_last_value_forecast(df)

    baseline_df = add_seasonal_naive_forecast(
        baseline_df,
        season_length=7,
    )

    print("Creating temporal split...")

    split = temporal_train_val_test_split(
        baseline_df,
        train_ratio=0.70,
        validation_ratio=0.15,
    )

    validation = split.validation.copy()

    print(
        f"Train rows:      {len(split.train):,}"
    )
    print(
        f"Validation rows: {len(validation):,}"
    )
    print(
        f"Test rows:       {len(split.test):,}"
    )

    print(
        "\nValidation date range: "
        f"{validation['date'].min().date()} "
        f"to {validation['date'].max().date()}"
    )

    # Last-value baseline
    last_value_eval = validation.dropna(
        subset=["last_value_forecast"]
    )

    last_value_metrics = evaluate_predictions(
        last_value_eval["sales"],
        last_value_eval["last_value_forecast"],
    )

    # Seasonal-naive baseline
    seasonal_eval = validation.dropna(
        subset=["seasonal_naive_forecast"]
    )

    seasonal_metrics = evaluate_predictions(
        seasonal_eval["sales"],
        seasonal_eval["seasonal_naive_forecast"],
    )

    print_metrics(
        "LAST VALUE BASELINE",
        last_value_metrics,
    )

    print_metrics(
        "SEASONAL NAIVE BASELINE",
        seasonal_metrics,
    )


if __name__ == "__main__":
    main()