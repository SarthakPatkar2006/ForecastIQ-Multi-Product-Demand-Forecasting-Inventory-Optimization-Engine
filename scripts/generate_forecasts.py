from pathlib import Path

from src.data.loader import load_sales_data
from src.forecasting.recursive import (
    recursive_forecast,
)
from src.models.persistence import (
    load_xgboost_model,
)
from src.models.train import FEATURE_COLUMNS


DATA_PATH = "data/raw/train.csv"

MODEL_PATH = (
    "artifacts/forecastiq_xgboost.json"
)

OUTPUT_PATH = (
    "outputs/forecast_7_day.csv"
)

FORECAST_HORIZON = 7


def main() -> None:
    print("Loading historical data...")

    history_df = load_sales_data(
        DATA_PATH
    )

    print(
        f"Historical rows: "
        f"{len(history_df):,}"
    )

    series_count = (
        history_df[
            ["store_id", "product_id"]
        ]
        .drop_duplicates()
        .shape[0]
    )

    print(
        f"Demand series: "
        f"{series_count:,}"
    )

    print("\nLoading trained model...")

    model = load_xgboost_model(
        MODEL_PATH
    )

    print(
        f"\nGenerating "
        f"{FORECAST_HORIZON}-day recursive forecasts..."
    )

    forecasts = recursive_forecast(
        model=model,
        history_df=history_df,
        feature_columns=FEATURE_COLUMNS,
        horizon=FORECAST_HORIZON,
    )

    expected_rows = (
        series_count
        * FORECAST_HORIZON
    )

    if len(forecasts) != expected_rows:
        raise RuntimeError(
            "Unexpected forecast row count: "
            f"expected {expected_rows:,}, "
            f"got {len(forecasts):,}."
        )

    if forecasts["forecast"].isna().any():
        raise RuntimeError(
            "Generated forecasts contain missing values."
        )

    if (forecasts["forecast"] < 0).any():
        raise RuntimeError(
            "Generated forecasts contain negative values."
        )

    output_path = Path(
        OUTPUT_PATH
    )

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    forecasts.to_csv(
        output_path,
        index=False,
    )

    print("\nForecast generation complete.")

    print(
        f"Forecast rows: "
        f"{len(forecasts):,}"
    )

    print(
        f"Expected rows: "
        f"{expected_rows:,}"
    )

    print(
        f"Forecast date range: "
        f"{forecasts['date'].min().date()} "
        f"to "
        f"{forecasts['date'].max().date()}"
    )

    print(
        f"Output saved to: "
        f"{output_path}"
    )

    print("\nSample forecasts:")

    print(
        forecasts.head(10).to_string(
            index=False
        )
    )


if __name__ == "__main__":
    main()