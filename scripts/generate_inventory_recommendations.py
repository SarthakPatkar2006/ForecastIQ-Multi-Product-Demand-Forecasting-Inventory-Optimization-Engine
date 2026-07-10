from pathlib import Path

import pandas as pd

from src.data.loader import load_sales_data
from src.inventory.engine import (
    make_inventory_decision,
)


DATA_PATH = "data/raw/train.csv"

FORECAST_PATH = (
    "outputs/forecast_7_day.csv"
)

INVENTORY_PATH = (
    "data/raw/inventory_state.csv"
)

OUTPUT_PATH = (
    "outputs/inventory_recommendations.csv"
)


REQUIRED_INVENTORY_COLUMNS = {
    "store_id",
    "product_id",
    "current_inventory",
    "lead_time_days",
    "service_level",
}


def load_inventory_state(
    path: str,
) -> pd.DataFrame:
    inventory = pd.read_csv(path)

    missing = (
        REQUIRED_INVENTORY_COLUMNS
        - set(inventory.columns)
    )

    if missing:
        raise ValueError(
            "Missing inventory columns: "
            + ", ".join(sorted(missing))
        )

    if inventory[
        ["store_id", "product_id"]
    ].duplicated().any():
        raise ValueError(
            "Inventory state contains duplicate "
            "store-product rows."
        )

    return inventory


def calculate_historical_volatility(
    history: pd.DataFrame,
    lookback_days: int = 28,
) -> pd.DataFrame:
    """
    Calculate recent demand volatility independently
    for each store-product series.
    """
    recent_history = (
        history
        .sort_values(
            [
                "store_id",
                "product_id",
                "date",
            ]
        )
        .groupby(
            [
                "store_id",
                "product_id",
            ],
            sort=False,
        )
        .tail(lookback_days)
    )

    volatility = (
        recent_history
        .groupby(
            [
                "store_id",
                "product_id",
            ],
            as_index=False,
        )["sales"]
        .std()
        .rename(
            columns={
                "sales": "historical_demand_std"
            }
        )
    )

    volatility[
        "historical_demand_std"
    ] = volatility[
        "historical_demand_std"
    ].fillna(0.0)

    return volatility


def main() -> None:
    print("Loading forecasts...")

    forecasts = pd.read_csv(
        FORECAST_PATH,
        parse_dates=["date"],
    )

    print(
        f"Forecast rows: "
        f"{len(forecasts):,}"
    )

    print("Loading inventory state...")

    inventory = load_inventory_state(
        INVENTORY_PATH
    )

    print(
        f"Inventory rows: "
        f"{len(inventory):,}"
    )

    print("Loading historical demand...")

    history = load_sales_data(
        DATA_PATH
    )

    print(
        "Calculating recent demand volatility..."
    )

    volatility = calculate_historical_volatility(
        history=history,
        lookback_days=28,
    )

    inventory = inventory.merge(
        volatility,
        on=[
            "store_id",
            "product_id",
        ],
        how="left",
        validate="one_to_one",
    )

    inventory[
        "historical_demand_std"
    ] = inventory[
        "historical_demand_std"
    ].fillna(0.0)

    forecast_groups = {
        key: group.sort_values(
            "horizon_step"
        )
        for key, group in forecasts.groupby(
            [
                "store_id",
                "product_id",
            ]
        )
    }

    recommendations = []

    print(
        "Generating inventory recommendations..."
    )

    for row in inventory.itertuples(
        index=False
    ):
        key = (
            row.store_id,
            row.product_id,
        )

        if key not in forecast_groups:
            raise ValueError(
                "Missing forecast for series "
                f"{key}."
            )

        series_forecast = (
            forecast_groups[key]
        )

        forecast_values = (
            series_forecast[
                "forecast"
            ]
            .to_numpy(dtype=float)
        )

        decision = make_inventory_decision(
            forecast_values=forecast_values,
            current_inventory=float(
                row.current_inventory
            ),
            lead_time_days=int(
                row.lead_time_days
            ),
            service_level=float(
                row.service_level
            ),
            historical_demand_std=float(
                row.historical_demand_std
            ),
        )

        recommendations.append(
            {
                "store_id": row.store_id,
                "product_id": row.product_id,
                "current_inventory": float(
                    row.current_inventory
                ),
                "lead_time_days": int(
                    row.lead_time_days
                ),
                "service_level": float(
                    row.service_level
                ),
                "historical_demand_std": float(
                    row.historical_demand_std
                ),
                **decision.to_dict(),
            }
        )

    result = pd.DataFrame(
        recommendations
    )

    output_path = Path(
        OUTPUT_PATH
    )

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    result.to_csv(
        output_path,
        index=False,
    )

    print("\nRecommendation generation complete.")

    print(
        f"Recommendation rows: "
        f"{len(result):,}"
    )

    print(
        f"Output saved to: "
        f"{output_path}"
    )

    print("\nReorder status distribution:")

    print(
        result[
            "reorder_status"
        ]
        .value_counts()
        .to_string()
    )

    print("\nRisk level distribution:")

    print(
        result[
            "stockout_risk_level"
        ]
        .value_counts()
        .to_string()
    )

    print("\nSample recommendations:")

    print(
        result.head(10).to_string(
            index=False
        )
    )


if __name__ == "__main__":
    main()