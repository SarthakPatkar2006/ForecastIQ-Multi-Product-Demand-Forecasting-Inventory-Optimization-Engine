from pathlib import Path

from src.data.loader import load_sales_data


DATA_PATH = "data/raw/train.csv"

OUTPUT_PATH = (
    "data/raw/inventory_state.csv"
)


def main() -> None:
    print("Loading historical data...")

    history = load_sales_data(
        DATA_PATH
    )

    inventory = (
        history[
            [
                "store_id",
                "product_id",
            ]
        ]
        .drop_duplicates()
        .sort_values(
            [
                "store_id",
                "product_id",
            ]
        )
        .reset_index(drop=True)
    )

    # Explicit scenario defaults.
    # These are user/business inputs, not inferred truth.
    inventory[
        "current_inventory"
    ] = 100.0

    inventory[
        "lead_time_days"
    ] = 7

    inventory[
        "service_level"
    ] = 0.95

    output_path = Path(
        OUTPUT_PATH
    )

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    inventory.to_csv(
        output_path,
        index=False,
    )

    print(
        f"Template rows: "
        f"{len(inventory):,}"
    )

    print(
        f"Saved to: "
        f"{output_path}"
    )

    print(
        "\nWARNING: current_inventory=100 is "
        "a scenario default, not observed stock data."
    )


if __name__ == "__main__":
    main()