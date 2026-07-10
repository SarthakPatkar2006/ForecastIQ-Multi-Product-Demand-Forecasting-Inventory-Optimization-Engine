from pathlib import Path

import pandas as pd


COLUMN_MAPPING = {
    "store": "store_id",
    "item": "product_id",
}


def load_sales_data(file_path: str | Path) -> pd.DataFrame:
    """
    Load and normalize historical retail sales data.

    Expected raw schema:
        date, store, item, sales

    Normalized schema:
        date, store_id, product_id, sales
    """
    path = Path(file_path)

    if not path.exists():
        raise FileNotFoundError(
            f"Dataset not found: {path}"
        )

    if path.suffix.lower() != ".csv":
        raise ValueError(
            f"Expected CSV file, got: {path.suffix}"
        )

    df = pd.read_csv(path)

    df = df.rename(columns=COLUMN_MAPPING)

    if "date" in df.columns:
        df["date"] = pd.to_datetime(
            df["date"],
            errors="coerce",
        )

    if "sales" in df.columns:
        df["sales"] = pd.to_numeric(
            df["sales"],
            errors="coerce",
        )

    sort_columns = [
        column
        for column in [
            "store_id",
            "product_id",
            "date",
        ]
        if column in df.columns
    ]

    if sort_columns:
        df = df.sort_values(
            by=sort_columns,
            kind="stable",
        )

    return df.reset_index(drop=True)