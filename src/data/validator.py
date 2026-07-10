from dataclasses import dataclass, field

import pandas as pd


REQUIRED_COLUMNS = {
    "date",
    "store_id",
    "product_id",
    "sales",
}


@dataclass
class ValidationReport:
    is_valid: bool
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)


def validate_sales_data(
    df: pd.DataFrame,
) -> ValidationReport:
    """
    Validate normalized multi-product retail sales data.
    """
    errors: list[str] = []
    warnings: list[str] = []

    if df.empty:
        return ValidationReport(
            is_valid=False,
            errors=["Dataset is empty."],
        )

    missing_columns = REQUIRED_COLUMNS - set(df.columns)

    if missing_columns:
        errors.append(
            "Missing required columns: "
            + ", ".join(sorted(missing_columns))
        )

        return ValidationReport(
            is_valid=False,
            errors=errors,
            warnings=warnings,
        )

    # Invalid or missing dates
    invalid_dates = int(df["date"].isna().sum())

    if invalid_dates > 0:
        errors.append(
            f"Found {invalid_dates} invalid or missing dates."
        )

    # Missing store IDs
    missing_stores = int(df["store_id"].isna().sum())

    if missing_stores > 0:
        errors.append(
            f"Found {missing_stores} missing store IDs."
        )

    # Missing product IDs
    missing_products = int(
        df["product_id"].isna().sum()
    )

    if missing_products > 0:
        errors.append(
            f"Found {missing_products} missing product IDs."
        )

    # Missing sales
    missing_sales = int(df["sales"].isna().sum())

    if missing_sales > 0:
        errors.append(
            f"Found {missing_sales} missing sales values."
        )

    # Negative sales
    negative_sales = int(
        (df["sales"] < 0).sum()
    )

    if negative_sales > 0:
        errors.append(
            f"Found {negative_sales} negative sales values."
        )

    # Duplicate observations within one demand series
    duplicate_count = int(
        df.duplicated(
            subset=[
                "date",
                "store_id",
                "product_id",
            ]
        ).sum()
    )

    if duplicate_count > 0:
        errors.append(
            f"Found {duplicate_count} duplicate "
            "store-product-date observations."
        )

    # Check history depth per store-product series
    history_counts = (
        df.groupby(
            ["store_id", "product_id"]
        )["date"]
        .nunique()
    )

    short_series = int(
        (history_counts < 35).sum()
    )

    if short_series > 0:
        warnings.append(
            f"{short_series} demand series have fewer "
            "than 35 historical observations."
        )

    return ValidationReport(
        is_valid=len(errors) == 0,
        errors=errors,
        warnings=warnings,
    )