from src.data.loader import load_sales_data
from src.data.validator import validate_sales_data


DATA_PATH = "data/raw/train.csv"


df = load_sales_data(DATA_PATH)

report = validate_sales_data(df)


print("\n=== DATASET ===")
print(df.head())

print("\n=== SHAPE ===")
print(df.shape)

print("\n=== COLUMNS ===")
print(df.columns.tolist())

print("\n=== DATE RANGE ===")
print(df["date"].min(), "to", df["date"].max())

print("\n=== STORES ===")
print(df["store_id"].nunique())

print("\n=== PRODUCTS ===")
print(df["product_id"].nunique())

print("\n=== DEMAND SERIES ===")
print(
    df[
        ["store_id", "product_id"]
    ]
    .drop_duplicates()
    .shape[0]
)

print("\n=== VALIDATION ===")
print("Valid:", report.is_valid)
print("Errors:", report.errors)
print("Warnings:", report.warnings)