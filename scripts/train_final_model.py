from src.data.loader import load_sales_data
from src.features.build_features import build_features
from src.models.persistence import (
    save_xgboost_model,
)
from src.models.train import FEATURE_COLUMNS

from src.models.config import build_xgboost_model


DATA_PATH = "data/raw/train.csv"

MODEL_PATH = (
    "artifacts/forecastiq_xgboost.json"
)



def main() -> None:
    print("Loading full historical dataset...")

    df = load_sales_data(DATA_PATH)

    print(f"Raw rows: {len(df):,}")

    print("\nBuilding leakage-safe features...")

    featured_df = build_features(
        df,
        drop_na=True,
    )

    print(
        f"Training rows: "
        f"{len(featured_df):,}"
    )

    X = featured_df[
        FEATURE_COLUMNS
    ]

    y = featured_df[
        "sales"
    ]

    print(
        "\nTraining production XGBoost "
        "on all available history..."
    )

    model = build_xgboost_model()

    model.fit(
        X,
        y,
        verbose=False,
    )

    print("\nSaving model artifact...")

    saved_path = save_xgboost_model(
        model,
        MODEL_PATH,
    )

    print(
        f"Model saved to: {saved_path}"
    )


if __name__ == "__main__":
    main()