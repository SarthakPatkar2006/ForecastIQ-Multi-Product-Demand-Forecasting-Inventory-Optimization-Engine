from pathlib import Path

from src.models.explain import (
    get_feature_importance,
)
from src.models.persistence import (
    load_xgboost_model,
)
from src.models.train import FEATURE_COLUMNS


MODEL_PATH = (
    "artifacts/forecastiq_xgboost.json"
)

OUTPUT_PATH = (
    "outputs/feature_importance.csv"
)


def main() -> None:
    print("Loading production model...")

    model = load_xgboost_model(
        MODEL_PATH
    )

    print("Extracting feature importance...")

    importance_df = get_feature_importance(
        model=model,
        feature_columns=FEATURE_COLUMNS,
    )

    output_path = Path(
        OUTPUT_PATH
    )

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    importance_df.to_csv(
        output_path,
        index=False,
    )

    print("\n=== FEATURE IMPORTANCE ===")

    print(
        importance_df.to_string(
            index=False
        )
    )

    print(
        f"\nSaved to: {output_path}"
    )


if __name__ == "__main__":
    main()