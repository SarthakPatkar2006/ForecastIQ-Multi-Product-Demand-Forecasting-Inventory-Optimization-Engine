from pathlib import Path

from xgboost import XGBRegressor


def save_xgboost_model(
    model: XGBRegressor,
    file_path: str | Path,
) -> Path:
    """
    Persist an XGBoost model using native JSON format.
    """
    path = Path(file_path)

    if path.suffix.lower() != ".json":
        raise ValueError(
            "XGBoost model path must use .json extension."
        )

    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    model.save_model(path)

    return path


def load_xgboost_model(
    file_path: str | Path,
) -> XGBRegressor:
    """
    Load a persisted XGBoost JSON model.
    """
    path = Path(file_path)

    if not path.exists():
        raise FileNotFoundError(
            f"Model artifact not found: {path}"
        )

    model = XGBRegressor()

    model.load_model(path)

    return model