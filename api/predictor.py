from pathlib import Path

import pandas as pd

from src.models.config import build_xgboost_model


MODEL_PATH = Path("artifacts/forecastiq_xgboost.json")


class ForecastPredictor:
    def __init__(self) -> None:
        self.model = build_xgboost_model()

        self.model.load_model(MODEL_PATH)

    def predict(
        self,
        features: pd.DataFrame,
    ) -> list[float]:
        predictions = self.model.predict(features)

        predictions = predictions.clip(min=0)

        return predictions.tolist()


predictor = ForecastPredictor()