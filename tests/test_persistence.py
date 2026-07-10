import numpy as np
import pandas as pd
import pytest

from xgboost import XGBRegressor

from src.models.persistence import (
    load_xgboost_model,
    save_xgboost_model,
)


def test_model_round_trip_preserves_predictions(
    tmp_path,
):
    X = pd.DataFrame(
        {
            "feature_1": [1, 2, 3, 4, 5, 6],
            "feature_2": [6, 5, 4, 3, 2, 1],
        }
    )

    y = np.array(
        [2, 4, 6, 8, 10, 12],
        dtype=float,
    )

    model = XGBRegressor(
        n_estimators=10,
        max_depth=2,
        random_state=42,
    )

    model.fit(X, y)

    expected = model.predict(X)

    model_path = (
        tmp_path / "test_model.json"
    )

    save_xgboost_model(
        model,
        model_path,
    )

    loaded_model = load_xgboost_model(
        model_path
    )

    actual = loaded_model.predict(X)

    np.testing.assert_allclose(
        expected,
        actual,
        rtol=1e-6,
        atol=1e-6,
    )


def test_save_rejects_non_json_extension(
    tmp_path,
):
    model = XGBRegressor()

    with pytest.raises(ValueError):
        save_xgboost_model(
            model,
            tmp_path / "model.pkl",
        )


def test_load_missing_model_raises_error(
    tmp_path,
):
    with pytest.raises(FileNotFoundError):
        load_xgboost_model(
            tmp_path / "missing.json"
        )