import numpy as np
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
)


def mae(
    y_true,
    y_pred,
) -> float:
    return float(
        mean_absolute_error(
            y_true,
            y_pred,
        )
    )


def rmse(
    y_true,
    y_pred,
) -> float:
    return float(
        np.sqrt(
            mean_squared_error(
                y_true,
                y_pred,
            )
        )
    )


def wape(
    y_true,
    y_pred,
) -> float:
    actual = np.asarray(
        y_true,
        dtype=float,
    )

    predicted = np.asarray(
        y_pred,
        dtype=float,
    )

    denominator = np.abs(actual).sum()

    if denominator == 0:
        raise ValueError(
            "WAPE is undefined when total actual demand is zero."
        )

    return float(
        np.abs(actual - predicted).sum()
        / denominator
    )


def evaluate_predictions(
    y_true,
    y_pred,
) -> dict[str, float]:
    return {
        "mae": mae(y_true, y_pred),
        "rmse": rmse(y_true, y_pred),
        "wape": wape(y_true, y_pred),
    }