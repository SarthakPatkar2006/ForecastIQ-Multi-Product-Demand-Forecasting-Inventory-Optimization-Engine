import pytest

from src.models.evaluate import (
    evaluate_predictions,
    mae,
    rmse,
    wape,
)


def test_mae():
    y_true = [10, 20, 30]
    y_pred = [12, 18, 33]

    result = mae(y_true, y_pred)

    assert result == pytest.approx(
        7 / 3
    )


def test_rmse():
    y_true = [10, 20, 30]
    y_pred = [12, 18, 33]

    result = rmse(y_true, y_pred)

    expected = (
        (4 + 4 + 9) / 3
    ) ** 0.5

    assert result == pytest.approx(expected)


def test_wape():
    y_true = [10, 20, 30]
    y_pred = [12, 18, 33]

    result = wape(y_true, y_pred)

    assert result == pytest.approx(
        7 / 60
    )


def test_evaluate_predictions_returns_all_metrics():
    result = evaluate_predictions(
        [10, 20, 30],
        [12, 18, 33],
    )

    assert set(result.keys()) == {
        "mae",
        "rmse",
        "wape",
    }


def test_wape_rejects_zero_total_demand():
    with pytest.raises(ValueError):
        wape(
            [0, 0, 0],
            [1, 2, 3],
        )