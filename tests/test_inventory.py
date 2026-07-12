import pytest

from src.inventory.engine import (
    calculate_expected_lead_time_demand,
    calculate_recommended_order_quantity,
    calculate_reorder_status,
    calculate_safety_stock,
    calculate_stockout_risk,
    make_inventory_decision,
)


def test_expected_lead_time_demand_uses_forecast_sum():
    result = calculate_expected_lead_time_demand(
        forecast_values=[10, 12, 14, 16],
        lead_time_days=3,
    )

    assert result == pytest.approx(36.0)


def test_expected_demand_extrapolates_beyond_horizon():
    result = calculate_expected_lead_time_demand(
        forecast_values=[10, 20],
        lead_time_days=4,
    )

    assert result == pytest.approx(60.0)


def test_safety_stock_is_non_negative():
    result = calculate_safety_stock(
        historical_demand_std=5.0,
        lead_time_days=7,
        service_level=0.95,
    )

    assert result > 0


def test_reorder_now_status():
    assert (
        calculate_reorder_status(
            current_inventory=90,
            reorder_point=100,
        )
        == "REORDER_NOW"
    )


def test_monitor_status():
    assert (
        calculate_reorder_status(
            current_inventory=110,
            reorder_point=100,
        )
        == "MONITOR"
    )


def test_sufficient_stock_status():
    assert (
        calculate_reorder_status(
            current_inventory=130,
            reorder_point=100,
        )
        == "SUFFICIENT_STOCK"
    )


def test_recommended_quantity_clamped_to_zero():
    result = calculate_recommended_order_quantity(
        forecast_values=[10, 10, 10],
        safety_stock=5,
        current_inventory=100,
    )

    assert result == 0.0


def test_stockout_risk_is_bounded():
    score, level = calculate_stockout_risk(
        current_inventory=20,
        expected_lead_time_demand=100,
        safety_stock=10,
    )

    assert 0.0 <= score <= 1.0

    assert level in {
        "LOW",
        "MEDIUM",
        "HIGH",
    }


def test_zero_risk_adjusted_demand_is_low_risk():
    score, level = calculate_stockout_risk(
        current_inventory=10,
        expected_lead_time_demand=0,
        safety_stock=0,
    )

    assert score == 0.0
    assert level == "LOW"


def test_inventory_decision_returns_consistent_values():
    decision = make_inventory_decision(
        forecast_values=[
            10,
            12,
            14,
            16,
            18,
            20,
            22,
        ],
        current_inventory=50,
        lead_time_days=3,
        service_level=0.95,
        historical_demand_std=4.0,
    )

    assert (
        decision.expected_lead_time_demand
        == pytest.approx(36.0)
    )

    assert decision.safety_stock > 0

    assert decision.reorder_point == pytest.approx(
        decision.expected_lead_time_demand
        + decision.safety_stock
    )


def test_negative_inventory_rejected():
    with pytest.raises(ValueError):
        make_inventory_decision(
            forecast_values=[10, 20, 30],
            current_inventory=-1,
        )


def test_invalid_service_level_rejected():
    with pytest.raises(ValueError):
        make_inventory_decision(
            forecast_values=[10, 20, 30],
            current_inventory=50,
            service_level=1.0,
        )


def test_negative_forecast_rejected():
    with pytest.raises(ValueError):
        make_inventory_decision(
            forecast_values=[10, -2, 30],
            current_inventory=50,
      
        )