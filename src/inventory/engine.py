from __future__ import annotations

from dataclasses import asdict, dataclass
from math import sqrt

import numpy as np
from scipy.stats import norm


VALID_RISK_LEVELS = {
    "LOW",
    "MEDIUM",
    "HIGH",
}

VALID_REORDER_STATUSES = {
    "REORDER_NOW",
    "MONITOR",
    "SUFFICIENT_STOCK",
}


@dataclass(frozen=True)
class InventoryDecision:
    expected_lead_time_demand: float
    safety_stock: float
    reorder_point: float
    reorder_status: str
    recommended_order_quantity: float
    stockout_risk_score: float
    stockout_risk_level: str

    def to_dict(self) -> dict:
        return asdict(self)


def _validate_inputs(
    forecast_values,
    current_inventory: float,
    lead_time_days: int,
    service_level: float,
    historical_demand_std: float,
) -> np.ndarray:
    forecasts = np.asarray(
        forecast_values,
        dtype=float,
    )

    if forecasts.ndim != 1:
        raise ValueError(
            "forecast_values must be one-dimensional."
        )

    if forecasts.size == 0:
        raise ValueError(
            "forecast_values cannot be empty."
        )

    if not np.isfinite(forecasts).all():
        raise ValueError(
            "forecast_values must contain only finite values."
        )

    if (forecasts < 0).any():
        raise ValueError(
            "forecast_values cannot contain negative values."
        )

    if not np.isfinite(current_inventory):
        raise ValueError(
            "current_inventory must be finite."
        )

    if current_inventory < 0:
        raise ValueError(
            "current_inventory cannot be negative."
        )

    if (
        not isinstance(lead_time_days, int)
        or isinstance(lead_time_days, bool)
        or lead_time_days <= 0
    ):
        raise ValueError(
            "lead_time_days must be a positive integer."
        )

    if not np.isfinite(service_level):
        raise ValueError(
            "service_level must be finite."
        )

    if not 0 < service_level < 1:
        raise ValueError(
            "service_level must be between 0 and 1."
        )

    if not np.isfinite(historical_demand_std):
        raise ValueError(
            "historical_demand_std must be finite."
        )

    if historical_demand_std < 0:
        raise ValueError(
            "historical_demand_std cannot be negative."
        )

    return forecasts


def calculate_expected_lead_time_demand(
    forecast_values,
    lead_time_days: int,
) -> float:
    """
    Estimate demand during replenishment lead time.

    If lead time exceeds the forecast horizon, use the
    average forecast demand to extrapolate the remaining days.
    """
    forecasts = np.asarray(
        forecast_values,
        dtype=float,
    )

    if forecasts.size == 0:
        raise ValueError(
            "forecast_values cannot be empty."
        )

    if lead_time_days <= 0:
        raise ValueError(
            "lead_time_days must be positive."
        )

    covered_days = min(
        lead_time_days,
        forecasts.size,
    )

    expected_demand = float(
        forecasts[:covered_days].sum()
    )

    remaining_days = (
        lead_time_days - covered_days
    )

    if remaining_days > 0:
        expected_demand += float(
            forecasts.mean()
            * remaining_days
        )

    return expected_demand


def calculate_safety_stock(
    historical_demand_std: float,
    lead_time_days: int,
    service_level: float,
) -> float:
    """
    Safety Stock =
        Z(service level)
        × demand standard deviation
        × sqrt(lead time)
    """
    z_score = float(
        norm.ppf(service_level)
    )

    safety_stock = (
        z_score
        * historical_demand_std
        * sqrt(lead_time_days)
    )

    return float(
        max(0.0, safety_stock)
    )


def calculate_reorder_status(
    current_inventory: float,
    reorder_point: float,
) -> str:
    if current_inventory <= reorder_point:
        return "REORDER_NOW"

    if current_inventory <= (
        reorder_point * 1.20
    ):
        return "MONITOR"

    return "SUFFICIENT_STOCK"


def calculate_recommended_order_quantity(
    forecast_values,
    safety_stock: float,
    current_inventory: float,
) -> float:
    forecast_horizon_demand = float(
        np.sum(forecast_values)
    )

    quantity = (
        forecast_horizon_demand
        + safety_stock
        - current_inventory
    )

    return float(
        max(0.0, quantity)
    )


def calculate_stockout_risk(
    current_inventory: float,
    expected_lead_time_demand: float,
    safety_stock: float,
) -> tuple[float, str]:
    """
    Heuristic stockout-risk score.

    This is intentionally NOT presented as a calibrated
    probability of stockout.
    """
    risk_adjusted_demand = (
        expected_lead_time_demand
        + safety_stock
    )

    if risk_adjusted_demand <= 0:
        score = 0.0
    else:
        score = (
            risk_adjusted_demand
            - current_inventory
        ) / risk_adjusted_demand

        score = float(
            np.clip(
                score,
                a_min=0.0,
                a_max=1.0,
            )
        )

    if score <= 0.30:
        level = "LOW"
    elif score <= 0.60:
        level = "MEDIUM"
    else:
        level = "HIGH"

    return score, level


def make_inventory_decision(
    forecast_values,
    current_inventory: float,
    lead_time_days: int = 7,
    service_level: float = 0.95,
    historical_demand_std: float = 0.0,
) -> InventoryDecision:
    """
    Convert demand forecasts into inventory actions.
    """
    forecasts = _validate_inputs(
        forecast_values=forecast_values,
        current_inventory=current_inventory,
        lead_time_days=lead_time_days,
        service_level=service_level,
        historical_demand_std=historical_demand_std,
    )

    expected_lead_time_demand = (
        calculate_expected_lead_time_demand(
            forecast_values=forecasts,
            lead_time_days=lead_time_days,
        )
    )

    safety_stock = calculate_safety_stock(
        historical_demand_std=historical_demand_std,
        lead_time_days=lead_time_days,
        service_level=service_level,
    )

    reorder_point = (
        expected_lead_time_demand
        + safety_stock
    )

    reorder_status = (
        calculate_reorder_status(
            current_inventory=current_inventory,
            reorder_point=reorder_point,
        )
    )

    recommended_order_quantity = (
        calculate_recommended_order_quantity(
            forecast_values=forecasts,
            safety_stock=safety_stock,
            current_inventory=current_inventory,
        )
    )

    (
        stockout_risk_score,
        stockout_risk_level,
    ) = calculate_stockout_risk(
        current_inventory=current_inventory,
        expected_lead_time_demand=(
            expected_lead_time_demand
        ),
        safety_stock=safety_stock,
    )

    return InventoryDecision(
        expected_lead_time_demand=float(
            expected_lead_time_demand
        ),
        safety_stock=float(
            safety_stock
        ),
        reorder_point=float(
            reorder_point
        ),
        reorder_status=reorder_status,
        recommended_order_quantity=float(
            recommended_order_quantity
        ),
        stockout_risk_score=float(
            stockout_risk_score
        ),
        stockout_risk_level=stockout_risk_level,
    )