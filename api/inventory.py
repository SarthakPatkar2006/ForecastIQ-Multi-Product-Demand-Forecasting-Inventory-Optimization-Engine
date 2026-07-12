from api.schemas import (
    InventoryRequest,
    InventoryResponse,
)

from src.inventory.engine import (
    make_inventory_decision,
)


def generate_inventory_recommendation(
    request: InventoryRequest,
) -> InventoryResponse:
    """
    Generate an inventory recommendation from
    a demand forecast.
    """

    decision = make_inventory_decision(
        forecast_values=request.forecast,
        current_inventory=request.current_inventory,
        lead_time_days=request.lead_time_days,
        service_level=request.service_level,
        historical_demand_std=request.historical_demand_std,
    )

    return InventoryResponse(
        expected_lead_time_demand=decision.expected_lead_time_demand,
        safety_stock=decision.safety_stock,
        reorder_point=decision.reorder_point,
        reorder_status=decision.reorder_status,
        recommended_order_quantity=decision.recommended_order_quantity,
        stockout_risk_score=decision.stockout_risk_score,
        stockout_risk_level=decision.stockout_risk_level,
    )