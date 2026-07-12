from pydantic import BaseModel


class ForecastRequest(BaseModel):
    store_id: int
    product_id: int
    history_days: int = 60
    horizon: int = 7


class ForecastResponse(BaseModel):
    store_id: int
    product_id: int
    horizon: int
    forecast: list[float]


class InventoryRequest(BaseModel):
    forecast: list[float]
    current_inventory: float
    lead_time_days: int
    service_level: float
    historical_demand_std: float


class InventoryResponse(BaseModel):
    expected_lead_time_demand: float
    safety_stock: float
    reorder_point: float
    reorder_status: str
    recommended_order_quantity: float
    stockout_risk_score: float
    stockout_risk_level: str