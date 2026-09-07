"""Currently loaded model metadata response."""

from datetime import datetime

from pydantic import BaseModel


class ModelInfoResponse(BaseModel):
    registered_name: str
    alias: str
    version: str
    run_id: str
    features_count: int
    symbol: str
    timeframe: str
    loaded_at: datetime
    backtest_status: str | None
