from datetime import date

from pydantic import BaseModel, Field


class ForecastRequest(BaseModel):
    state: str
    district: str | None = None
    latitude: float | None = Field(default=None, ge=-90, le=90)
    longitude: float | None = Field(default=None, ge=-180, le=180)
    target_date: date


class ForecastResponse(BaseModel):
    state: str
    district: str | None
    target_date: date
    tmax_c: float
    tmin_c: float | None = None
    rainfall_mm: float | None = None
    confidence: str
    forecast_type: str
    note: str
