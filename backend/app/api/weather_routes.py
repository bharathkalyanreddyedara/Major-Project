from fastapi import APIRouter
from typing import Optional
from backend.app.services.weather_service import weather_service

router = APIRouter(prefix="/weather", tags=["Weather Information"])

@router.get("")
async def get_weather_data(city: Optional[str] = "Hyderabad", lat: Optional[float] = None, lon: Optional[float] = None):
    return weather_service.get_weather(city=city, lat=lat, lon=lon)
