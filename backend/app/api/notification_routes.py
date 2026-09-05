from fastapi import APIRouter
from typing import List, Dict, Any, Optional
from backend.app.services.notification_service import notification_service

router = APIRouter(prefix="/notifications", tags=["Notifications & Alerts"])

@router.get("", response_model=List[Dict[str, Any]])
async def get_notifications(crop_name: Optional[str] = None, city: Optional[str] = "Hyderabad"):
    return notification_service.get_notifications(crop_name=crop_name, city=city)
