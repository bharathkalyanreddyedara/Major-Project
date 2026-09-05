from fastapi import APIRouter
from backend.app.schemas.models import CropTimelineRequest, CropTimelineResponse
from backend.app.services.timeline_service import timeline_service

router = APIRouter(prefix="/timeline", tags=["Crop Lifecycle Timeline"])

@router.post("/generate", response_model=CropTimelineResponse)
async def generate_timeline(req: CropTimelineRequest):
    return timeline_service.generate_timeline(
        crop_name=req.crop_name,
        sowing_date_str=req.sowing_date,
        soil_type=req.soil_type,
        location=req.location
    )
