from fastapi import APIRouter
from backend.app.schemas.models import CropRecommendationRequest, CropRecommendationResponse
from backend.app.services.crop_service import crop_service

router = APIRouter(prefix="/crop", tags=["Crop Recommendation"])

@router.post("/recommend", response_model=CropRecommendationResponse)
async def recommend_crop(req: CropRecommendationRequest):
    return crop_service.recommend_crops(
        soil=req.soil_properties,
        city=req.city,
        lat=req.latitude,
        lon=req.longitude,
        custom_temp=req.temperature,
        custom_humidity=req.humidity,
        custom_rainfall=req.rainfall
    )
