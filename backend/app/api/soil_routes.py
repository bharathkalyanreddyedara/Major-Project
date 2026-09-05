from fastapi import APIRouter, UploadFile, File, HTTPException
from backend.app.services.soil_service import soil_service
from backend.app.schemas.models import SoilAnalysisResponse

router = APIRouter(prefix="/soil", tags=["Soil Analysis"])

@router.post("/analyze", response_model=SoilAnalysisResponse)
async def analyze_soil(file: UploadFile = File(...)):
    if not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="Uploaded file must be an image.")
    
    try:
        contents = await file.read()
        analysis = soil_service.analyze_soil_image(contents)
        return SoilAnalysisResponse(**analysis)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to analyze soil image: {str(e)}")
