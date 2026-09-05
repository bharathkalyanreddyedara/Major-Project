from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any

# Soil & Manual Properties
class SoilAnalysisResponse(BaseModel):
    detected_soil_type: str
    confidence: float
    is_valid_soil: bool = True
    rejection_reason: Optional[str] = None
    all_probabilities: Dict[str, float] = {}
    visual_features: Dict[str, Any] = {}

class ManualSoilProperties(BaseModel):
    nitrogen: float = Field(..., description="Nitrogen content (N) in kg/ha or ppm", example=90.0)
    phosphorus: float = Field(..., description="Phosphorus content (P) in kg/ha or ppm", example=42.0)
    potassium: float = Field(..., description="Potassium content (K) in kg/ha or ppm", example=43.0)
    ph: float = Field(..., description="Soil pH level (0-14)", example=6.5)
    soil_type: Optional[str] = Field(None, description="Soil Type (e.g. Red, Black, Sandy, Clayey, Alluvial)", example="Black")
    moisture: Optional[float] = Field(45.0, description="Soil moisture percentage")
    zinc: Optional[float] = Field(None, description="Zinc content (Zn)")
    sulphur: Optional[float] = Field(None, description="Sulphur content (S)")
    electrical_conductivity: Optional[float] = Field(None, description="EC in dS/m")

class CropRecommendationRequest(BaseModel):
    soil_properties: ManualSoilProperties
    latitude: Optional[float] = None
    longitude: Optional[float] = None
    city: Optional[str] = "Hyderabad"
    temperature: Optional[float] = None
    humidity: Optional[float] = None
    rainfall: Optional[float] = None

class RecommendedCrop(BaseModel):
    crop_name: str
    confidence: float
    suitability_score: float
    recommended_season: str
    soil_compatibility: str
    water_requirement: str
    growth_duration_days: int
    optimal_fertilizers: List[str]

class CropRecommendationResponse(BaseModel):
    recommendations: List[RecommendedCrop]
    environmental_context: Dict[str, Any]

# Lifecycle Timeline
class TimelineStage(BaseModel):
    stage_id: int
    stage_name: str
    start_day: int
    end_day: int
    status: str = "upcoming"  # upcoming, current, completed
    activities: List[str]
    irrigation_schedule: str
    fertilizer_advice: str
    pest_disease_watch: str
    critical_notes: str

class CropTimelineRequest(BaseModel):
    crop_name: str
    sowing_date: str # YYYY-MM-DD
    soil_type: str
    location: str

class CropTimelineResponse(BaseModel):
    crop_name: str
    sowing_date: str
    expected_harvest_date: str
    total_duration_days: int
    current_day: int
    current_stage: str
    stages: List[TimelineStage]
    active_notifications: List[Dict[str, Any]]

# Chat / RAG
class ChatMessage(BaseModel):
    role: str # user or assistant
    content: str

class ChatRequest(BaseModel):
    query: str
    crop_context: Optional[str] = None
    soil_context: Optional[str] = None
    growth_stage_context: Optional[str] = None
    weather_context: Optional[str] = None
    history: Optional[List[ChatMessage]] = []

class ChatResponse(BaseModel):
    answer: str
    grounded_sources: List[str]
    suggested_actions: List[str]
