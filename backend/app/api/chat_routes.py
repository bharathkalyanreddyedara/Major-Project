from fastapi import APIRouter
from backend.app.schemas.models import ChatRequest, ChatResponse
from backend.app.services.rag_assistant_service import rag_assistant_service

router = APIRouter(prefix="/chat", tags=["AI Knowledge Assistant (RAG)"])

@router.post("", response_model=ChatResponse)
async def chat_with_agri_assistant(req: ChatRequest):
    res = rag_assistant_service.answer_query(
        query=req.query,
        crop_context=req.crop_context,
        soil_context=req.soil_context,
        growth_stage_context=req.growth_stage_context,
        weather_context=req.weather_context,
        history=req.history
    )
    return ChatResponse(**res)
