from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from backend.app.config import settings
from backend.app.api.soil_routes import router as soil_router
from backend.app.api.crop_routes import router as crop_router
from backend.app.api.timeline_routes import router as timeline_router
from backend.app.api.notification_routes import router as notification_router
from backend.app.api.chat_routes import router as chat_router
from backend.app.api.weather_routes import router as weather_router

app = FastAPI(
    title=settings.PROJECT_NAME,
    version=settings.VERSION,
    description="Multimodal Generative AI System for Adaptive Crop Planning and Proactive Farm Intelligence"
)

# Enable CORS for Frontend React integration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include Routers
app.include_router(soil_router, prefix="/api")
app.include_router(crop_router, prefix="/api")
app.include_router(timeline_router, prefix="/api")
app.include_router(notification_router, prefix="/api")
app.include_router(chat_router, prefix="/api")
app.include_router(weather_router, prefix="/api")

@app.get("/")
def root():
    return {
        "status": "healthy",
        "system": settings.PROJECT_NAME,
        "version": settings.VERSION,
        "endpoints": [
            "/api/soil/analyze",
            "/api/crop/recommend",
            "/api/timeline/generate",
            "/api/notifications",
            "/api/chat",
            "/api/weather"
        ]
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("backend.app.main:app", host="0.0.0.0", port=8000, reload=True)
