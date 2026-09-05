import os
from pydantic_settings import BaseSettings

class Settings(BaseSettings):
    PROJECT_NAME: str = "Multimodal Agri-AI Decision System"
    VERSION: str = "1.0.0"
    API_V1_STR: str = "/api"
    
    # AI & API keys
    GEMINI_API_KEY: str = os.getenv("GEMINI_API_KEY", "")
    OPENWEATHER_API_KEY: str = os.getenv("OPENWEATHER_API_KEY", "")
    
    # Paths
    BASE_DIR: str = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    MODELS_DIR: str = os.path.join(BASE_DIR, "models")
    DATA_DIR: str = os.path.join(BASE_DIR, "data")
    KNOWLEDGE_DIR: str = os.path.join(DATA_DIR, "knowledge")
    
    # Database
    DATABASE_URL: str = "sqlite:///./agri_assistant.db"

    class Config:
        env_file = ".env"

settings = Settings()
