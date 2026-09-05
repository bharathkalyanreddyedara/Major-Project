import os
import joblib
import numpy as np
import pandas as pd
from backend.app.config import settings
from backend.app.schemas.models import ManualSoilProperties, RecommendedCrop, CropRecommendationResponse
from backend.app.services.weather_service import weather_service
from backend.app.services.fertilizer_service import fertilizer_service
from typing import List, Dict, Any

class CropService:
    def __init__(self):
        self.bundle_path = os.path.join(settings.MODELS_DIR, "crop_recommender.joblib")
        self.bundle = None
        self.load_model()
        
        # Agronomic profile database for all major crops
        self.crop_profiles = {
            "Rice": {"season": "Kharif (Monsoon)", "water": "High (1200-1500 mm)", "duration": 120, "opt_ph": (5.5, 7.0), "opt_soil": ["Clayey", "Alluvial", "Loamy", "Black"]},
            "Wheat": {"season": "Rabi (Winter)", "water": "Medium (450-650 mm)", "duration": 130, "opt_ph": (6.0, 7.5), "opt_soil": ["Alluvial", "Loamy", "Clayey", "Black"]},
            "Barley": {"season": "Rabi (Winter)", "water": "Low-Medium (350-500 mm)", "duration": 115, "opt_ph": (6.0, 8.0), "opt_soil": ["Sandy", "Loamy", "Alluvial", "Yellow"]},
            "Cotton": {"season": "Kharif", "water": "Medium (600-800 mm)", "duration": 160, "opt_ph": (5.8, 8.0), "opt_soil": ["Black", "Alluvial", "Red"]},
            "Maize": {"season": "Kharif / Rabi", "water": "Medium (500-750 mm)", "duration": 105, "opt_ph": (5.8, 7.2), "opt_soil": ["Alluvial", "Loamy", "Red", "Sandy"]},
            "Sugarcane": {"season": "Perennial", "water": "Very High (1500-2500 mm)", "duration": 360, "opt_ph": (6.0, 7.5), "opt_soil": ["Loamy", "Alluvial", "Black"]},
            "Groundnuts": {"season": "Kharif / Zaid", "water": "Low-Medium (400-600 mm)", "duration": 110, "opt_ph": (5.5, 7.0), "opt_soil": ["Sandy", "Red", "Loamy"]},
            "Millets": {"season": "Kharif", "water": "Low (300-450 mm)", "duration": 90, "opt_ph": (5.0, 7.5), "opt_soil": ["Red", "Sandy", "Laterite", "Arid"]},
            "Pomegranate": {"season": "All Seasons", "water": "Low-Medium (drip)", "duration": 210, "opt_ph": (6.5, 7.5), "opt_soil": ["Alluvial", "Sandy", "Black", "Red"]},
            "Pulses": {"season": "Rabi / Kharif", "water": "Low (350-500 mm)", "duration": 95, "opt_ph": (6.0, 7.5), "opt_soil": ["Loamy", "Alluvial", "Black", "Red"]}
        }

    def load_model(self):
        if os.path.exists(self.bundle_path):
            try:
                self.bundle = joblib.load(self.bundle_path)
                print("Crop Recommendation ML model loaded.")
            except Exception as e:
                print(f"Error loading crop model: {e}")

    def recommend_crops(
        self,
        soil: ManualSoilProperties,
        city: str = "Hyderabad",
        lat: float = None,
        lon: float = None,
        custom_temp: float = None,
        custom_humidity: float = None,
        custom_rainfall: float = None
    ) -> CropRecommendationResponse:
        
        # 1. Fetch live or customized weather
        weather = weather_service.get_weather(city=city, lat=lat, lon=lon)
        temperature = custom_temp if custom_temp is not None else weather["temperature"]
        humidity = custom_humidity if custom_humidity is not None else weather["humidity"]
        rainfall = custom_rainfall if custom_rainfall is not None else weather["rainfall"]

        soil_type = (soil.soil_type or "Alluvial").strip().title()

        crop_scores = {}

        # 2. ML Model Scoring if available
        if self.bundle is not None:
            try:
                soil_classes = self.bundle["soil_classes"]
                soil_type_query = soil_type.lower()
                
                # Best match soil class
                soil_enc_val = 0
                if soil_type_query in soil_classes:
                    soil_enc_val = soil_classes.index(soil_type_query)
                
                features = [
                    soil.nitrogen,
                    soil.phosphorus,
                    soil.potassium,
                    soil.ph,
                    temperature,
                    humidity,
                    rainfall,
                    soil_enc_val
                ]
                
                features_scaled = self.bundle["scaler"].transform([features])
                probs = self.bundle["model"].predict_proba(features_scaled)[0]
                crop_classes = self.bundle["crop_classes"]
                
                for idx, p in enumerate(probs):
                    c_name = crop_classes[idx]
                    crop_scores[c_name] = float(p)
            except Exception as e:
                print(f"ML inference fallback: {e}")

        # If model didn't cover or fallback needed, ensure common agronomy crops
        for c in self.crop_profiles.keys():
            if c not in crop_scores:
                crop_scores[c] = 0.15

        # 3. Agronomic Hybrid Scoring (Penalize / Boost based on pH & Soil Type suitability)
        recommendations: List[RecommendedCrop] = []
        for crop_name, base_prob in crop_scores.items():
            profile = self.crop_profiles.get(crop_name, {
                "season": "General",
                "water": "Moderate",
                "duration": 120,
                "opt_ph": (5.5, 7.5),
                "opt_soil": ["Alluvial", "Black", "Red", "Loamy"]
            })

            # Calculate agronomic suitability
            ph_min, ph_max = profile["opt_ph"]
            ph_penalty = 0.0
            if soil.ph < ph_min:
                ph_penalty = (ph_min - soil.ph) * 0.1
            elif soil.ph > ph_max:
                ph_penalty = (soil.ph - ph_max) * 0.1

            soil_match_boost = 0.15 if any(s.lower() in soil_type.lower() for s in profile["opt_soil"]) else -0.1

            final_score = max(0.05, min(0.98, base_prob + soil_match_boost - ph_penalty))
            
            fert_info = fertilizer_service.recommend(
                crop=crop_name,
                soil_type=soil_type,
                n=soil.nitrogen,
                p=soil.phosphorus,
                k=soil.potassium
            )

            recommendations.append(RecommendedCrop(
                crop_name=crop_name,
                confidence=round(float(base_prob), 3),
                suitability_score=round(float(final_score * 100), 1),
                recommended_season=profile["season"],
                soil_compatibility="Excellent" if soil_match_boost > 0 else "Moderate",
                water_requirement=profile["water"],
                growth_duration_days=profile["duration"],
                optimal_fertilizers=fert_info["recommended_fertilizers"]
            ))

        # Sort by suitability score descending
        recommendations.sort(key=lambda x: x.suitability_score, reverse=True)
        top_recommendations = recommendations[:6]

        return CropRecommendationResponse(
            recommendations=top_recommendations,
            environmental_context={
                "location": weather["city"],
                "temperature": temperature,
                "humidity": humidity,
                "rainfall": rainfall,
                "weather_condition": weather["weather_condition"],
                "is_live_weather": weather.get("is_live", False),
                "provided_soil": {
                    "nitrogen": soil.nitrogen,
                    "phosphorus": soil.phosphorus,
                    "potassium": soil.potassium,
                    "ph": soil.ph,
                    "soil_type": soil_type
                }
            }
        )

crop_service = CropService()
