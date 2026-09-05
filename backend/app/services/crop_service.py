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
        
        # Comprehensive Agronomic Profiles for all 22+ precision crops
        self.crop_profiles = {
            "Rice": {"season": "Kharif (Monsoon)", "water": "High (1200-1500 mm)", "duration": 120, "opt_ph": (5.5, 7.0), "opt_temp": (20, 35), "opt_soil": ["Clayey", "Alluvial", "Loamy", "Black"]},
            "Wheat": {"season": "Rabi (Winter)", "water": "Medium (450-650 mm)", "duration": 130, "opt_ph": (6.0, 7.5), "opt_temp": (15, 25), "opt_soil": ["Alluvial", "Loamy", "Clayey", "Black"]},
            "Barley": {"season": "Rabi (Winter)", "water": "Low-Medium (350-500 mm)", "duration": 115, "opt_ph": (6.0, 8.0), "opt_temp": (12, 28), "opt_soil": ["Sandy", "Loamy", "Alluvial", "Yellow"]},
            "Cotton": {"season": "Kharif (Monsoon)", "water": "Medium (600-800 mm)", "duration": 160, "opt_ph": (5.8, 8.0), "opt_temp": (21, 35), "opt_soil": ["Black", "Alluvial", "Red"]},
            "Maize": {"season": "Kharif / Rabi", "water": "Medium (500-750 mm)", "duration": 105, "opt_ph": (5.8, 7.2), "opt_temp": (18, 32), "opt_soil": ["Alluvial", "Loamy", "Red", "Sandy"]},
            "Sugarcane": {"season": "Perennial (Annual)", "water": "Very High (1500-2500 mm)", "duration": 360, "opt_ph": (6.0, 7.5), "opt_temp": (20, 35), "opt_soil": ["Loamy", "Alluvial", "Black"]},
            "Groundnuts": {"season": "Kharif / Zaid", "water": "Low-Medium (400-600 mm)", "duration": 110, "opt_ph": (5.5, 7.0), "opt_temp": (22, 30), "opt_soil": ["Sandy", "Red", "Loamy"]},
            "Millets": {"season": "Kharif", "water": "Low (300-450 mm)", "duration": 90, "opt_ph": (5.0, 7.5), "opt_temp": (25, 35), "opt_soil": ["Red", "Sandy", "Laterite", "Arid"]},
            "Pomegranate": {"season": "Perennial / All Seasons", "water": "Low-Medium (Drip)", "duration": 210, "opt_ph": (6.5, 7.5), "opt_temp": (20, 38), "opt_soil": ["Alluvial", "Sandy", "Black", "Red"]},
            "Pulses": {"season": "Rabi / Kharif", "water": "Low (350-500 mm)", "duration": 95, "opt_ph": (6.0, 7.5), "opt_temp": (18, 30), "opt_soil": ["Loamy", "Alluvial", "Black", "Red"]},
            "Chickpea": {"season": "Rabi (Winter)", "water": "Low (250-400 mm)", "duration": 100, "opt_ph": (6.0, 7.5), "opt_temp": (15, 25), "opt_soil": ["Loamy", "Black", "Alluvial"]},
            "Kidneybeans": {"season": "Rabi / Kharif", "water": "Medium (450-600 mm)", "duration": 90, "opt_ph": (5.5, 6.8), "opt_temp": (15, 24), "opt_soil": ["Loamy", "Alluvial", "Red"]},
            "Pigeonpeas": {"season": "Kharif", "water": "Low-Medium (500-650 mm)", "duration": 160, "opt_ph": (6.0, 7.5), "opt_temp": (20, 35), "opt_soil": ["Black", "Alluvial", "Red"]},
            "Mothbeans": {"season": "Kharif (Arid)", "water": "Very Low (200-350 mm)", "duration": 80, "opt_ph": (5.5, 7.5), "opt_temp": (25, 40), "opt_soil": ["Sandy", "Arid", "Red"]},
            "Mungbean": {"season": "Kharif / Zaid", "water": "Low (300-450 mm)", "duration": 75, "opt_ph": (6.2, 7.2), "opt_temp": (25, 35), "opt_soil": ["Loamy", "Alluvial", "Red"]},
            "Blackgram": {"season": "Kharif / Rabi", "water": "Low (350-500 mm)", "duration": 80, "opt_ph": (6.0, 7.5), "opt_temp": (25, 35), "opt_soil": ["Black", "Alluvial", "Loamy"]},
            "Lentil": {"season": "Rabi (Winter)", "water": "Low (250-350 mm)", "duration": 110, "opt_ph": (6.0, 7.5), "opt_temp": (15, 25), "opt_soil": ["Alluvial", "Loamy", "Clayey"]},
            "Coffee": {"season": "Perennial (Plantation)", "water": "High (1500-2200 mm)", "duration": 270, "opt_ph": (5.5, 6.5), "opt_temp": (18, 28), "opt_soil": ["Laterite", "Red", "Mountain"]},
            "Jute": {"season": "Kharif (Monsoon)", "water": "High (1200-1600 mm)", "duration": 125, "opt_ph": (6.0, 7.5), "opt_temp": (24, 37), "opt_soil": ["Alluvial", "Clayey", "Loamy"]},
            "Coconut": {"season": "Perennial", "water": "High (1300-2000 mm)", "duration": 365, "opt_ph": (5.2, 8.0), "opt_temp": (22, 34), "opt_soil": ["Alluvial", "Sandy", "Laterite", "Red"]},
            "Apple": {"season": "Temperate", "water": "Medium (800-1100 mm)", "duration": 180, "opt_ph": (5.5, 6.8), "opt_temp": (10, 24), "opt_soil": ["Mountain", "Loamy", "Alluvial"]},
            "Orange": {"season": "Sub-tropical", "water": "Medium (750-1000 mm)", "duration": 240, "opt_ph": (6.0, 7.5), "opt_temp": (15, 32), "opt_soil": ["Alluvial", "Black", "Red"]},
            "Papaya": {"season": "All Seasons", "water": "Medium (1000-1200 mm)", "duration": 270, "opt_ph": (6.0, 7.0), "opt_temp": (22, 35), "opt_soil": ["Alluvial", "Loamy", "Red"]},
            "Banana": {"season": "Perennial / Tropical", "water": "Very High (1500-2200 mm)", "duration": 330, "opt_ph": (6.0, 7.5), "opt_temp": (20, 35), "opt_soil": ["Alluvial", "Clayey", "Loamy", "Black"]},
            "Mango": {"season": "Summer / Tropical", "water": "Medium (750-1200 mm)", "duration": 150, "opt_ph": (5.5, 7.5), "opt_temp": (24, 38), "opt_soil": ["Alluvial", "Red", "Loamy"]},
            "Grapes": {"season": "Summer / Rabi", "water": "Medium (Drip)", "duration": 140, "opt_ph": (6.5, 7.8), "opt_temp": (15, 35), "opt_soil": ["Sandy", "Loamy", "Red", "Black"]},
            "Watermelon": {"season": "Zaid (Summer)", "water": "Low-Medium (400-600 mm)", "duration": 85, "opt_ph": (6.0, 7.0), "opt_temp": (24, 35), "opt_soil": ["Sandy", "Alluvial", "Loamy"]},
            "Muskmelon": {"season": "Zaid (Summer)", "water": "Low (350-500 mm)", "duration": 80, "opt_ph": (6.0, 7.0), "opt_temp": (24, 35), "opt_soil": ["Sandy", "Alluvial", "Loamy"]}
        }

    def load_model(self):
        if os.path.exists(self.bundle_path):
            try:
                self.bundle = joblib.load(self.bundle_path)
                print(f"[CropService] Loaded trained model: {self.bundle.get('best_model_name', 'Ensemble')}")
            except Exception as e:
                print(f"[CropService] Error loading model: {e}")

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
        
        # 1. Fetch live or customized weather dynamically
        weather = weather_service.get_weather(city=city, lat=lat, lon=lon)
        temperature = custom_temp if custom_temp is not None else weather["temperature"]
        humidity = custom_humidity if custom_humidity is not None else weather["humidity"]
        rainfall = custom_rainfall if custom_rainfall is not None else weather["rainfall"]

        soil_type = (soil.soil_type or "Alluvial").strip().title()

        crop_scores = {}

        # 2. Dynamic ML Model Prediction
        if self.bundle is not None:
            try:
                soil_classes = self.bundle.get("soil_classes", ["Alluvial", "Black", "Red", "Laterite", "Arid", "Mountain", "Yellow"])
                soil_type_query = soil_type.lower()
                
                soil_enc_val = 1
                for idx, sc in enumerate(soil_classes):
                    if sc.lower() in soil_type_query:
                        soil_enc_val = idx
                        break

                n_to_p = soil.nitrogen / (soil.phosphorus + 1e-4)
                n_to_k = soil.nitrogen / (soil.potassium + 1e-4)
                p_to_k = soil.phosphorus / (soil.potassium + 1e-4)
                npk_sum = soil.nitrogen + soil.phosphorus + soil.potassium
                
                feature_cols = self.bundle.get("feature_cols", [])
                if len(feature_cols) == 12:
                    features = [
                        soil.nitrogen, soil.phosphorus, soil.potassium, soil.ph,
                        temperature, humidity, rainfall, soil_enc_val,
                        n_to_p, n_to_k, p_to_k, npk_sum
                    ]
                else:
                    features = [
                        soil.nitrogen, soil.phosphorus, soil.potassium, soil.ph,
                        temperature, humidity, rainfall, soil_enc_val
                    ]
                
                df_feat = pd.DataFrame([features], columns=feature_cols if feature_cols else None)
                features_scaled = self.bundle["scaler"].transform(df_feat)
                probs = self.bundle["model"].predict_proba(features_scaled)[0]
                crop_classes = self.bundle["crop_classes"]
                
                for idx, p in enumerate(probs):
                    c_name = str(crop_classes[idx]).title()
                    crop_scores[c_name] = float(p)
            except Exception as e:
                print(f"[CropService] ML inference notice: {e}")

        # Ensure all crop profiles have baseline representation
        for c in self.crop_profiles.keys():
            if c not in crop_scores:
                crop_scores[c] = 0.01

        # 3. Dynamic Agronomic Hybrid Ranking
        recommendations: List[RecommendedCrop] = []
        for crop_name, base_prob in crop_scores.items():
            profile = self.crop_profiles.get(crop_name, {
                "season": "General",
                "water": "Moderate",
                "duration": 120,
                "opt_ph": (5.5, 7.5),
                "opt_temp": (18, 32),
                "opt_soil": ["Alluvial", "Black", "Red", "Loamy"]
            })

            # Calculate dynamic environmental & soil compatibility penalties/boosts
            ph_min, ph_max = profile["opt_ph"]
            ph_penalty = 0.0
            if soil.ph < ph_min:
                ph_penalty = (ph_min - soil.ph) * 0.12
            elif soil.ph > ph_max:
                ph_penalty = (soil.ph - ph_max) * 0.12

            t_min, t_max = profile.get("opt_temp", (15, 35))
            temp_penalty = 0.0
            if temperature < t_min:
                temp_penalty = (t_min - temperature) * 0.02
            elif temperature > t_max:
                temp_penalty = (temperature - t_max) * 0.02

            soil_match_boost = 0.12 if any(s.lower() in soil_type.lower() for s in profile["opt_soil"]) else -0.05

            # Dynamic combined score
            final_score = max(0.05, min(0.99, (base_prob * 0.65) + (soil_match_boost * 0.20) - (ph_penalty * 0.10) - (temp_penalty * 0.05) + 0.10))
            
            fert_info = fertilizer_service.recommend(
                crop=crop_name,
                soil_type=soil_type,
                n=soil.nitrogen,
                p=soil.phosphorus,
                k=soil.potassium,
                temp=temperature,
                humidity=humidity
            )

            recommendations.append(RecommendedCrop(
                crop_name=crop_name,
                confidence=round(float(base_prob), 4),
                suitability_score=round(float(final_score * 100), 1),
                recommended_season=profile["season"],
                soil_compatibility="Excellent" if soil_match_boost > 0 else "Moderate",
                water_requirement=profile["water"],
                growth_duration_days=profile["duration"],
                optimal_fertilizers=fert_info["recommended_fertilizers"]
            ))

        # Rank by combined confidence and suitability
        recommendations.sort(key=lambda x: (x.confidence, x.suitability_score), reverse=True)
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
