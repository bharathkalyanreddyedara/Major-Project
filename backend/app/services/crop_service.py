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
        
        # Authoritative Agro-Ecological Profiles (ICAR, TNAU, ICRISAT, FAO)
        self.crop_profiles = {
            "Cotton": {
                "season": "Kharif (Monsoon)",
                "water": "Medium (600-800 mm)",
                "duration": 160,
                "opt_ph": (5.8, 8.2),
                "opt_temp": (21, 36),
                "opt_soil": ["Black", "Vertisol", "Alluvial", "Red", "Loamy"],
                "disallowed_soils": ["Arid", "Laterite", "Mountain"],
                "regional_affinity": ["telangana", "hyderabad", "andhra", "maharashtra", "gujarat", "karnataka", "madhya pradesh", "deccan"]
            },
            "Maize": {
                "season": "Kharif / Rabi",
                "water": "Medium (500-750 mm)",
                "duration": 105,
                "opt_ph": (5.8, 7.8),
                "opt_temp": (18, 35),
                "opt_soil": ["Alluvial", "Loamy", "Black", "Red", "Sandy"],
                "disallowed_soils": ["Arid", "Laterite"],
                "regional_affinity": ["telangana", "hyderabad", "andhra", "karnataka", "bihar", "punjab", "rajasthan"]
            },
            "Pigeonpeas": {
                "season": "Kharif (Monsoon)",
                "water": "Low-Medium (500-650 mm)",
                "duration": 160,
                "opt_ph": (6.0, 8.0),
                "opt_temp": (20, 36),
                "opt_soil": ["Black", "Alluvial", "Red", "Loamy"],
                "disallowed_soils": ["Mountain", "Laterite"],
                "regional_affinity": ["telangana", "hyderabad", "maharashtra", "karnataka", "madhya pradesh", "andhra"]
            },
            "Chickpea": {
                "season": "Rabi (Winter)",
                "water": "Low (250-400 mm)",
                "duration": 100,
                "opt_ph": (6.0, 8.0),
                "opt_temp": (15, 30),
                "opt_soil": ["Black", "Loamy", "Alluvial", "Clayey"],
                "disallowed_soils": ["Laterite", "Mountain", "Arid"],
                "regional_affinity": ["telangana", "hyderabad", "madhya pradesh", "maharashtra", "rajasthan", "uttar pradesh"]
            },
            "Soybean": {
                "season": "Kharif (Monsoon)",
                "water": "Medium (500-750 mm)",
                "duration": 105,
                "opt_ph": (6.0, 7.8),
                "opt_temp": (20, 34),
                "opt_soil": ["Black", "Alluvial", "Loamy"],
                "disallowed_soils": ["Sandy", "Arid", "Laterite"],
                "regional_affinity": ["madhya pradesh", "maharashtra", "telangana", "rajasthan", "karnataka"]
            },
            "Groundnut": {
                "season": "Kharif / Zaid",
                "water": "Low-Medium (400-600 mm)",
                "duration": 110,
                "opt_ph": (5.5, 7.5),
                "opt_temp": (22, 34),
                "opt_soil": ["Sandy", "Red", "Loamy", "Black"],
                "disallowed_soils": ["Heavy Clay", "Mountain"],
                "regional_affinity": ["gujarat", "andhra", "telangana", "tamil nadu", "rajasthan", "karnataka"]
            },
            "Groundnuts": {
                "season": "Kharif / Zaid",
                "water": "Low-Medium (400-600 mm)",
                "duration": 110,
                "opt_ph": (5.5, 7.5),
                "opt_temp": (22, 34),
                "opt_soil": ["Sandy", "Red", "Loamy", "Black"],
                "disallowed_soils": ["Heavy Clay", "Mountain"],
                "regional_affinity": ["gujarat", "andhra", "telangana", "tamil nadu", "rajasthan", "karnataka"]
            },
            "Rice": {
                "season": "Kharif (Monsoon)",
                "water": "High (1200-1600 mm)",
                "duration": 120,
                "opt_ph": (5.5, 7.5),
                "opt_temp": (20, 36),
                "opt_soil": ["Clayey", "Alluvial", "Loamy", "Black"],
                "disallowed_soils": ["Sandy", "Arid"],
                "regional_affinity": ["andhra", "telangana", "west bengal", "punjab", "tamil nadu", "odisha", "bihar"]
            },
            "Wheat": {
                "season": "Rabi (Winter)",
                "water": "Medium (450-650 mm)",
                "duration": 130,
                "opt_ph": (6.0, 7.8),
                "opt_temp": (12, 26),
                "opt_soil": ["Alluvial", "Loamy", "Clayey", "Black"],
                "disallowed_soils": ["Laterite", "Arid"],
                "regional_affinity": ["punjab", "haryana", "uttar pradesh", "madhya pradesh", "rajasthan", "bihar"]
            },
            "Tomato": {
                "season": "Rabi / Kharif / Summer",
                "water": "Medium (500-700 mm)",
                "duration": 115,
                "opt_ph": (6.0, 7.2),
                "opt_temp": (18, 32),
                "opt_soil": ["Loamy", "Alluvial", "Red", "Black"],
                "disallowed_soils": ["Arid", "Waterlogged"],
                "regional_affinity": ["andhra", "telangana", "karnataka", "maharashtra", "madhya pradesh"]
            },
            "Potato": {
                "season": "Rabi (Winter)",
                "water": "Medium (450-600 mm)",
                "duration": 100,
                "opt_ph": (5.2, 6.8),
                "opt_temp": (14, 24),
                "opt_soil": ["Sandy", "Loamy", "Alluvial"],
                "disallowed_soils": ["Heavy Black Clay", "Laterite"],
                "regional_affinity": ["uttar pradesh", "west bengal", "bihar", "gujarat", "punjab"]
            },
            "Sorghum": {
                "season": "Kharif / Rabi",
                "water": "Low (400-600 mm)",
                "duration": 110,
                "opt_ph": (6.0, 8.5),
                "opt_temp": (24, 38),
                "opt_soil": ["Black", "Alluvial", "Loamy", "Red"],
                "disallowed_soils": ["Waterlogged"],
                "regional_affinity": ["maharashtra", "karnataka", "telangana", "rajasthan", "andhra"]
            },
            "Millets": {
                "season": "Kharif",
                "water": "Low (300-450 mm)",
                "duration": 90,
                "opt_ph": (5.0, 8.0),
                "opt_temp": (24, 38),
                "opt_soil": ["Red", "Sandy", "Laterite", "Arid", "Black"],
                "disallowed_soils": ["Waterlogged Clay"],
                "regional_affinity": ["rajasthan", "karnataka", "maharashtra", "tamil nadu", "telangana", "andhra"]
            },
            "Mustard": {
                "season": "Rabi (Winter)",
                "water": "Low (250-400 mm)",
                "duration": 110,
                "opt_ph": (6.0, 7.8),
                "opt_temp": (12, 26),
                "opt_soil": ["Alluvial", "Loamy", "Sandy", "Clayey"],
                "disallowed_soils": ["Laterite", "Acidic"],
                "regional_affinity": ["rajasthan", "haryana", "madhya pradesh", "uttar pradesh", "west bengal"]
            },
            "Sugarcane": {
                "season": "Perennial (Annual)",
                "water": "Very High (1500-2500 mm)",
                "duration": 360,
                "opt_ph": (6.0, 8.0),
                "opt_temp": (20, 36),
                "opt_soil": ["Loamy", "Alluvial", "Black"],
                "disallowed_soils": ["Arid", "Shallow Red"],
                "regional_affinity": ["uttar pradesh", "maharashtra", "karnataka", "tamil nadu", "andhra", "telangana"]
            },
            "Pomegranate": {
                "season": "Perennial / All Seasons",
                "water": "Low-Medium (Drip)",
                "duration": 210,
                "opt_ph": (6.5, 7.8),
                "opt_temp": (20, 38),
                "opt_soil": ["Alluvial", "Sandy", "Black", "Red"],
                "disallowed_soils": ["Heavy Waterlogged"],
                "regional_affinity": ["maharashtra", "karnataka", "gujarat", "andhra", "telangana"]
            },
            "Watermelon": {
                "season": "Zaid (Summer)",
                "water": "Low-Medium (400-600 mm)",
                "duration": 85,
                "opt_ph": (6.0, 7.5),
                "opt_temp": (24, 38),
                "opt_soil": ["Sandy", "Alluvial", "Loamy", "Black"],
                "disallowed_soils": ["Waterlogged"],
                "regional_affinity": ["andhra", "telangana", "tamil nadu", "karnataka", "uttar pradesh"]
            },
            "Muskmelon": {
                "season": "Zaid (Summer)",
                "water": "Low (350-500 mm)",
                "duration": 80,
                "opt_ph": (6.0, 7.5),
                "opt_temp": (24, 38),
                "opt_soil": ["Sandy", "Alluvial", "Loamy", "Black"],
                "disallowed_soils": ["Waterlogged"],
                "regional_affinity": ["uttar pradesh", "punjab", "andhra", "telangana", "rajasthan"]
            },
            "Blackgram": {
                "season": "Kharif / Rabi",
                "water": "Low (350-500 mm)",
                "duration": 80,
                "opt_ph": (6.0, 7.8),
                "opt_temp": (25, 36),
                "opt_soil": ["Black", "Alluvial", "Loamy"],
                "disallowed_soils": ["Laterite", "Acidic"],
                "regional_affinity": ["andhra", "telangana", "madhya pradesh", "tamil nadu", "maharashtra"]
            },
            "Mungbean": {
                "season": "Kharif / Zaid",
                "water": "Low (300-450 mm)",
                "duration": 75,
                "opt_ph": (6.2, 7.6),
                "opt_temp": (25, 36),
                "opt_soil": ["Loamy", "Alluvial", "Red", "Black"],
                "disallowed_soils": ["Waterlogged"],
                "regional_affinity": ["rajasthan", "maharashtra", "andhra", "telangana", "karnataka"]
            },
            "Mothbeans": {
                "season": "Kharif (Arid)",
                "water": "Very Low (200-350 mm)",
                "duration": 80,
                "opt_ph": (5.5, 8.0),
                "opt_temp": (26, 42),
                "opt_soil": ["Sandy", "Arid", "Red"],
                "disallowed_soils": ["Heavy Clay", "Black", "Waterlogged"],
                "regional_affinity": ["rajasthan", "gujarat", "haryana"]
            },
            "Kidneybeans": {
                "season": "Rabi / Kharif (Hills/Plains)",
                "water": "Medium (450-600 mm)",
                "duration": 90,
                "opt_ph": (5.5, 6.8),
                "opt_temp": (15, 25),
                "opt_soil": ["Loamy", "Alluvial", "Red"],
                "disallowed_soils": ["Alkaline", "Black Heavy Clay"],
                "regional_affinity": ["jammu", "himachal", "uttarakhand", "maharashtra"]
            },
            "Lentil": {
                "season": "Rabi (Winter)",
                "water": "Low (250-350 mm)",
                "duration": 110,
                "opt_ph": (6.0, 7.8),
                "opt_temp": (12, 26),
                "opt_soil": ["Alluvial", "Loamy", "Clayey"],
                "disallowed_soils": ["Laterite", "Acidic"],
                "regional_affinity": ["madhya pradesh", "uttar pradesh", "bihar", "west bengal"]
            },
            "Banana": {
                "season": "Perennial / Tropical",
                "water": "Very High (1500-2200 mm)",
                "duration": 330,
                "opt_ph": (6.0, 7.8),
                "opt_temp": (20, 36),
                "opt_soil": ["Alluvial", "Clayey", "Loamy", "Black"],
                "disallowed_soils": ["Sandy", "Arid"],
                "regional_affinity": ["tamil nadu", "maharashtra", "gujarat", "andhra", "telangana", "kerala"]
            },
            "Mango": {
                "season": "Summer / Tropical",
                "water": "Medium (750-1200 mm)",
                "duration": 150,
                "opt_ph": (5.5, 7.5),
                "opt_temp": (22, 38),
                "opt_soil": ["Alluvial", "Red", "Loamy", "Black"],
                "disallowed_soils": ["Saline", "Waterlogged"],
                "regional_affinity": ["uttar pradesh", "andhra", "telangana", "karnataka", "bihar", "gujarat", "maharashtra"]
            },
            "Grapes": {
                "season": "Summer / Rabi",
                "water": "Medium (Drip)",
                "duration": 140,
                "opt_ph": (6.5, 7.8),
                "opt_temp": (15, 36),
                "opt_soil": ["Sandy", "Loamy", "Red", "Black"],
                "disallowed_soils": ["Heavy Waterlogged"],
                "regional_affinity": ["maharashtra", "karnataka", "tamil nadu", "andhra", "telangana"]
            },
            "Papaya": {
                "season": "All Seasons",
                "water": "Medium (1000-1200 mm)",
                "duration": 270,
                "opt_ph": (6.0, 7.2),
                "opt_temp": (22, 36),
                "opt_soil": ["Alluvial", "Loamy", "Red"],
                "disallowed_soils": ["Heavy Waterlogged Black"],
                "regional_affinity": ["andhra", "telangana", "gujarat", "karnataka", "maharashtra"]
            },
            "Orange": {
                "season": "Sub-tropical",
                "water": "Medium (750-1000 mm)",
                "duration": 240,
                "opt_ph": (6.0, 7.8),
                "opt_temp": (15, 34),
                "opt_soil": ["Alluvial", "Black", "Red"],
                "disallowed_soils": ["Saline", "High Water Table"],
                "regional_affinity": ["maharashtra", "madhya pradesh", "andhra", "telangana", "punjab", "assam"]
            },
            "Barley": {
                "season": "Rabi (Winter)",
                "water": "Low-Medium (350-500 mm)",
                "duration": 115,
                "opt_ph": (6.0, 8.2),
                "opt_temp": (12, 28),
                "opt_soil": ["Sandy", "Loamy", "Alluvial", "Yellow"],
                "disallowed_soils": ["Heavy Clay", "Acidic"],
                "regional_affinity": ["rajasthan", "uttar pradesh", "haryana", "punjab", "madhya pradesh"]
            },
            "Coffee": {
                "season": "Perennial (High Altitude Plantation)",
                "water": "High (1500-2200 mm)",
                "duration": 270,
                "opt_ph": (5.0, 6.4),
                "opt_temp": (16, 28),
                "opt_soil": ["Laterite", "Mountain", "Forest Loam"],
                "disallowed_soils": ["Black", "Alluvial", "Arid", "Sandy"],
                "regional_affinity": ["coorg", "chikmagalur", "wayanad", "nilgiri", "kodagu", "hassan", "munnar", "araku", "kerala", "western ghats"]
            },
            "Apple": {
                "season": "Temperate (Chilling Units Required)",
                "water": "Medium (800-1100 mm)",
                "duration": 180,
                "opt_ph": (5.5, 6.8),
                "opt_temp": (8, 24),
                "opt_soil": ["Mountain", "Loamy", "Gravelly"],
                "disallowed_soils": ["Black", "Alluvial", "Arid", "Red", "Laterite"],
                "regional_affinity": ["kashmir", "shimla", "kullu", "himachal", "uttarakhand", "ladakh", "jammu"]
            },
            "Jute": {
                "season": "Kharif (Humid Floodplains)",
                "water": "High (1200-1600 mm)",
                "duration": 125,
                "opt_ph": (6.0, 7.5),
                "opt_temp": (24, 37),
                "opt_soil": ["Alluvial", "Clayey", "Loamy"],
                "disallowed_soils": ["Arid", "Black", "Sandy", "Mountain"],
                "regional_affinity": ["west bengal", "bengal", "assam", "bihar", "odisha", "meghalaya"]
            },
            "Coconut": {
                "season": "Perennial (Coastal / Tropical)",
                "water": "High (1300-2000 mm)",
                "duration": 365,
                "opt_ph": (5.2, 8.0),
                "opt_temp": (22, 34),
                "opt_soil": ["Alluvial", "Sandy", "Laterite", "Red"],
                "disallowed_soils": ["Arid", "Mountain"],
                "regional_affinity": ["kerala", "tamil nadu", "karnataka", "andhra", "goa", "odisha", "coastal"]
            }
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
        soil: Any,
        city: str = "Hyderabad",
        lat: float = None,
        lon: float = None,
        custom_temp: float = None,
        custom_humidity: float = None,
        custom_rainfall: float = None
    ) -> Dict[str, Any]:
        
        # Handle CropRecommendationRequest
        if hasattr(soil, "soil_properties"):
            req_obj = soil
            soil = req_obj.soil_properties
            city = getattr(req_obj, "city", city) or city
            lat = getattr(req_obj, "latitude", lat)
            lon = getattr(req_obj, "longitude", lon)
            custom_temp = getattr(req_obj, "temperature", custom_temp)
            custom_humidity = getattr(req_obj, "humidity", custom_humidity)
            custom_rainfall = getattr(req_obj, "rainfall", custom_rainfall)

        # 1. Fetch live or customized weather dynamically
        weather = weather_service.get_weather(city=city, lat=lat, lon=lon)
        temperature = custom_temp if custom_temp is not None else weather["temperature"]
        humidity = custom_humidity if custom_humidity is not None else weather["humidity"]
        daily_rainfall = custom_rainfall if custom_rainfall is not None else weather["rainfall"]

        soil_type = (getattr(soil, "soil_type", None) or "Alluvial").strip().title()
        city_lower = (weather.get("city", city) or city).lower()

        # 2. Convert rainfall to seasonal crop model scale (cm / 10mm units matching training distribution)
        # If daily rainfall is passed (< 50mm), estimate growing season moisture availability (typically 70-150 cm)
        if daily_rainfall is not None and daily_rainfall > 50.0:
            seasonal_rain_feature = min(300.0, float(daily_rainfall) / 10.0)
        else:
            # Regional seasonal normal defaults
            if any(w in city_lower for w in ["mumbai", "kerala", "goa", "konkan", "assam", "bengal"]):
                seasonal_rain_feature = 200.0 # High rainfall belt (2000 mm)
            elif any(w in city_lower for w in ["rajasthan", "jaipur", "jodhpur", "bikaner", "kutch", "arid"]):
                seasonal_rain_feature = 35.0 # Arid zone (350 mm)
            elif any(w in city_lower for w in ["punjab", "haryana", "delhi"]):
                seasonal_rain_feature = 70.0 # North plains (700 mm)
            else:
                seasonal_rain_feature = 85.0 # Deccan / South-Central semi-arid (850 mm)

        crop_scores = {}

        # 3. Dynamic ML Model Prediction
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
                        temperature, humidity, seasonal_rain_feature, soil_enc_val,
                        n_to_p, n_to_k, p_to_k, npk_sum
                    ]
                else:
                    features = [
                        soil.nitrogen, soil.phosphorus, soil.potassium, soil.ph,
                        temperature, humidity, seasonal_rain_feature, soil_enc_val
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
                crop_scores[c] = 0.02

        # 4. Authoritative Agro-Ecological Bioclimatic Suitability Engine
        recommendations: List[RecommendedCrop] = []
        for crop_name, base_prob in crop_scores.items():
            profile = self.crop_profiles.get(crop_name)
            if not profile:
                continue

            # Hard Constraint 1: Disallowed Soil Types
            disallowed = [s.lower() for s in profile.get("disallowed_soils", [])]
            if any(d in soil_type.lower() for d in disallowed):
                continue # Strictly incompatible soil order

            # Hard Constraint 2: Physiological Temperature Envelope
            t_min, t_max = profile["opt_temp"]
            if temperature < (t_min - 6) or temperature > (t_max + 6):
                continue # Ambient temperature outside survival threshold

            # Hard Constraint 3: Specific Regional Ecology (Plantation / Temperate isolation)
            if crop_name == "Coffee":
                is_hill_zone = any(w in city_lower for w in profile["regional_affinity"])
                if not is_hill_zone or soil_type.lower() in ["black", "alluvial", "arid"] or soil.ph > 6.8:
                    continue # Coffee cannot grow in semi-arid plains / alkaline vertisols

            if crop_name == "Apple":
                is_temperate_zone = any(w in city_lower for w in profile["regional_affinity"])
                if not is_temperate_zone or temperature > 25.0 or soil_type.lower() != "mountain":
                    continue # Apple strictly requires high chilling temperate mountain ecology

            if crop_name == "Jute":
                is_jute_zone = any(w in city_lower for w in profile["regional_affinity"])
                if not is_jute_zone and soil_type.lower() not in ["alluvial", "clayey", "loamy"]:
                    continue

            # Calculate Agronomic Compatibility Boosts & Penalties
            # A. Soil Order Match
            opt_soils = [s.lower() for s in profile.get("opt_soil", [])]
            is_opt_soil = any(s in soil_type.lower() for s in opt_soils)
            soil_score = 0.35 if is_opt_soil else 0.05

            # B. pH Compatibility
            ph_min, ph_max = profile["opt_ph"]
            if ph_min <= soil.ph <= ph_max:
                ph_score = 0.20
            else:
                ph_diff = min(abs(soil.ph - ph_min), abs(soil.ph - ph_max))
                ph_score = max(0.0, 0.20 - (ph_diff * 0.10))

            # C. Thermal Compatibility
            if t_min <= temperature <= t_max:
                temp_score = 0.20
            else:
                temp_diff = min(abs(temperature - t_min), abs(temperature - t_max))
                temp_score = max(0.0, 0.20 - (temp_diff * 0.03))

            # D. Regional & Soil Affinity Multiplier (ICAR Agro-Climatic Zone)
            regional_match = any(r in city_lower for r in profile.get("regional_affinity", []))
            region_score = 0.15 if regional_match else 0.05

            # E. Black Soil (Vertisol) Prime Crop Specialization
            if soil_type.lower() == "black" and crop_name in ["Cotton", "Maize", "Pigeonpeas", "Soybean", "Chickpea", "Sorghum", "Tomato", "Pomegranate"]:
                region_score += 0.10

            # Combined Suitability Score
            combined_suitability = (base_prob * 0.25) + soil_score + ph_score + temp_score + region_score
            final_suitability = min(0.98, max(0.20, combined_suitability))

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
                confidence=round(float(final_suitability), 3),
                suitability_score=round(float(final_suitability * 100), 1),
                recommended_season=profile["season"],
                soil_compatibility="Excellent" if is_opt_soil else "Moderate",
                water_requirement=profile["water"],
                growth_duration_days=profile["duration"],
                optimal_fertilizers=fert_info["recommended_fertilizers"]
            ))

        # Rank by combined suitability score
        recommendations.sort(key=lambda x: (x.suitability_score, x.confidence), reverse=True)
        top_recommendations = [
            {
                "crop_name": r.crop_name,
                "confidence": r.confidence,
                "suitability_score": r.suitability_score,
                "recommended_season": r.recommended_season,
                "soil_compatibility": r.soil_compatibility,
                "water_requirement": r.water_requirement,
                "growth_duration_days": r.growth_duration_days,
                "optimal_fertilizers": r.optimal_fertilizers
            }
            for r in recommendations[:6]
        ]

        return {
            "recommendations": top_recommendations,
            "environmental_context": {
                "location": weather["city"],
                "temperature": temperature,
                "humidity": humidity,
                "rainfall": daily_rainfall,
                "seasonal_rainfall_est_mm": round(seasonal_rain_feature * 10, 1),
                "weather_condition": weather["weather_condition"],
                "is_live_weather": weather.get("is_live", False),
                "provided_soil": {
                    "nitrogen": getattr(soil, "nitrogen", 0),
                    "phosphorus": getattr(soil, "phosphorus", 0),
                    "potassium": getattr(soil, "potassium", 0),
                    "ph": getattr(soil, "ph", 7.0),
                    "soil_type": soil_type
                }
            }
        }

crop_service = CropService()
