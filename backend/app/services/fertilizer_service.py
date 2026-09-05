import os
import joblib
import pandas as pd
from backend.app.config import settings
from typing import List, Dict, Any

class FertilizerService:
    def __init__(self):
        self.bundle_path = os.path.join(settings.MODELS_DIR, "fertilizer_recommender.joblib")
        self.bundle = None
        self.load_model()

    def load_model(self):
        if os.path.exists(self.bundle_path):
            try:
                self.bundle = joblib.load(self.bundle_path)
                print("Fertilizer model loaded.")
            except Exception as e:
                print(f"Error loading fertilizer model: {e}")

    def recommend(self, crop: str, soil_type: str, n: float, p: float, k: float, temp: float = 26.0, humidity: float = 60.0, moisture: float = 40.0) -> Dict[str, Any]:
        crop_clean = crop.strip().title()
        soil_clean = soil_type.strip().title()

        # Specific agronomic fertilizer mapping
        fertilizer_guide = {
            "Barley": ["Urea (Top-dressing)", "DAP (Basal)", "MOP (Potash)"],
            "Wheat": ["Urea", "DAP (Di-Ammonium Phosphate)", "NPK 12:32:16"],
            "Paddy": ["Urea (Split doses)", "SSP (Single Super Phosphate)", "MOP"],
            "Cotton": ["Urea", "NPK 20:20:0:13", "Potassium Nitrate"],
            "Maize": ["Urea", "DAP", "Zinc Sulphate"],
            "Sugarcane": ["Urea", "Single Super Phosphate", "Murate of Potash"],
            "Groundnuts": ["Gypsum (Calcium & Sulphur)", "SSP", "Bio-fertilizers (Rhizobium)"],
            "Tobacco": ["NPK 10:20:20", "Potassium Sulphate"],
            "Millets": ["Farmyard Manure (FYM)", "Urea", "DAP"],
            "Pulses": ["DAP", "Rhizobium Bio-fertilizer", "SSP"]
        }

        # Targeted dosage analysis based on NPK
        deficits = []
        if n < 40:
            deficits.append("Nitrogen deficit: Apply Urea / Calcium Ammonium Nitrate")
        if p < 25:
            deficits.append("Phosphorus deficit: Apply DAP / Single Super Phosphate (SSP)")
        if k < 120:
            deficits.append("Potassium deficit: Apply MOP (Muriate of Potash)")

        recs = fertilizer_guide.get(crop_clean, ["NPK 19:19:19 (Balanced)", "Urea", "DAP"])
        return {
            "recommended_fertilizers": recs,
            "soil_deficits": deficits,
            "application_tips": "Apply 50% nitrogen + full P & K as basal dose at sowing; split remaining nitrogen during vegetative and flowering stages."
        }

fertilizer_service = FertilizerService()
