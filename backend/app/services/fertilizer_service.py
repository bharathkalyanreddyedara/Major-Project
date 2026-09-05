import os
import joblib
import pandas as pd
import numpy as np
from backend.app.config import settings
from typing import List, Dict, Any

class FertilizerService:
    def __init__(self):
        self.bundle_path = os.path.join(settings.MODELS_DIR, "fertilizer_recommender.joblib")
        self.bundle = None
        self.load_model()

        # Optimal NPK targets (in kg/ha) per crop from ICAR agricultural guidelines
        self.crop_npk_targets = {
            "Rice": {"N": 120, "P": 60, "K": 60},
            "Wheat": {"N": 120, "P": 60, "K": 40},
            "Barley": {"N": 80, "P": 40, "K": 30},
            "Cotton": {"N": 150, "P": 60, "K": 60},
            "Maize": {"N": 120, "P": 60, "K": 50},
            "Sugarcane": {"N": 250, "P": 100, "K": 120},
            "Groundnuts": {"N": 25, "P": 50, "K": 40},
            "Millets": {"N": 60, "P": 30, "K": 30},
            "Pomegranate": {"N": 200, "P": 100, "K": 150},
            "Pulses": {"N": 20, "P": 40, "K": 20},
            "Chickpea": {"N": 25, "P": 50, "K": 25},
            "Coffee": {"N": 140, "P": 90, "K": 120},
            "Jute": {"N": 80, "P": 40, "K": 40},
            "Coconut": {"N": 500, "P": 320, "K": 1200},
            "Apple": {"N": 70, "P": 35, "K": 70},
            "Mango": {"N": 100, "P": 50, "K": 100}
        }

    def load_model(self):
        if os.path.exists(self.bundle_path):
            try:
                self.bundle = joblib.load(self.bundle_path)
                print(f"[FertilizerService] Loaded model: {self.bundle.get('best_model_name', 'Trained Classifier')}")
            except Exception as e:
                print(f"[FertilizerService] Notice loading model: {e}")

    def recommend(self, crop: str, soil_type: str, n: float, p: float, k: float, temp: float = 26.0, humidity: float = 60.0, moisture: float = 40.0) -> Dict[str, Any]:
        crop_clean = crop.strip().title()
        soil_clean = soil_type.strip().title()

        predicted_fertilizer = None
        probs_dict = {}

        # 1. Dynamic ML Model Prediction
        if self.bundle is not None:
            try:
                soil_encoder = self.bundle["soil_encoder"]
                crop_encoder = self.bundle["crop_encoder"]
                target_encoder = self.bundle["target_encoder"]
                scaler = self.bundle["scaler"]
                model = self.bundle["model"]
                feature_cols = self.bundle["feature_cols"]

                # Safe category matching
                soil_classes = [c.lower() for c in soil_encoder.classes_]
                soil_enc = 0
                for idx, s in enumerate(soil_classes):
                    if s in soil_clean.lower():
                        soil_enc = idx
                        break

                crop_classes = [c.lower() for c in crop_encoder.classes_]
                crop_enc = 0
                for idx, c in enumerate(crop_classes):
                    if c in crop_clean.lower():
                        crop_enc = idx
                        break

                n_to_p = n / (p + 1e-4)
                n_to_k = n / (k + 1e-4)
                p_to_k = p / (k + 1e-4)
                npk_total = n + p + k

                if len(feature_cols) == 12:
                    raw_feats = [temp, humidity, moisture, soil_enc, crop_enc, n, k, p, n_to_p, n_to_k, p_to_k, npk_total]
                else:
                    raw_feats = [temp, humidity, moisture, soil_enc, crop_enc, n, k, p]

                df_f = pd.DataFrame([raw_feats], columns=feature_cols)
                scaled_f = scaler.transform(df_f)
                probs = model.predict_proba(scaled_f)[0]
                top_idx = int(np.argmax(probs))
                predicted_fertilizer = str(target_encoder.classes_[top_idx])

                for idx, prob in enumerate(probs):
                    probs_dict[str(target_encoder.classes_[idx])] = round(float(prob), 3)
            except Exception as e:
                print(f"[FertilizerService] ML inference note: {e}")

        # 2. Dynamic Agronomic Nutrient Deficit Calculations
        target = self.crop_npk_targets.get(crop_clean, {"N": 100, "P": 50, "K": 50})
        n_deficit = max(0.0, target["N"] - n)
        p_deficit = max(0.0, target["P"] - p)
        k_deficit = max(0.0, target["K"] - k)

        deficits = []
        if n_deficit > 20:
            deficits.append(f"Nitrogen deficit: ~{int(n_deficit)} kg/ha required. Apply Urea (46% N) in 2-3 split doses.")
        if p_deficit > 15:
            deficits.append(f"Phosphorus deficit: ~{int(p_deficit)} kg/ha required. Apply DAP (18:46:0) or SSP as basal dose.")
        if k_deficit > 20:
            deficits.append(f"Potassium deficit: ~{int(k_deficit)} kg/ha required. Apply MOP (Muriate of Potash 60% K2O).")

        if not deficits:
            deficits.append("Soil macronutrients are in optimal range for the target crop. Maintain organic compost.")

        # Recommendations list
        recs = []
        if predicted_fertilizer:
            recs.append(f"{predicted_fertilizer} (ML Primary Match)")

        if p_deficit > 20 and "DAP" not in [r.split()[0] for r in recs]:
            recs.append("DAP (Di-Ammonium Phosphate)")
        if n_deficit > 20 and "Urea" not in [r.split()[0] for r in recs]:
            recs.append("Urea (Top-dressing)")
        if k_deficit > 20:
            recs.append("MOP (Potash)")

        if not recs:
            recs = ["NPK 19:19:19 (Balanced Foliar)", "Organic Vermicompost", "Urea"]

        return {
            "recommended_fertilizers": recs[:4],
            "primary_fertilizer": predicted_fertilizer or recs[0],
            "soil_deficits": deficits,
            "target_npk_ratio": f"{target['N']}:{target['P']}:{target['K']}",
            "probabilities": probs_dict,
            "application_tips": "Apply 50% Nitrogen and 100% Phosphorus & Potassium as basal dose before sowing. Apply remaining Nitrogen in split doses at tillering and panicle/flowering stages."
        }

fertilizer_service = FertilizerService()
