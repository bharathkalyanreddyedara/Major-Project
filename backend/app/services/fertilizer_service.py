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

    def recommend_fertilizers(self, req: Any) -> Dict[str, Any]:
        # Handle FertilizerRecommendationRequest or dict
        if hasattr(req, "soil_properties"):
            props = req.soil_properties
            crop = getattr(req, "crop_name", "Rice")
            n = getattr(props, "nitrogen", 90.0)
            p = getattr(props, "phosphorus", 42.0)
            k = getattr(props, "potassium", 43.0)
            ph = getattr(props, "ph", 6.5)
            soil_type = getattr(props, "soil_type", "Black")
            zinc = getattr(props, "zinc", 1.2)
            sulphur = getattr(props, "sulphur", 15.0)
            moist = getattr(props, "moisture", 40.0)
            temp = 26.0
            humid = 60.0
        elif isinstance(req, dict):
            crop = req.get("crop_name", "Rice")
            n = float(req.get("nitrogen", 90.0))
            p = float(req.get("phosphorus", 42.0))
            k = float(req.get("potassium", 43.0))
            ph = float(req.get("ph", 6.5))
            soil_type = req.get("soil_type", "Black")
            zinc = float(req.get("zinc", 1.2) or 1.2)
            sulphur = float(req.get("sulphur", 15.0) or 15.0)
            moist = float(req.get("moisture", 40.0) or 40.0)
            temp = float(req.get("temperature", 26.0) or 26.0)
            humid = float(req.get("humidity", 60.0) or 60.0)
        else:
            crop = getattr(req, "crop_name", "Rice")
            n = getattr(req, "nitrogen", 90.0)
            p = getattr(req, "phosphorus", 42.0)
            k = getattr(req, "potassium", 43.0)
            ph = getattr(req, "ph", 6.5)
            soil_type = getattr(req, "soil_type", "Black")
            zinc = getattr(req, "zinc", 1.2)
            sulphur = getattr(req, "sulphur", 15.0)
            moist = getattr(req, "moisture", 40.0)
            temp = getattr(req, "temperature", 26.0)
            humid = getattr(req, "humidity", 60.0)

        # Baseline recommendation
        base = self.recommend(
            crop=crop,
            soil_type=soil_type,
            n=n,
            p=p,
            k=k,
            temp=temp,
            humidity=humid,
            moisture=moist
        )

        target = self.crop_npk_targets.get(crop.strip().title(), {"N": 100, "P": 50, "K": 50})
        n_def = max(0.0, target["N"] - n)
        p_def = max(0.0, target["P"] - p)
        k_def = max(0.0, target["K"] - k)

        # 50kg bag calculations
        urea_bags = round((n_def / 0.46) / 50.0, 1) if n_def > 0 else 0.0
        dap_bags = round((p_def / 0.46) / 50.0, 1) if p_def > 0 else 0.0
        mop_bags = round((k_def / 0.60) / 50.0, 1) if k_def > 0 else 0.0
        ssp_bags = round((p_def / 0.16) / 50.0, 1) if p_def > 0 else 0.0

        micro_advice = {}
        if zinc is not None and zinc < 0.6:
            micro_advice["Zinc"] = f"Soil zinc ({zinc} ppm) is below critical threshold (0.6 ppm). Broadcast Zinc Sulphate (21% Zn) @ 25 kg/ha."
        if sulphur is not None and sulphur < 10.0:
            micro_advice["Sulphur"] = f"Soil sulphur ({sulphur} ppm) is deficient (<10 ppm). Apply Single Super Phosphate (SSP) or Gypsum @ 20 kg/ha."
        if ph < 5.5:
            micro_advice["Acidity (Lime)"] = f"Soil pH ({ph}) is acidic. Incorporate Agricultural Lime (CaCO3) @ 500-1000 kg/ha before sowing."
        elif ph > 8.5:
            micro_advice["Alkalinity (Gypsum)"] = f"Soil pH ({ph}) is alkaline. Apply agricultural Gypsum @ 1000 kg/ha with deep summer ploughing."

        return {
            "primary_fertilizer": base.get("primary_fertilizer", "Urea"),
            "recommended_fertilizers": base.get("recommended_fertilizers", []),
            "application_guidelines": base.get("application_tips", "Apply 50% N and full P & K as basal dose before sowing."),
            "dosage_kg_per_ha": {
                "nitrogen_deficit_kg_ha": int(n_def),
                "phosphorus_deficit_kg_ha": int(p_def),
                "potassium_deficit_kg_ha": int(k_def)
            },
            "commercial_bags_recommended": {
                "urea_50kg_bags": urea_bags,
                "dap_50kg_bags": dap_bags,
                "mop_50kg_bags": mop_bags,
                "ssp_50kg_bags": ssp_bags
            },
            "organic_alternatives": "Incorporate well-decomposed Farm Yard Manure (FYM) @ 10 tonnes/ha or Vermicompost @ 2.5 tonnes/ha with biofertilizers (Azospirillum & PSB).",
            "micronutrient_advice": micro_advice,
            "target_npk_ratio": base.get("target_npk_ratio", "120:60:60")
        }

fertilizer_service = FertilizerService()
