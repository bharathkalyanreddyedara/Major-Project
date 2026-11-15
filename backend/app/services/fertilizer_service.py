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

        # Crop targets: N (kg/ha), available P expressed as P2O5 (kg/ha), and available K expressed as K2O (kg/ha). Verify crop-specific values against local recommendations before field use.
        self.crop_npk_targets = {
            "Rice": {"N": 120, "P2O5": 60, "K2O": 60},
            "Wheat": {"N": 120, "P2O5": 60, "K2O": 40},
            "Barley": {"N": 80, "P2O5": 40, "K2O": 30},
            "Cotton": {"N": 150, "P2O5": 60, "K2O": 60},
            "Maize": {"N": 120, "P2O5": 60, "K2O": 50},
            "Sugarcane": {"N": 250, "P2O5": 100, "K2O": 120},
            "Groundnut": {"N": 25, "P2O5": 50, "K2O": 40},
            "Groundnuts": {"N": 25, "P2O5": 50, "K2O": 40},
            "Millets": {"N": 60, "P2O5": 30, "K2O": 30},
            "Sorghum": {"N": 80, "P2O5": 40, "K2O": 40},
            "Pomegranate": {"N": 200, "P2O5": 100, "K2O": 150},
            "Pulses": {"N": 20, "P2O5": 40, "K2O": 20},
            "Chickpea": {"N": 25, "P2O5": 50, "K2O": 25},
            "Kidneybeans": {"N": 30, "P2O5": 60, "K2O": 30},
            "Pigeonpeas": {"N": 25, "P2O5": 50, "K2O": 25},
            "Mothbeans": {"N": 20, "P2O5": 40, "K2O": 20},
            "Mungbean": {"N": 20, "P2O5": 40, "K2O": 20},
            "Blackgram": {"N": 20, "P2O5": 40, "K2O": 20},
            "Lentil": {"N": 20, "P2O5": 40, "K2O": 20},
            "Coffee": {"N": 140, "P2O5": 90, "K2O": 120},
            "Jute": {"N": 80, "P2O5": 40, "K2O": 40},
            "Coconut": {"N": 500, "P2O5": 320, "K2O": 1200},
            "Apple": {"N": 70, "P2O5": 35, "K2O": 70},
            "Orange": {"N": 120, "P2O5": 60, "K2O": 100},
            "Papaya": {"N": 200, "P2O5": 200, "K2O": 250},
            "Banana": {"N": 200, "P2O5": 60, "K2O": 300},
            "Mango": {"N": 100, "P2O5": 50, "K2O": 100},
            "Grapes": {"N": 150, "P2O5": 100, "K2O": 200},
            "Watermelon": {"N": 100, "P2O5": 60, "K2O": 80},
            "Muskmelon": {"N": 80, "P2O5": 50, "K2O": 70},
            "Tomato": {"N": 150, "P2O5": 100, "K2O": 150},
            "Potato": {"N": 180, "P2O5": 100, "K2O": 150},
            "Mustard": {"N": 80, "P2O5": 40, "K2O": 40},
            "Soybean": {"N": 25, "P2O5": 60, "K2O": 40}
        }

        # Crop-specific agronomic application guidelines & split-dose schedules
        self.crop_split_schedules = {
            "Rice": "Split schedule: 50% N + 100% P & K as basal dose during final puddling; 25% N at active tillering (20-25 DAT); 25% N at panicle initiation (45-50 DAT).",
            "Wheat": "Split schedule: 50% N + 100% P & K as basal dose at sowing; 50% N top-dressed just prior to 1st irrigation at Crown Root Initiation (CRI at 21 DAS).",
            "Cotton": "Split schedule: 100% P & K + 33% N as basal dose; 33% N top-dressed at square formation (45 DAS); 33% N at peak boll development (75 DAS) with 2% DAP foliar spray.",
            "Maize": "Split schedule: 100% P & K + 25% N as basal dose; 50% N top-dressed at knee-high stage (30 DAS); remaining 25% N at tasseling/silking stage.",
            "Sugarcane": "Split schedule (4 doses): 25% N + 100% P as basal; 25% N at 45 DAP (tillering); 25% N at 90 DAP (formative); remaining 25% N + 100% K at 120 DAP (grand growth).",
            "Groundnut": "Split schedule: 100% N, P, K applied as basal dose before sowing. Apply Gypsum @ 400 kg/ha at pegging (40-45 DAS) for pod filling and shell hardening.",
            "Tomato": "Fertigation schedule: Basal 20% N, P, K; 8 weekly fertigation splits using water-soluble 19:19:19 during vegetative, 12:61:0 at flowering, and 13:0:45 during fruit sizing.",
            "Potato": "Split schedule: 50% N + 100% P & K (preferably SOP) as basal dose; remaining 50% N top-dressed at first earthing-up (30-35 DAS) to avoid greening.",
            "Chickpea": "Single basal application: Full N (starter dose), P, K and Sulphur applied in seed-furrow at sowing. Rhizobium fixes N biologically after 20 DAS.",
            "Soybean": "Single basal dose: 25 kg N starter + 60 kg P2O5 + 40 kg K2O + 30 kg Sulphur at sowing. Rhizobium nodules supply required nitrogen.",
            "Mustard": "Split schedule: 50% N + 100% P, K, Sulphur as basal; remaining 50% N top-dressed at rosette/pre-flowering stage (30 DAS) before 1st irrigation.",
            "Banana": "Monthly fertigation: Apply 200g N, 60g P2O5, 300g K2O per plant across 8 monthly installments starting from 2nd to 9th month after planting.",
            "Mango": "Ring-basin schedule: Apply full organic manure + 500g N, 250g P, 750g K per mature tree post-harvest (July-Aug). Apply second light K spray at marble fruit stage.",
            "Grapes": "Split pruning schedule: 40% N + 100% P + 25% K after April foundation pruning; 60% N + 75% K in weekly fertigations post October forward pruning.",
            "Pomegranate": "Bahar schedule: Apply 250g N + 250g P + 250g K + 25 kg FYM per plant at Bahar initiation. Top-dress 250g K at fruit sizing (60 DAFS)."
        }

        # Crop-specific organic and biofertilizer recommendations
        self.crop_organic_protocols = {
            "Rice": "Incorporate green manure (*Sesbania aculeata* / Dhaincha @ 20 kg/ha) at puddling + Blue-Green Algae (BGA @ 10 kg/ha) or *Azospirillum* (2 kg/ha).",
            "Wheat": "Apply well-decomposed FYM @ 10 t/ha + *Azotobacter* seed treatment (200g/10kg seed) + Phosphate Solubilizing Bacteria (PSB @ 5 kg/ha).",
            "Cotton": "Apply Farm Yard Manure @ 10-12 t/ha + Neem cake @ 250 kg/ha in furrows to deter soil-borne root grubs and nematodes.",
            "Sugarcane": "Apply Pressmud / Filter cake @ 10 t/ha + *Gluconacetobacter diazotrophicus* (endophytic nitrogen fixer) + Trash mulching @ 5 t/ha in alternate furrows.",
            "Groundnut": "Inoculate kernels with *Rhizobium* (NC-92) @ 500 g/ha + *Trichoderma viride* (4 g/kg seed) + Phosphobacteria + FYM @ 7.5 t/ha.",
            "Tomato": "Apply Vermicompost @ 3 t/ha + *Trichoderma harzianum* (2.5 kg/ha) enriched FYM + Arbuscular Mycorrhizal Fungi (VAM @ 10 kg/ha) for root vitality.",
            "Potato": "Apply well-rotted FYM @ 25 t/ha + *Azotobacter* + *Bacillus subtilis* to prevent Common Scab and Rhizoctonia black scurf.",
            "Chickpea": "Seed inoculation with *Mesorhizobium ciceri* (20 g/kg seed) + PSB (10 g/kg) + Vermicompost @ 2 t/ha. Strictly avoid excessive synthetic nitrogen.",
            "Soybean": "Seed inoculation with *Bradyrhizobium japonicum* + PSB (10 g/kg) + 5 tonnes FYM/ha. Improves nodulation and seed protein synthesis.",
            "Mustard": "Apply FYM @ 8 t/ha + *Azotobacter* + Sulphur-oxidizing bacteria (*Thiobacillus* @ 2 kg/ha) to enhance oil content.",
            "Banana": "Apply 10 kg FYM + 1 kg Neem cake + 50g VAM per pit at planting. Top-dress Vermicompost @ 2 kg/plant at 3rd and 5th month.",
            "Mango": "Apply 50 kg well-decomposed FYM + 2.5 kg Neem cake + 250g *Trichoderma* per mature tree basin during post-harvest clearing.",
            "Grapes": "Apply 30 tonnes FYM/ha + Green manuring with Sunnhemp in inter-row spaces during monsoon + Humic acid fertigation.",
            "Pomegranate": "Apply 25 kg FYM + 2 kg Neem cake + 50g *Pseudomonas fluorescens* per plant basin to prevent bacterial blight."
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
        target = self.crop_npk_targets.get(crop_clean, {"N": 100, "P2O5": 50, "K2O": 50})
        n_deficit = max(0.0, target["N"] - n)
        p_deficit = max(0.0, target["P2O5"] - p)
        k_deficit = max(0.0, target["K2O"] - k)

        deficits = []
        if n_deficit > 20:
            deficits.append(f"Nitrogen deficit: ~{int(n_deficit)} kg/ha required. Apply Urea (46% N) in split doses according to crop stages.")
        if p_deficit > 15:
            deficits.append(f"Phosphorus deficit: ~{int(p_deficit)} kg/ha required. Apply DAP (18:46:0) or Single Super Phosphate (SSP) as basal placement.")
        if k_deficit > 20:
            deficits.append(f"Potassium deficit: ~{int(k_deficit)} kg/ha required. Apply MOP (Muriate of Potash 60% K2O) or Sulphate of Potash (SOP).")

        if not deficits:
            deficits.append("Soil macronutrients are in optimal balance for the target crop. Maintain organic compost and microbial inoculants.")

        # Recommendations list
        recs = []
        if predicted_fertilizer:
            recs.append(f"{predicted_fertilizer} (ML Primary Match)")

        if p_deficit > 15 and "DAP" not in [r.split()[0] for r in recs]:
            recs.append("DAP (Di-Ammonium Phosphate)")
        if n_deficit > 20 and "Urea" not in [r.split()[0] for r in recs]:
            recs.append("Urea (Top-dressing)")
        if k_deficit > 20:
            recs.append("MOP (Potash 60% K2O)")

        if not recs:
            recs = ["NPK 19:19:19 (Balanced Water Soluble)", "Organic Vermicompost", "Urea (Split)"]

        custom_guide = self.crop_split_schedules.get(crop_clean, f"Apply 50% Nitrogen and 100% Phosphorus & Potassium as basal dose before sowing. Apply remaining Nitrogen in split doses at active vegetative and flowering stages.")

        return {
            "recommended_fertilizers": recs[:4],
            "primary_fertilizer": predicted_fertilizer or recs[0],
            "soil_deficits": deficits,
            "target_npk_ratio": f"{target['N']}:{target['P2O5']}:{target['K2O']} (N:P2O5:K2O, kg/ha)",
            "probabilities": probs_dict,
            "application_tips": custom_guide
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

        crop_clean = crop.strip().title()

        # Baseline recommendation
        base = self.recommend(
            crop=crop_clean,
            soil_type=soil_type,
            n=n,
            p=p,
            k=k,
            temp=temp,
            humidity=humid,
            moisture=moist
        )

        target = self.crop_npk_targets.get(crop_clean, {"N": 100, "P2O5": 50, "K2O": 50})
        n_def = max(0.0, target["N"] - n)
        p_def = max(0.0, target["P2O5"] - p)
        k_def = max(0.0, target["K2O"] - k)

        # Trigger thresholds: N >20, P2O5 >15, K2O >20 kg/ha.
        # Fertilizer grades are expressed as N-P2O5-K2O.
        n_triggered = n_def > 20.0
        p_triggered = p_def > 15.0
        k_triggered = k_def > 20.0

        # DAP (18-46-0) supplies both P2O5 and N. Credit its N contribution
        # before calculating the additional urea requirement.
        dap_product_kg_ha = (p_def / 0.46) if p_triggered else 0.0
        dap_n_contribution = dap_product_kg_ha * 0.18
        remaining_n_after_dap = max(0.0, n_def - dap_n_contribution)
        urea_product_kg_ha = (remaining_n_after_dap / 0.46) if n_triggered else 0.0
        mop_product_kg_ha = (k_def / 0.60) if k_triggered else 0.0
        ssp_product_kg_ha = (p_def / 0.16) if p_triggered else 0.0

        # DAP and SSP are alternative phosphorus sources, not additive doses.
        urea_bags = round(urea_product_kg_ha / 50.0, 1)
        dap_bags = round(dap_product_kg_ha / 50.0, 1)
        mop_bags = round(mop_product_kg_ha / 50.0, 1)
        ssp_bags = round(ssp_product_kg_ha / 50.0, 1)

        # Crop-aware micronutrient & pH amendment alerts
        micro_advice = {}
        if zinc is not None and zinc < 0.8:
            if crop_clean in ["Rice", "Wheat", "Maize"]:
                micro_advice["Zinc Deficit (Khaira / White Bud Risk)"] = f"Soil Zn is {zinc} ppm (<0.8 ppm threshold). Broadcast Zinc Sulphate (ZnSO4 21%) @ 25 kg/ha as basal or spray 0.5% ZnSO4 + 0.25% Lime."
            else:
                micro_advice["Zinc"] = f"Soil zinc ({zinc} ppm) is below critical limit. Broadcast Zinc Sulphate @ 20 kg/ha."

        if sulphur is not None and sulphur < 10.0:
            if crop_clean in ["Mustard", "Groundnut", "Soybean", "Groundnuts"]:
                micro_advice["Sulphur Deficit (Oil Synthesis Critical)"] = f"Soil S is {sulphur} ppm (<10 ppm). Apply Elemental Sulphur @ 30 kg/ha or Single Super Phosphate (SSP) @ 250 kg/ha to ensure high oil percentage."
            else:
                micro_advice["Sulphur"] = f"Soil sulphur ({sulphur} ppm) is deficient. Apply Gypsum / SSP @ 20 kg/ha."

        if ph < 5.5:
            micro_advice["Soil Acidity (Lime Correction)"] = f"Soil pH is {ph} (Acidic). Incorporate Agricultural Lime (CaCO3) @ 500-1000 kg/ha 3 weeks before sowing to neutralize Al/Fe toxicity."
        elif ph > 8.2:
            micro_advice["Soil Alkalinity (Gypsum Correction)"] = f"Soil pH is {ph} (Alkaline/Calcareous). Apply agricultural Gypsum (CaSO4) @ 1000-1500 kg/ha with green manuring (Dhaincha)."

        if crop_clean in ["Tomato", "Potato", "Pomegranate", "Apple"]:
            micro_advice["Calcium & Boron (Fruit Setting & Blossom End Rot)"] = f"For {crop_clean}, spray Calcium Nitrate (0.5%) + Boron / Solubor (0.1%) during flowering and early fruit setting."
        elif crop_clean == "Cotton":
            micro_advice["Magnesium (Leaf Reddening Prevention)"] = "Spray Magnesium Sulphate (MgSO4 1%) + 1% Urea at 50 & 75 DAS to prevent leaf reddening."

        # Fetch tailored organic protocol
        organic_advice = self.crop_organic_protocols.get(
            crop_clean,
            "Incorporate well-decomposed Farm Yard Manure (FYM) @ 10 tonnes/ha or Vermicompost @ 2.5 tonnes/ha with biofertilizers (Azospirillum & PSB)."
        )

        return {
            "primary_fertilizer": base.get("primary_fertilizer", "Urea"),
            "recommended_fertilizers": base.get("recommended_fertilizers", []),
            "application_guidelines": base.get("application_tips"),
            "dosage_kg_per_ha": {
                "nitrogen_deficit_kg_ha": round(n_def, 1),
                "phosphorus_deficit_as_P2O5_kg_ha": round(p_def, 1),
                "potassium_deficit_as_K2O_kg_ha": round(k_def, 1)
            },
            "fertilizer_product_rates_kg_ha": {"urea": round(urea_product_kg_ha, 1), "dap": round(dap_product_kg_ha, 1), "mop": round(mop_product_kg_ha, 1), "ssp_alternative_to_dap": round(ssp_product_kg_ha, 1)},
            "commercial_bags_recommended": {
                "urea_50kg_bags": urea_bags,
                "dap_50kg_bags": dap_bags,
                "mop_50kg_bags": mop_bags,
                "ssp_50kg_bags": ssp_bags
            },
            "organic_alternatives": organic_advice,
            "micronutrient_advice": micro_advice,
            "target_npk_ratio": base.get("target_npk_ratio", "120:60:60 (N:P2O5:K2O, kg/ha)")
        }

fertilizer_service = FertilizerService()
