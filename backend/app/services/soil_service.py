import os
import json
import joblib
import numpy as np
from PIL import Image
import io
from backend.app.config import settings
from typing import Dict, Any

class SoilService:
    def __init__(self):
        self.vision_bundle_path = os.path.join(settings.MODELS_DIR, "soil_vision_model.joblib")
        self.classes_path = os.path.join(settings.MODELS_DIR, "soil_classes.json")
        self.vision_bundle = None
        self.classes = [
            "Alluvial_Soil",
            "Arid_Soil",
            "Black_Soil",
            "Laterite_Soil",
            "Mountain_Soil",
            "Red_Soil",
            "Yellow_Soil"
        ]
        self.load_resources()

    def load_resources(self):
        if os.path.exists(self.classes_path):
            try:
                with open(self.classes_path, "r") as f:
                    cls_map = json.load(f)
                    self.classes = [cls_map[str(i)] for i in range(len(cls_map))]
            except Exception as e:
                print(f"[SoilService] Notice loading classes: {e}")

        if os.path.exists(self.vision_bundle_path):
            try:
                self.vision_bundle = joblib.load(self.vision_bundle_path)
                print(f"[SoilService] Successfully loaded trained model: {self.vision_bundle.get('best_model_name', 'Ensemble')}")
            except Exception as e:
                print(f"[SoilService] Error loading vision bundle: {e}")

    def validate_is_soil(self, scaled_feats: np.ndarray, arr: np.ndarray, hsv_arr: np.ndarray) -> Dict[str, Any]:
        """
        Out-of-Distribution (OOD) Soil Domain Detector.
        Uses trained IsolationForest ensemble on empirical soil distributions,
        complemented by domain heuristics (blank UI screenshot, extreme artificial colors).
        """
        # 1. Blank image, UI screenshot, or pure document detection
        mean_rgb = float(np.mean(arr))
        rgb_std = float(np.std(arr))
        white_pixel_ratio = float(np.mean(arr > 240))
        if (mean_rgb > 225 and rgb_std < 40) or white_pixel_ratio > 0.65:
            return {
                "is_valid_soil": False,
                "rejection_reason": "Document, UI screenshot, or blank background detected. Please upload an authentic photo of field soil."
            }

        # 2. Solid color or non-texture graphic
        if float(np.var(arr)) < 45:
            return {
                "is_valid_soil": False,
                "rejection_reason": "Uniform solid graphic detected. Please upload a clear photo of your field soil."
            }

        # 3. Human Portrait & Synthetic Clothing Check
        r = arr[:, :, 0]
        g = arr[:, :, 1]
        b = arr[:, :, 2]
        face_region_r = r[25:120, 30:170]
        face_region_g = g[25:120, 30:170]
        face_region_b = b[25:120, 30:170]
        face_skin = (face_region_r > (face_region_g + 10)) & (face_region_g > (face_region_b + 5)) & (face_region_r > 100) & (face_region_b > 35) & (face_region_b < 175)
        face_skin_ratio = float(np.mean(face_skin))

        body_r = r[120:200, 20:180]
        body_g = g[120:200, 20:180]
        body_b = b[120:200, 20:180]
        blue_clothing_bottom = (body_b > (body_r + 12)) & (body_b > (body_g + 8)) & (body_b > 45)
        blue_bottom_ratio = float(np.mean(blue_clothing_bottom))

        top_r = r[0:80, :]
        top_g = g[0:80, :]
        top_b = b[0:80, :]
        blue_top = (top_b > (top_r + 12)) & (top_b > (top_g + 8)) & (top_b > 45)
        blue_top_ratio = float(np.mean(blue_top))

        if face_skin_ratio > 0.15 and blue_bottom_ratio > 0.08 and (blue_bottom_ratio >= blue_top_ratio):
            return {
                "is_valid_soil": False,
                "rejection_reason": "Human portrait / non-soil object detected. Please upload an authentic photo of agricultural field soil."
            }

        # 4. Isolation Forest OOD Model Check
        if self.vision_bundle and "ood_detector" in self.vision_bundle:
            ood_detector = self.vision_bundle["ood_detector"]
            ood_score = float(ood_detector.decision_function(scaled_feats)[0])
            if ood_score < -0.005:
                return {
                    "is_valid_soil": False,
                    "rejection_reason": "The uploaded photo does not exhibit natural soil or agricultural land characteristics. Please upload a photo of field soil."
                }

        return {
            "is_valid_soil": True,
            "rejection_reason": ""
        }

    def extract_features(self, img: Image.Image):
        img_rgb = img.convert("RGB").resize((200, 200))
        arr = np.array(img_rgb, dtype=np.float32)

        features = []

        # 1. RGB statistics (21 features)
        for ch in range(3):
            channel = arr[:, :, ch]
            features.extend([
                float(np.mean(channel)),
                float(np.std(channel)),
                float(np.percentile(channel, 10)),
                float(np.percentile(channel, 25)),
                float(np.percentile(channel, 50)),
                float(np.percentile(channel, 75)),
                float(np.percentile(channel, 90))
            ])

        # 2. RGB Histograms (48 features)
        for ch in range(3):
            hist, _ = np.histogram(arr[:, :, ch], bins=16, range=(0, 256), density=True)
            features.extend([float(v) for v in hist])

        # 3. HSV Color Space (60 features)
        hsv = img_rgb.convert("HSV")
        hsv_arr = np.array(hsv, dtype=np.float32)
        for ch in range(3):
            channel = hsv_arr[:, :, ch]
            features.extend([
                float(np.mean(channel)),
                float(np.std(channel)),
                float(np.percentile(channel, 25)),
                float(np.percentile(channel, 75))
            ])
        for ch in range(3):
            hist, _ = np.histogram(hsv_arr[:, :, ch], bins=16, range=(0, 256), density=True)
            features.extend([float(v) for v in hist])

        # 4. Spatial Gradients (4 features)
        gray = np.mean(arr, axis=2)
        grad_x = np.diff(gray, axis=1)
        grad_y = np.diff(gray, axis=0)
        features.extend([
            float(np.var(grad_x)),
            float(np.var(grad_y)),
            float(np.mean(np.abs(grad_x))),
            float(np.mean(np.abs(grad_y)))
        ])

        return np.array(features, dtype=np.float32), arr, hsv_arr

    def analyze_soil_image(self, image_bytes: bytes) -> Dict[str, Any]:
        img = Image.open(io.BytesIO(image_bytes))
        feats, arr, hsv_arr = self.extract_features(img)

        if self.vision_bundle is None:
            raise RuntimeError("Soil ML Model is not loaded.")

        scaler = self.vision_bundle["scaler"]
        model = self.vision_bundle["model"]
        classes = self.vision_bundle["classes"]

        scaled_feats = scaler.transform([feats])

        # 1. Run Domain Validation (Ensure it's actually soil)
        validation = self.validate_is_soil(scaled_feats, arr, hsv_arr)

        h_mean = float(np.mean(hsv_arr[:, :, 0]))
        s_mean = float(np.mean(hsv_arr[:, :, 1]))
        v_mean = float(np.mean(hsv_arr[:, :, 2]))

        visual_features = {
            "mean_hue": round(h_mean, 2),
            "mean_saturation": round(s_mean, 2),
            "mean_brightness": round(v_mean, 2),
            "texture_roughness": round(float(np.var(arr)), 2),
            "estimated_visual_moisture": "High" if v_mean < 85 else ("Medium" if v_mean < 150 else "Low")
        }

        # If it is NOT a soil image, return a clear rejection
        if not validation["is_valid_soil"]:
            return {
                "detected_soil_type": "Invalid (Non-Soil Image)",
                "confidence": 0.0,
                "is_valid_soil": False,
                "rejection_reason": validation["rejection_reason"] or "The uploaded photo does not appear to be soil or agricultural land. Please upload a clear photo of your field soil.",
                "all_probabilities": {},
                "visual_features": visual_features
            }

        # 2. Run Classification for Real Soil
        probs = model.predict_proba(scaled_feats)[0]
        top_idx = int(np.argmax(probs))

        detected_type = classes[top_idx]
        confidence = float(probs[top_idx])

        probabilities = {
            c.replace("_", " "): round(float(p), 4)
            for c, p in zip(classes, probs)
        }

        sorted_probs = dict(sorted(probabilities.items(), key=lambda item: item[1], reverse=True))

        return {
            "detected_soil_type": detected_type.replace("_", " "),
            "confidence": round(confidence, 4),
            "is_valid_soil": True,
            "rejection_reason": None,
            "all_probabilities": sorted_probs,
            "visual_features": visual_features
        }

soil_service = SoilService()
