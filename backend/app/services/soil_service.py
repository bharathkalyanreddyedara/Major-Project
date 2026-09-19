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
        Rejects portraits, cartoons, studio dark backgrounds (e.g. Joker/movie posters),
        artificial colors, documents, and non-agricultural objects.
        """
        r = arr[:, :, 0].astype(float)
        g = arr[:, :, 1].astype(float)
        b = arr[:, :, 2].astype(float)
        lum = (r + g + b) / 3.0

        # 1. Blank Document / White Screen / UI Screenshot
        if (np.mean(lum) > 225 and np.std(lum) < 35) or np.mean(lum > 245) > 0.60:
            return {
                "is_valid_soil": False,
                "rejection_reason": "Document, UI screenshot, or blank white background detected. Please upload a real photo of agricultural soil."
            }

        # 2. Solid Color / Flat Graphic
        if float(np.var(lum)) < 30.0:
            return {
                "is_valid_soil": False,
                "rejection_reason": "Uniform solid graphic detected. Please upload a clear photo of your field soil."
            }

        # 3. Studio Black Backdrop / Dark Poster / Dark Wallpaper (e.g., Joker / Portrait on Black Background)
        # Real soil (even black vertisol) has ambient illumination and granular texture.
        # Studio black backgrounds have pure near-zero black pixels (RGB < 18) with flat zero variance.
        pitch_black_mask = (r < 18) & (g < 18) & (b < 18)
        pitch_black_ratio = float(np.mean(pitch_black_mask))
        
        # If > 30% of image is pure studio black void, reject immediately
        if pitch_black_ratio > 0.30:
            return {
                "is_valid_soil": False,
                "rejection_reason": "Dark studio backdrop / black wallpaper detected. Agricultural soil photos should show field ground rather than an object against a dark backdrop."
            }

        # If > 15% is pitch black AND there is a high-contrast subject (like Joker face paint or clothing)
        high_contrast_bright = float(np.mean(lum > 185))
        if pitch_black_ratio > 0.15 and high_contrast_bright > 0.08:
            return {
                "is_valid_soil": False,
                "rejection_reason": "Character portrait / object on dark background detected. Please upload a direct photo of the soil surface."
            }

        # 4. Out-of-Gamut Synthetic Colors (Electric Blue, Vibrant Violet/Purple, Pure Neon Pink/Magenta)
        electric_purple = (r > 100) & (b > 100) & (g < (r - 40)) & (g < (b - 40))
        pure_magenta = (r > 160) & (b > 140) & (g < 70)
        intense_cyan = (b > 180) & (g > 160) & (r < 60)
        pure_lipstick_red = (r > 190) & (g < 40) & (b < 40)
        
        synthetic_ratio = float(np.mean(electric_purple | pure_magenta | intense_cyan | pure_lipstick_red))
        if synthetic_ratio > 0.035:
            return {
                "is_valid_soil": False,
                "rejection_reason": "Artificial / synthetic colors detected (cosplay, neon makeup, vibrant clothing, or digital graphics). Natural soil does not exhibit these pigments."
            }

        # 5. Isolation Forest Statistical OOD Check
        if self.vision_bundle and "ood_detector" in self.vision_bundle:
            ood_detector = self.vision_bundle["ood_detector"]
            ood_score = float(ood_detector.decision_function(scaled_feats)[0])
            if ood_score < -0.06:
                return {
                    "is_valid_soil": False,
                    "rejection_reason": "The uploaded photo does not exhibit natural soil or agricultural land characteristics. Please upload a clear photo of field soil."
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
