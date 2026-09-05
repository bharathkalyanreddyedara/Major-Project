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
        # 1. Load Classes Map
        if os.path.exists(self.classes_path):
            try:
                with open(self.classes_path, "r") as f:
                    cls_map = json.load(f)
                    self.classes = [cls_map[str(i)] for i in range(len(cls_map))]
            except Exception as e:
                print(f"[SoilService] Notice loading classes: {e}")

        # 2. Load Vision ML Bundle
        if os.path.exists(self.vision_bundle_path):
            try:
                self.vision_bundle = joblib.load(self.vision_bundle_path)
                print(f"[SoilService] Successfully loaded trained model: {self.vision_bundle.get('best_model_name', 'Ensemble')}")
            except Exception as e:
                print(f"[SoilService] Error loading vision bundle: {e}")
        else:
            print(f"[SoilService] Warning: {self.vision_bundle_path} not found.")

    def extract_features(self, img: Image.Image):
        # Resize to fixed standard dimensions
        img_rgb = img.convert("RGB").resize((200, 200))
        arr = np.array(img_rgb, dtype=np.float32)

        features = []

        # 1. RGB statistics (7 stats * 3 channels = 21 features)
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

        # 2. RGB Histograms (16 bins * 3 channels = 48 features)
        for ch in range(3):
            hist, _ = np.histogram(arr[:, :, ch], bins=16, range=(0, 256), density=True)
            features.extend([float(v) for v in hist])

        # 3. HSV Color Space
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

        # 4. Spatial Gradients & Texture Variances
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

        # Compute dynamic visual metrics from actual uploaded image pixels
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

        if self.vision_bundle is None:
            raise RuntimeError("Soil ML Model is not loaded. Please ensure backend/models/soil_vision_model.joblib exists.")

        scaler = self.vision_bundle["scaler"]
        model = self.vision_bundle["model"]
        classes = self.vision_bundle["classes"]

        scaled_feats = scaler.transform([feats])
        probs = model.predict_proba(scaled_feats)[0]
        top_idx = int(np.argmax(probs))

        detected_type = classes[top_idx]
        confidence = float(probs[top_idx])

        probabilities = {
            c.replace("_", " "): round(float(p), 4)
            for c, p in zip(classes, probs)
        }

        # Sort probabilities descending
        sorted_probs = dict(sorted(probabilities.items(), key=lambda item: item[1], reverse=True))

        return {
            "detected_soil_type": detected_type.replace("_", " "),
            "confidence": round(confidence, 4),
            "all_probabilities": sorted_probs,
            "visual_features": visual_features
        }

soil_service = SoilService()
