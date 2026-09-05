"""
Soil Image Classifier Training Script (Vision Feature Extraction + ML Ensemble)
Extracts multi-channel color moments and texture metrics using PIL & NumPy
from all 1,189 images across the 7 classes in Orignal-Dataset.
Trains an Ensemble Random Forest Classifier.
"""

import os
import glob
import json
import joblib
import numpy as np
from PIL import Image
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import classification_report, accuracy_score

def extract_features_from_image(img_path):
    try:
        with Image.open(img_path) as img_raw:
            img = img_raw.convert("RGB").resize((200, 200))
            arr = np.array(img, dtype=np.float32)

            features = []

            # 1. RGB statistics (Mean, Std, Percentiles)
            for ch in range(3):
                channel = arr[:, :, ch]
                features.extend([
                    np.mean(channel),
                    np.std(channel),
                    np.percentile(channel, 25),
                    np.percentile(channel, 75)
                ])

            # 2. Color Histograms (16 bins per channel)
            for ch in range(3):
                hist, _ = np.histogram(arr[:, :, ch], bins=16, range=(0, 256), density=True)
                features.extend(hist)

            # 3. HSV conversion statistics
            hsv = img.convert("HSV")
            hsv_arr = np.array(hsv, dtype=np.float32)
            for ch in range(3):
                channel = hsv_arr[:, :, ch]
                features.extend([
                    np.mean(channel),
                    np.std(channel),
                    np.percentile(channel, 25),
                    np.percentile(channel, 75)
                ])

            # 4. Spatial gradient / texture variance (Grayscale difference)
            gray = np.mean(arr, axis=2)
            grad_x = np.diff(gray, axis=1)
            grad_y = np.diff(gray, axis=0)
            features.append(np.var(grad_x))
            features.append(np.var(grad_y))

            return np.array(features, dtype=np.float32)
    except Exception as e:
        return None

def train_soil_vision_model(
    dataset_dir="Orignal-Dataset/Orignal-Dataset",
    model_output_path="backend/models/soil_vision_model.joblib",
    classes_json_path="backend/models/soil_classes.json"
):
    if not os.path.exists(dataset_dir):
        if os.path.exists("Orignal-Dataset") and os.path.exists("Orignal-Dataset/Alluvial_Soil"):
            dataset_dir = "Orignal-Dataset"

    print("Loading image dataset from:", dataset_dir)
    class_folders = sorted([f for f in os.listdir(dataset_dir) if os.path.isdir(os.path.join(dataset_dir, f))])
    print("Found Soil Classes:", class_folders)

    X = []
    y = []

    for class_name in class_folders:
        folder_path = os.path.join(dataset_dir, class_name)
        image_files = glob.glob(os.path.join(folder_path, "*.*"))
        print(f"Extracting features from {len(image_files)} images in '{class_name}'...")

        for img_p in image_files:
            feats = extract_features_from_image(img_p)
            if feats is not None:
                X.append(feats)
                y.append(class_name)

    X = np.array(X)
    y = np.array(y)
    print(f"Extracted features shape: {X.shape}, Total valid samples: {len(y)}")

    encoder = LabelEncoder()
    y_enc = encoder.fit_transform(y)

    X_train, X_test, y_train, y_test = train_test_split(X, y_enc, test_size=0.2, random_state=42, stratify=y_enc)

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    print("Training Random Forest Classifier on Soil Visual Features...")
    model = RandomForestClassifier(n_estimators=200, max_depth=20, random_state=42, n_jobs=-1)
    model.fit(X_train_scaled, y_train)

    y_pred = model.predict(X_test_scaled)
    acc = accuracy_score(y_test, y_pred)
    print("==================================================")
    print(f"Soil Model Test Accuracy: {acc * 100:.2f}%")
    print("==================================================")
    print(classification_report(y_test, y_pred, target_names=encoder.classes_))

    os.makedirs(os.path.dirname(model_output_path), exist_ok=True)
    bundle = {
        "model": model,
        "scaler": scaler,
        "encoder": encoder,
        "classes": list(encoder.classes_)
    }
    joblib.dump(bundle, model_output_path)
    print("Saved Soil Vision model to:", model_output_path)

    # Save classes mapping
    index_to_class = {str(i): c for i, c in enumerate(encoder.classes_)}
    with open(classes_json_path, "w") as f:
        json.dump(index_to_class, f, indent=4)
    print("Saved class mappings to:", classes_json_path)

if __name__ == "__main__":
    train_soil_vision_model()
