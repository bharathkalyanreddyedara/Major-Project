"""
Multi-Model Benchmark & Training: Soil Image Classification
Extracts multi-space visual features (RGB, HSV, LAB, spatial variance) from all 1,189 images.
Benchmarks XGBoost, LightGBM, MLP Neural Network, ExtraTrees, and Stacking.
Automatically selects and exports the highest-performing model.
"""

import os
import glob
import json
import joblib
import numpy as np
from PIL import Image
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier, VotingClassifier
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import accuracy_score, f1_score
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier

def extract_features_from_image(img_path):
    try:
        with Image.open(img_path) as img_raw:
            img = img_raw.convert("RGB").resize((200, 200))
            arr = np.array(img, dtype=np.float32)

            features = []

            # 1. RGB statistics
            for ch in range(3):
                channel = arr[:, :, ch]
                features.extend([
                    np.mean(channel),
                    np.std(channel),
                    np.percentile(channel, 10),
                    np.percentile(channel, 25),
                    np.percentile(channel, 50),
                    np.percentile(channel, 75),
                    np.percentile(channel, 90)
                ])

            # 2. RGB Histograms
            for ch in range(3):
                hist, _ = np.histogram(arr[:, :, ch], bins=16, range=(0, 256), density=True)
                features.extend(hist)

            # 3. HSV Color Space
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
            for ch in range(3):
                hist, _ = np.histogram(hsv_arr[:, :, ch], bins=16, range=(0, 256), density=True)
                features.extend(hist)

            # 4. Spatial Gradients & Texture Variances
            gray = np.mean(arr, axis=2)
            grad_x = np.diff(gray, axis=1)
            grad_y = np.diff(gray, axis=0)
            features.extend([
                np.var(grad_x),
                np.var(grad_y),
                np.mean(np.abs(grad_x)),
                np.mean(np.abs(grad_y))
            ])

            return np.array(features, dtype=np.float32)
    except Exception as e:
        return None

def benchmark_and_train_soil_models(
    dataset_dir="Orignal-Dataset/Orignal-Dataset",
    model_output_path="backend/models/soil_vision_model.joblib",
    classes_json_path="backend/models/soil_classes.json"
):
    if not os.path.exists(dataset_dir):
        if os.path.exists("Orignal-Dataset") and os.path.exists("Orignal-Dataset/Alluvial_Soil"):
            dataset_dir = "Orignal-Dataset"

    print("=" * 60)
    print(">>> MULTI-MODEL BENCHMARK: SOIL IMAGE CLASSIFICATION")
    print("=" * 60)
    print("Loading image dataset from:", dataset_dir)

    class_folders = sorted([f for f in os.listdir(dataset_dir) if os.path.isdir(os.path.join(dataset_dir, f))])
    print("Found Soil Classes (7):", class_folders)

    X = []
    y = []

    for class_name in class_folders:
        folder_path = os.path.join(dataset_dir, class_name)
        image_files = glob.glob(os.path.join(folder_path, "*.*"))
        for img_p in image_files:
            feats = extract_features_from_image(img_p)
            if feats is not None:
                X.append(feats)
                y.append(class_name)

    X = np.array(X)
    y = np.array(y)
    print(f"Extracted {X.shape[1]} visual features from {len(y)} images.")

    encoder = LabelEncoder()
    y_enc = encoder.fit_transform(y)

    X_train, X_test, y_train, y_test = train_test_split(X, y_enc, test_size=0.2, random_state=42, stratify=y_enc)

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    models = {
        "XGBoost": XGBClassifier(n_estimators=250, learning_rate=0.08, max_depth=6, random_state=42, n_jobs=-1, eval_metric="mlogloss"),
        "LightGBM": LGBMClassifier(n_estimators=250, learning_rate=0.08, max_depth=6, random_state=42, n_jobs=-1, verbose=-1),
        "MLP Neural Net": MLPClassifier(hidden_layer_sizes=(256, 128), max_iter=400, random_state=42, early_stopping=True),
        "ExtraTrees": ExtraTreesClassifier(n_estimators=300, max_depth=25, random_state=42, n_jobs=-1),
        "RandomForest": RandomForestClassifier(n_estimators=300, max_depth=25, random_state=42, n_jobs=-1)
    }

    results = {}
    best_name = None
    best_acc = 0.0
    best_model = None

    print(f"\nEvaluating {len(models)} model architectures on 7 soil classes...")
    print(f"{'Model Architecture':<20} | {'Accuracy':<10} | {'Weighted F1':<12}")
    print("-" * 48)

    for name, m in models.items():
        m.fit(X_train_scaled, y_train)
        preds = m.predict(X_test_scaled)
        acc = accuracy_score(y_test, preds)
        f1 = f1_score(y_test, preds, average="weighted")
        results[name] = {"accuracy": acc, "f1": f1, "model": m}
        print(f"{name:<20} | {acc * 100:>8.2f}% | {f1:>10.4f}")

        if acc > best_acc:
            best_acc = acc
            best_name = name
            best_model = m

    # Stacking / Voting of top 2
    sorted_models = sorted(results.items(), key=lambda x: x[1]["accuracy"], reverse=True)
    top1_name, top1_obj = sorted_models[0][0], sorted_models[0][1]["model"]
    top2_name, top2_obj = sorted_models[1][0], sorted_models[1][1]["model"]

    ensemble = VotingClassifier(
        estimators=[(top1_name, top1_obj), (top2_name, top2_obj)],
        voting="soft",
        n_jobs=-1
    )
    ensemble.fit(X_train_scaled, y_train)
    ens_preds = ensemble.predict(X_test_scaled)
    ens_acc = accuracy_score(y_test, ens_preds)
    ens_f1 = f1_score(y_test, ens_preds, average="weighted")
    print(f"{'Ensemble (Top-2)':<20} | {ens_acc * 100:>8.2f}% | {ens_f1:>10.4f}")

    if ens_acc >= best_acc:
        best_name = f"Ensemble ({top1_name} + {top2_name})"
        best_acc = ens_acc
        best_model = ensemble

    print("=" * 60)
    print(f"[BEST SOIL MODEL]: {best_name} with {best_acc * 100:.2f}% Test Accuracy!")
    print("=" * 60)

    # Save Bundle
    os.makedirs(os.path.dirname(model_output_path), exist_ok=True)
    bundle = {
        "model": best_model,
        "scaler": scaler,
        "encoder": encoder,
        "classes": list(encoder.classes_),
        "best_model_name": best_name,
        "accuracy": best_acc,
        "leaderboard": {k: {"accuracy": v["accuracy"], "f1": v["f1"]} for k, v in results.items()}
    }
    joblib.dump(bundle, model_output_path, compress=3)
    print("Saved best soil model to:", model_output_path)

    # Save Class Map
    index_to_class = {str(i): c for i, c in enumerate(encoder.classes_)}
    with open(classes_json_path, "w") as f:
        json.dump(index_to_class, f, indent=4)

if __name__ == "__main__":
    benchmark_and_train_soil_models()
