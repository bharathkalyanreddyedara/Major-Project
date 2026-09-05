"""
Multi-Model Benchmark & Training: Fertilizer Recommendation
Benchmarks XGBoost, LightGBM, MLP Neural Network, and ExtraTrees.
Selects and exports the highest-performing model.
"""

import os
import joblib
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier, VotingClassifier
from sklearn.metrics import accuracy_score, f1_score
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier

def benchmark_and_train_fertilizer_models(
    dataset_path="backend/data/Crop_vs_Fertilizer.csv",
    model_output_dir="backend/models"
):
    if not os.path.exists(dataset_path) and os.path.exists("Crop_vs_Fertilizer.csv"):
        dataset_path = "Crop_vs_Fertilizer.csv"

    print("=" * 60)
    print(">>> MULTI-MODEL BENCHMARK: FERTILIZER RECOMMENDATION")
    print("=" * 60)

    df = pd.read_csv(dataset_path)
    df.columns = [c.strip() for c in df.columns]

    soil_encoder = LabelEncoder()
    df["Soil_Type_Enc"] = soil_encoder.fit_transform(df["Soil Type"].astype(str).str.strip().str.title())

    crop_encoder = LabelEncoder()
    df["Crop_Type_Enc"] = crop_encoder.fit_transform(df["Crop Type"].astype(str).str.strip().str.title())

    target_encoder = LabelEncoder()
    df["Fertilizer_Enc"] = target_encoder.fit_transform(df["Fertilizer Name"].astype(str).str.strip())

    # Feature Engineering
    df["N_to_P"] = df["Nitrogen"] / (df["Phosphorous"] + 1e-4)
    df["N_to_K"] = df["Nitrogen"] / (df["Potassium"] + 1e-4)
    df["P_to_K"] = df["Phosphorous"] / (df["Potassium"] + 1e-4)
    df["NPK_Total"] = df["Nitrogen"] + df["Phosphorous"] + df["Potassium"]

    feature_cols = ["Temparature", "Humidity", "Moisture", "Soil_Type_Enc", "Crop_Type_Enc", "Nitrogen", "Potassium", "Phosphorous", "N_to_P", "N_to_K", "P_to_K", "NPK_Total"]
    
    X = df[feature_cols]
    y = df["Fertilizer_Enc"]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    models = {
        "XGBoost": XGBClassifier(n_estimators=150, learning_rate=0.08, max_depth=6, random_state=42, n_jobs=-1, eval_metric="mlogloss"),
        "LightGBM": LGBMClassifier(n_estimators=150, learning_rate=0.08, max_depth=6, random_state=42, n_jobs=-1, verbose=-1),
        "MLP Neural Net": MLPClassifier(hidden_layer_sizes=(128, 64), max_iter=350, random_state=42, early_stopping=True),
        "ExtraTrees": ExtraTreesClassifier(n_estimators=200, max_depth=16, random_state=42, n_jobs=-1),
        "RandomForest": RandomForestClassifier(n_estimators=200, max_depth=16, random_state=42, n_jobs=-1)
    }

    results = {}
    best_name = None
    best_acc = 0.0
    best_model = None

    print(f"\nEvaluating {len(models)} model architectures...")
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

    print("=" * 60)
    print(f"[BEST FERTILIZER MODEL]: {best_name} with {best_acc * 100:.2f}% Test Accuracy!")
    print("=" * 60)

    os.makedirs(model_output_dir, exist_ok=True)
    bundle = {
        "model": best_model,
        "scaler": scaler,
        "soil_encoder": soil_encoder,
        "crop_encoder": crop_encoder,
        "target_encoder": target_encoder,
        "fertilizer_classes": list(target_encoder.classes_),
        "soil_classes": list(soil_encoder.classes_),
        "crop_classes": list(crop_encoder.classes_),
        "feature_cols": feature_cols,
        "best_model_name": best_name,
        "accuracy": best_acc,
        "leaderboard": {k: {"accuracy": v["accuracy"], "f1": v["f1"]} for k, v in results.items()}
    }

    bundle_path = os.path.join(model_output_dir, "fertilizer_recommender.joblib")
    joblib.dump(bundle, bundle_path, compress=3)
    print(f"Exported top fertilizer model bundle to: {bundle_path}\n")

if __name__ == "__main__":
    benchmark_and_train_fertilizer_models()
