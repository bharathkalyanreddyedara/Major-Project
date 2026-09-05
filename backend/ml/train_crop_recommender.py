import os
import joblib
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

def train_crop_model(
    dataset1_path="backend/data/Soil_vs_Crop.csv",
    dataset2_path="backend/data/SoilProp_vs_Crop.csv",
    model_output_dir="backend/models"
):
    if not os.path.exists(dataset1_path) and os.path.exists("Soil_vs_Crop.csv"):
        dataset1_path = "Soil_vs_Crop.csv"
        dataset2_path = "SoilProp_vs_Crop.csv"

    print("Loading Crop Datasets...")
    os.makedirs(model_output_dir, exist_ok=True)
    
    df1 = pd.read_csv(dataset1_path)
    df1.columns = [c.strip() for c in df1.columns]
    
    temp_cols = [c for c in df1.columns if "T2M_MAX" in c or "T2M_MIN" in c]
    rain_cols = [c for c in df1.columns if "PRECTOTCORR" in c]
    
    df1["Temperature"] = df1[temp_cols].mean(axis=1) if temp_cols else 25.0
    df1["Rainfall"] = df1[rain_cols].sum(axis=1) if rain_cols else 100.0
    df1["Humidity"] = 65.0
    df1["SoilType"] = df1["Soilcolor"].str.lower().str.strip()
    
    clean_df1 = pd.DataFrame({
        "N": pd.to_numeric(df1["N"], errors="coerce").fillna(50),
        "P": pd.to_numeric(df1["P"], errors="coerce").fillna(30),
        "K": pd.to_numeric(df1["K"], errors="coerce").fillna(150),
        "pH": pd.to_numeric(df1["Ph"], errors="coerce").fillna(6.5),
        "Temperature": df1["Temperature"].fillna(25.0),
        "Humidity": df1["Humidity"].fillna(60.0),
        "Rainfall": df1["Rainfall"].fillna(100.0),
        "SoilType": df1["SoilType"],
        "Crop": df1["label"].str.strip().str.title()
    })
    
    records = [clean_df1]
    if os.path.exists(dataset2_path):
        df2 = pd.read_csv(dataset2_path)
        df2.columns = [c.strip() for c in df2.columns]
        clean_df2 = pd.DataFrame({
            "N": pd.to_numeric(df2["N"], errors="coerce").fillna(50),
            "P": pd.to_numeric(df2["P"], errors="coerce").fillna(30),
            "K": pd.to_numeric(df2["K"], errors="coerce").fillna(150),
            "pH": pd.to_numeric(df2["ph"], errors="coerce").fillna(6.5),
            "Temperature": 27.0,
            "Humidity": 65.0,
            "Rainfall": 120.0,
            "SoilType": "red",
            "Crop": df2["label"].str.strip().str.title()
        })
        records.append(clean_df2)

    combined_df = pd.concat(records, ignore_index=True)
    print("Total Combined Training Samples:", len(combined_df))
    print("Unique Crops:", combined_df["Crop"].nunique())

    soil_encoder = LabelEncoder()
    combined_df["SoilType_Encoded"] = soil_encoder.fit_transform(combined_df["SoilType"].astype(str))
    
    crop_encoder = LabelEncoder()
    combined_df["Crop_Encoded"] = crop_encoder.fit_transform(combined_df["Crop"])

    feature_cols = ["N", "P", "K", "pH", "Temperature", "Humidity", "Rainfall", "SoilType_Encoded"]
    X = combined_df[feature_cols]
    y = combined_df["Crop_Encoded"]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    print("Training Random Forest Crop Classifier...")
    model = RandomForestClassifier(n_estimators=150, max_depth=16, random_state=42, n_jobs=-1)
    model.fit(X_train_scaled, y_train)

    y_pred = model.predict(X_test_scaled)
    acc = accuracy_score(y_test, y_pred)
    print("Crop Model Test Accuracy: {:.2f}%".format(acc * 100))

    model_bundle = {
        "model": model,
        "scaler": scaler,
        "soil_encoder": soil_encoder,
        "crop_encoder": crop_encoder,
        "feature_cols": feature_cols,
        "crop_classes": list(crop_encoder.classes_),
        "soil_classes": list(soil_encoder.classes_)
    }

    bundle_path = os.path.join(model_output_dir, "crop_recommender.joblib")
    joblib.dump(model_bundle, bundle_path)
    print("Model bundle saved to:", bundle_path)

if __name__ == "__main__":
    train_crop_model()
