import os
import joblib
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

def train_fertilizer_model(
    dataset_path="backend/data/Crop_vs_Fertilizer.csv",
    model_output_dir="backend/models"
):
    if not os.path.exists(dataset_path) and os.path.exists("Crop_vs_Fertilizer.csv"):
        dataset_path = "Crop_vs_Fertilizer.csv"

    print("Loading Fertilizer dataset from:", dataset_path)
    df = pd.read_csv(dataset_path)
    df.columns = [c.strip() for c in df.columns]

    soil_encoder = LabelEncoder()
    df["Soil_Type_Enc"] = soil_encoder.fit_transform(df["Soil Type"].astype(str).str.strip().str.title())

    crop_encoder = LabelEncoder()
    df["Crop_Type_Enc"] = crop_encoder.fit_transform(df["Crop Type"].astype(str).str.strip().str.title())

    target_encoder = LabelEncoder()
    df["Fertilizer_Enc"] = target_encoder.fit_transform(df["Fertilizer Name"].astype(str).str.strip())

    feature_cols = ["Temparature", "Humidity", "Moisture", "Soil_Type_Enc", "Crop_Type_Enc", "Nitrogen", "Potassium", "Phosphorous"]
    
    X = df[feature_cols]
    y = df["Fertilizer_Enc"]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    print("Training Fertilizer Classifier...")
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train_scaled, y_train)

    y_pred = model.predict(X_test_scaled)
    acc = accuracy_score(y_test, y_pred)
    print("Fertilizer Model Accuracy: {:.2f}%".format(acc * 100))

    os.makedirs(model_output_dir, exist_ok=True)
    bundle = {
        "model": model,
        "scaler": scaler,
        "soil_encoder": soil_encoder,
        "crop_encoder": crop_encoder,
        "target_encoder": target_encoder,
        "fertilizer_classes": list(target_encoder.classes_),
        "soil_classes": list(soil_encoder.classes_),
        "crop_classes": list(crop_encoder.classes_),
        "feature_cols": feature_cols
    }

    bundle_path = os.path.join(model_output_dir, "fertilizer_recommender.joblib")
    joblib.dump(bundle, bundle_path)
    print("Fertilizer bundle saved to:", bundle_path)

if __name__ == "__main__":
    train_fertilizer_model()
