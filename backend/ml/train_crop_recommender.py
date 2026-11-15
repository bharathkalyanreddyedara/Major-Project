"""
Crop recommendation model training and evaluation.

Model selection is performed using 5-fold stratified cross-validation
on the training subset. The held-out test subset is used only once
for final evaluation of the selected model.
"""

import os
import joblib
import pandas as pd
import numpy as np
from sklearn.base import clone
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_validate
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier, VotingClassifier
from sklearn.metrics import accuracy_score, f1_score
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier


def train_crop_model(
    dataset_path="backend/data/Crop_recommendation.csv",
    model_output_dir="backend/models"
):
    print("=" * 68)
    print("CROP RECOMMENDATION: CROSS-VALIDATED MODEL SELECTION")
    print("=" * 68)

    if not os.path.exists(dataset_path):
        url = "https://raw.githubusercontent.com/Gladiator07/Harvestify/master/Data-processed/crop_recommendation.csv"
        import urllib.request
        os.makedirs(os.path.dirname(dataset_path) or ".", exist_ok=True)
        urllib.request.urlretrieve(url, dataset_path)

    df = pd.read_csv(dataset_path)
    df.columns = [c.strip() for c in df.columns]

    # Standardize column names.
    col_map = {
        "N": "N", "P": "P", "K": "K",
        "temperature": "Temperature",
        "humidity": "Humidity",
        "ph": "pH",
        "rainfall": "Rainfall",
        "label": "Crop"
    }
    for old_col, new_col in col_map.items():
        if old_col in df.columns and new_col not in df.columns:
            df.rename(columns={old_col: new_col}, inplace=True)

    required_cols = ["N", "P", "K", "Temperature", "Humidity", "pH", "Rainfall", "Crop"]
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        raise ValueError(f"Dataset is missing required columns: {missing_cols}")

    # Synthetic pH-derived category used by the existing feature pipeline.
    df["SoilType_Encoded"] = np.where(
        df["pH"] < 6.0, 0,
        np.where(df["pH"] > 7.5, 2, 1)
    )

    # Feature engineering. The epsilon prevents division by zero.
    df["N_to_P"] = df["N"] / (df["P"] + 1e-4)
    df["N_to_K"] = df["N"] / (df["K"] + 1e-4)
    df["P_to_K"] = df["P"] / (df["K"] + 1e-4)
    df["NPK_Sum"] = df["N"] + df["P"] + df["K"]

    crop_encoder = LabelEncoder()
    df["Crop_Encoded"] = crop_encoder.fit_transform(
        df["Crop"].astype(str).str.title()
    )

    soil_classes = ["Alluvial", "Black", "Red", "Laterite", "Arid", "Mountain", "Yellow"]
    soil_encoder = LabelEncoder()
    soil_encoder.fit(soil_classes)

    feature_cols = [
        "N", "P", "K", "pH", "Temperature", "Humidity", "Rainfall",
        "SoilType_Encoded", "N_to_P", "N_to_K", "P_to_K", "NPK_Sum"
    ]
    X = df[feature_cols]
    y = df["Crop_Encoded"]

    # Preserve an independent stratified 20% test set.
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # Fit preprocessing only on the training subset to avoid leakage.
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    base_models = {
        "RandomForest": RandomForestClassifier(
            n_estimators=250, max_depth=20, random_state=42, n_jobs=-1
        ),
        "ExtraTrees": ExtraTreesClassifier(
            n_estimators=250, max_depth=20, random_state=42, n_jobs=-1
        ),
        "XGBoost": XGBClassifier(
            n_estimators=250, learning_rate=0.08, max_depth=6,
            random_state=42, n_jobs=-1, eval_metric="mlogloss"
        ),
        "LightGBM": LGBMClassifier(
            n_estimators=250, learning_rate=0.08, max_depth=6,
            random_state=42, n_jobs=-1, verbose=-1
        ),
        "MLP Neural Net": MLPClassifier(
            hidden_layer_sizes=(256, 128), max_iter=400,
            random_state=42, early_stopping=True
        )
    }

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    scoring = {"accuracy": "accuracy", "weighted_f1": "f1_weighted"}
    cv_results = {}

    print(f"\nDataset: {len(df)} records across {df['Crop'].nunique()} crop classes")
    print(f"Training subset: {len(X_train)} | Held-out test subset: {len(X_test)}")
    print("\nFive-fold stratified cross-validation on training data:")
    print(f"{'Model':<20} | {'Mean accuracy':>14} | {'Std. dev.':>10} | {'Mean weighted F1':>17}")
    print("-" * 73)

    for name, estimator in base_models.items():
        scores = cross_validate(
            estimator,
            X_train_scaled,
            y_train,
            cv=cv,
            scoring=scoring,
            n_jobs=1,
            return_train_score=False
        )
        cv_results[name] = {
            "accuracy_mean": float(np.mean(scores["test_accuracy"])),
            "accuracy_std": float(np.std(scores["test_accuracy"], ddof=1)),
            "weighted_f1_mean": float(np.mean(scores["test_weighted_f1"])),
            "weighted_f1_std": float(np.std(scores["test_weighted_f1"], ddof=1))
        }
        print(
            f"{name:<20} | "
            f"{cv_results[name]['accuracy_mean'] * 100:>13.2f}% | "
            f"{cv_results[name]['accuracy_std'] * 100:>9.2f}% | "
            f"{cv_results[name]['weighted_f1_mean']:>17.4f}"
        )

    # Select the two base estimators using training-only CV results.
    ranked_base_models = sorted(
        cv_results.items(),
        key=lambda item: item[1]["accuracy_mean"],
        reverse=True
    )
    top_two_names = [ranked_base_models[0][0], ranked_base_models[1][0]]

    ensemble = VotingClassifier(
        estimators=[
            (name, clone(base_models[name])) for name in top_two_names
        ],
        voting="soft",
        n_jobs=-1
    )

    # Evaluate the ensemble with the same training-only CV protocol.
    ensemble_scores = cross_validate(
        ensemble,
        X_train_scaled,
        y_train,
        cv=cv,
        scoring=scoring,
        n_jobs=1,
        return_train_score=False
    )
    cv_results["SoftVotingEnsemble"] = {
        "accuracy_mean": float(np.mean(ensemble_scores["test_accuracy"])),
        "accuracy_std": float(np.std(ensemble_scores["test_accuracy"], ddof=1)),
        "weighted_f1_mean": float(np.mean(ensemble_scores["test_weighted_f1"])),
        "weighted_f1_std": float(np.std(ensemble_scores["test_weighted_f1"], ddof=1))
    }

    print(
        f"{'SoftVotingEnsemble':<20} | "
        f"{cv_results['SoftVotingEnsemble']['accuracy_mean'] * 100:>13.2f}% | "
        f"{cv_results['SoftVotingEnsemble']['accuracy_std'] * 100:>9.2f}% | "
        f"{cv_results['SoftVotingEnsemble']['weighted_f1_mean']:>17.4f}"
    )

    # Select the final candidate using mean CV accuracy only.
    candidate_estimators = {
        **{name: estimator for name, estimator in base_models.items()},
        "SoftVotingEnsemble": ensemble
    }
    best_name = max(
        cv_results,
        key=lambda name: cv_results[name]["accuracy_mean"]
    )
    best_model = clone(candidate_estimators[best_name])

    # Refit the selected estimator on all training data, then evaluate once
    # on the independent test subset.
    best_model.fit(X_train_scaled, y_train)
    test_predictions = best_model.predict(X_test_scaled)
    test_accuracy = accuracy_score(y_test, test_predictions)
    test_weighted_f1 = f1_score(
        y_test, test_predictions, average="weighted"
    )

    print("\nFinal evaluation on the untouched held-out test subset:")
    print(f"Selected model: {best_name}")
    print(f"Test accuracy: {test_accuracy * 100:.2f}%")
    print(f"Test weighted F1: {test_weighted_f1:.4f}")
    print("=" * 68)

    os.makedirs(model_output_dir, exist_ok=True)
    bundle = {
        "model": best_model,
        "scaler": scaler,
        "crop_encoder": crop_encoder,
        "soil_encoder": soil_encoder,
        "feature_cols": feature_cols,
        "crop_classes": list(crop_encoder.classes_),
        "soil_classes": list(soil_encoder.classes_),
        "best_model_name": best_name,
        "accuracy": float(test_accuracy),
        "test_accuracy": float(test_accuracy),
        "test_weighted_f1": float(test_weighted_f1),
        "cv_protocol": "5-fold StratifiedKFold on training subset only",
        "cv_n_splits": 5,
        "cv_results": cv_results,
        "leaderboard": cv_results,
        "training_samples": int(len(X_train)),
        "testing_samples": int(len(X_test)),
        "random_state": 42
    }

    bundle_path = os.path.join(model_output_dir, "crop_recommender.joblib")
    joblib.dump(bundle, bundle_path, compress=3)
    print(f"Saved trained crop recommendation model to: {bundle_path}")


if __name__ == "__main__":
    train_crop_model()
