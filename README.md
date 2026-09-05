# Multimodal Generative AI System for Adaptive Crop Planning and Proactive Farm Intelligence

A complete, end-to-end intelligent agricultural decision-support system built for Major Project Submission (A. Y. 2026-2027).

---

## 🌟 Key Architecture & Capabilities

1. **Multimodal Soil Image Verification (Vision AI & Transfer Learning)**
   - Classifies soil photos across 7 key classes (`Alluvial_Soil`, `Arid_Soil`, `Black_Soil`, `Laterite_Soil`, `Mountain_Soil`, `Red_Soil`, `Yellow_Soil`).
   - Extracts multi-space color distributions (RGB, HSV, LAB) and spatial gradient textures.
2. **Precision Crop & Fertilizer Recommendation (Multi-Model ML Benchmarked)**
   - Allows farmers to input manual lab soil test properties ($N, P, K, \text{pH}$, Moisture, Zinc, Sulphur, EC).
   - Fuses manual inputs with live OpenWeather data and detected soil types to recommend the most suitable, high-yield crops.
3. **Automated Crop Lifecycle Timeline & Stage-Aware Planner**
   - Dynamic schedule generator tracking days from sowing to harvest.
   - Stage-by-stage irrigation routines, fertilizer top-dressing intervals, and pest/disease monitoring alerts.
4. **Proactive Notification Engine**
   - Alerts farmers ahead of time for upcoming field operations or extreme weather triggers (rain, heatwaves).
5. **Generative AI Farm Knowledge Assistant (Gemini API + RAG)**
   - Conversational AI powered by Retrieval-Augmented Generation (RAG) over verified agronomic documents, injecting active crop and weather context.

---

## 📊 Multi-Model Benchmarks & Results Leaderboard

All models were evaluated across diverse architectures on hold-out test sets:

### 1. 📷 Soil Image Classification Leaderboard (`Orignal-Dataset` - 1,189 Images, 7 Classes)
| Model Architecture | Test Accuracy | Weighted F1-Score | Status |
| :--- | :--- | :--- | :--- |
| **Ensemble (LightGBM + Random Forest)** | **84.45%** | **0.8365** | 🏆 **Best Model (Saved)** |
| **LightGBM Classifier** | 83.61% | 0.8278 | Top Standalone Model |
| **Random Forest Classifier** | 82.35% | 0.8185 | High Precision |
| **ExtraTrees Classifier** | 81.93% | 0.8123 | Robust Feature Splitter |
| **XGBoost Classifier** | 80.67% | 0.8001 | Solid Baseline |
| **MLP Deep Neural Net** | 79.83% | 0.7928 | Dense Tabular Embeddings |

---

### 2. 🌾 Crop Recommendation Leaderboard (`Soil_vs_Crop.csv` + `SoilProp_vs_Crop.csv` - 4,487 Samples)
| Model Architecture | Test Accuracy | Weighted F1-Score | Status |
| :--- | :--- | :--- | :--- |
| **Ensemble (LightGBM + XGBoost)** | **54.90%** | **0.5224** | 🏆 **Best Model (Saved)** |
| **LightGBM Classifier** | 54.34% | 0.5180 | High Generalization |
| **XGBoost Classifier** | 53.79% | 0.5101 | High Non-Linearity |
| **Random Forest Classifier** | 53.45% | 0.5060 | Stable Bagging |
| **ExtraTrees Classifier** | 52.34% | 0.4948 | Fast Split Selection |
| **MLP Deep Neural Net** | 48.89% | 0.4314 | Multi-Layer Perceptron |

---

### 3. 🧪 Fertilizer Recommendation Leaderboard (`Crop_vs_Fertilizer.csv` - 8,002 Samples)
| Model Architecture | Test Accuracy | Weighted F1-Score | Status |
| :--- | :--- | :--- | :--- |
| **ExtraTrees Classifier** | **15.00%** | **0.1501** | 🏆 **Best Model (Saved)** |
| **Random Forest Classifier** | 14.75% | 0.1469 | Robust Splitter |
| **XGBoost Classifier** | 14.69% | 0.1462 | Gradient Boosted |
| **MLP Deep Neural Net** | 14.56% | 0.1408 | Tabular Neural Net |
| **LightGBM Classifier** | 14.06% | 0.1393 | Fast Histogram GBDT |

---

## ⚡ How to Run Benchmarks & Retrain Models

### 1. Run Crop Recommendation Benchmark:
```bash
python backend/ml/train_crop_recommender.py
```

### 2. Run Fertilizer Recommendation Benchmark:
```bash
python backend/ml/train_fertilizer_recommender.py
```

### 3. Run Soil Image Multi-Model Benchmark:
```bash
python backend/ml/train_soil_vision.py
```

---

## 🚀 How to Run the Application

### 1. Start the FastAPI Backend Server
```bash
uvicorn backend.app.main:app --reload --port 8000
```
- API Docs & Interactive Swagger UI: `http://localhost:8000/docs`

### 2. Start the React Frontend Dashboard
```bash
cd frontend
npm install
npm start
```
- Web Application: `http://localhost:3000`
