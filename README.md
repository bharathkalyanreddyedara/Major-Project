# Multimodal Generative AI System for Adaptive Crop Planning and Proactive Farm Intelligence

A complete, end-to-end intelligent agricultural decision-support system built for Major Project Submission (A. Y. 2026-2027).

---

## 🌟 Key Architecture & Capabilities

1. **Multimodal Soil Image Verification (Vision AI & Transfer Learning)**
   - Classifies soil photos across 7 key classes (`Alluvial_Soil`, `Arid_Soil`, `Black_Soil`, `Laterite_Soil`, `Mountain_Soil`, `Red_Soil`, `Yellow_Soil`).
   - Extracts 133 multi-space color distributions (RGB, HSV, LAB) and spatial gradient textures.
2. **Precision Crop & Fertilizer Recommendation (Multi-Model ML Benchmarked)**
   - Allows farmers to input manual lab soil test properties ($N, P, K, \text{pH}$, Moisture, Zinc, Sulphur, EC).
   - Combines manual values with live OpenWeather data and detected soil types to recommend optimal crops with **99.55% accuracy**.
3. **Automated Crop Lifecycle Timeline & Stage-Aware Planner**
   - Dynamic schedule generator tracking days from sowing to harvest.
   - Stage-by-stage irrigation routines, fertilizer top-dressing intervals, and pest/disease monitoring alerts.
4. **Proactive Notification Engine**
   - Alerts farmers ahead of time for upcoming field operations or extreme weather triggers (rain, heatwaves).
5. **Generative AI Farm Knowledge Assistant (Gemini API + RAG)**
   - Conversational AI powered by Retrieval-Augmented Generation (RAG) over verified agronomic documents, injecting active crop and weather context.

---

## 📊 State-of-the-Art Multi-Model Benchmark Leaderboards

All models were evaluated across diverse architectures on hold-out test sets:

### 1. 🌾 Crop Recommendation Leaderboard (22 Crops, Precision Agro-Climatic Dataset)
| Rank | Model Architecture | Test Accuracy | Weighted F1-Score | Status |
| :---: | :--- | :---: | :---: | :--- |
| 🥇 | **Ensemble (Random Forest + ExtraTrees)** | **99.55%** | **0.9955** | 🏆 **Best Model (Saved)** |
| 🥇 | **Random Forest Classifier** | **99.55%** | **0.9955** | High Precision |
| 🥇 | **ExtraTrees Classifier** | **99.55%** | **0.9955** | Extreme Tree Splitter |
| 🥈 | **LightGBM Classifier** | **98.86%** | **0.9886** | Fast Gradient Boosting |
| 🥈 | **XGBoost Classifier** | **98.86%** | **0.9884** | Extreme Gradient Boost |
| 🥉 | **MLP Deep Neural Net** | **98.41%** | **0.9841** | Tabular Neural Network |

---

### 2. 🧪 Fertilizer Recommendation Leaderboard (7 Fertilizer Classes, Precision Dataset)
| Rank | Model Architecture | Test Accuracy | Weighted F1-Score | Status |
| :---: | :--- | :---: | :---: | :--- |
| 🥇 | **XGBoost Classifier** | **100.00%** | **1.0000** | 🏆 **Best Model (Saved)** |
| 🥇 | **LightGBM Classifier** | **100.00%** | **1.0000** | Perfect Generalization |
| 🥈 | **Random Forest Classifier** | **99.67%** | **0.9966** | High Robustness |
| 🥉 | **MLP Deep Neural Net** | **92.50%** | **0.9250** | Dense Architecture |
| 4 | **ExtraTrees Classifier** | **91.00%** | **0.9090** | Robust Splitter |

---

### 3. 📷 Soil Image Classification Leaderboard (`Orignal-Dataset` - 1,189 Images, 7 Classes)
| Rank | Model Architecture | Test Accuracy | Weighted F1-Score | Status |
| :---: | :--- | :---: | :---: | :--- |
| 🥇 | **Ensemble (LightGBM + Random Forest)** | **84.45%** | **0.8365** | 🏆 **Best Model (Saved)** |
| 🥈 | **LightGBM Classifier** | 83.61% | 0.8278 | Top Standalone Model |
| 🥉 | **Random Forest Classifier** | 82.35% | 0.8185 | High Precision |
| 4 | **ExtraTrees Classifier** | 81.93% | 0.8123 | Robust Feature Splitter |
| 5 | **XGBoost Classifier** | 80.67% | 0.8001 | Solid Baseline |
| 6 | **MLP Deep Neural Net** | 79.83% | 0.7928 | Dense Tabular Embeddings |

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
