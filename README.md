# Multimodal Generative AI System for Adaptive Crop Planning and Proactive Farm Intelligence

A complete, end-to-end intelligent agricultural decision-support system built for Major Project Submission (A. Y. 2026-2027).

---

## 🌟 Key Architecture & Capabilities

1. **Multimodal Soil Image Verification (Vision AI & Transfer Learning)**
   - Classifies soil photos across 7 key classes (`Alluvial_Soil`, `Arid_Soil`, `Black_Soil`, `Laterite_Soil`, `Mountain_Soil`, `Red_Soil`, `Yellow_Soil`).
   - Extracts 133 multi-space color distributions (RGB, HSV, LAB) and spatial gradient textures.
2. **Precision Crop & Fertilizer Recommendation (Multi-Model ML Benchmarked)**
   - Allows farmers to input manual lab soil test properties ($N, P, K, \text{pH}$, Moisture, Zinc, Sulphur, EC).
   - Trained with realistic field sensor jitter & 5-Fold Cross-Validation to eliminate overfitting.
3. **Automated Crop Lifecycle Timeline & Stage-Aware Planner**
   - Dynamic schedule generator tracking days from sowing to harvest.
   - Stage-by-stage irrigation routines, fertilizer top-dressing intervals, and pest/disease monitoring alerts.
4. **Proactive Notification Engine**
   - Alerts farmers ahead of time for upcoming field operations or extreme weather triggers (rain, heatwaves).
5. **Generative AI Farm Knowledge Assistant (Gemini API + RAG)**
   - Conversational AI powered by Retrieval-Augmented Generation (RAG) over verified agronomic documents, injecting active crop and weather context.

---

## 📊 Regularized Benchmark Results (5-Fold Stratified Cross-Validation)

All models include $L_1/L_2$ regularization, tree depth constraints, and simulated sensor noise ($\pm 5-8\%$) to ensure robust real-world generalization:

### 1. 🌾 Crop Recommendation Module (22 Crops)
| Rank | Model Architecture | Test Accuracy | 5-Fold CV Accuracy | Weighted F1 | Status |
| :---: | :--- | :---: | :---: | :---: | :--- |
| 🥇 | **ExtraTrees (Constrained)** | **98.64%** | **98.35%** | **0.9863** | 🏆 **Deployed in Backend** |
| 🥈 | **RandomForest (Constrained)** | **97.95%** | 98.41% | 0.9794 | Robust Bagging |
| 🥉 | **LightGBM (Regularized)** | **96.59%** | 97.27% | 0.9657 | Regularized GBDT |
| 4 | **XGBoost (Regularized)** | **95.45%** | 97.33% | 0.9545 | $L_1/L_2$ Penalized |
| 5 | **MLP Neural Net (Dropout)** | **95.45%** | 95.97% | 0.9541 | Deep Tabular |

---

### 2. 🧪 Fertilizer Recommendation Module (7 Fertilizer Classes)
| Rank | Model Architecture | Test Accuracy | 5-Fold CV Accuracy | Weighted F1 | Status |
| :---: | :--- | :---: | :---: | :---: | :--- |
| 🥇 | **LightGBM (Regularized)** | **92.33%** | **93.25%** | **0.9233** | 🏆 **Deployed in Backend** |
| 🥈 | **RandomForest (Constrained)** | **92.33%** | 92.67% | 0.9233 | Robust Bagging |
| 🥉 | **XGBoost (Regularized)** | **91.33%** | 93.50% | 0.9133 | $L_1/L_2$ Penalized |
| 4 | **MLP Neural Net (Dropout)** | **88.83%** | 87.37% | 0.8885 | Dense Regularized |
| 5 | **ExtraTrees (Constrained)** | **81.50%** | 81.46% | 0.7992 | Random Splitter |

---

### 3. 📷 Soil Image Classification Module (`Orignal-Dataset` - 1,189 Images, 7 Classes)
| Rank | Model Architecture | Test Accuracy | Weighted F1 | Status |
| :---: | :--- | :---: | :---: | :--- |
| 🥇 | **Ensemble (LightGBM + Random Forest)** | **84.45%** | **0.8365** | 🏆 **Deployed in Backend** |
| 🥈 | **LightGBM Classifier** | 83.61% | 0.8278 | Top Standalone Model |
| 🥉 | **Random Forest Classifier** | 82.35% | 0.8185 | High Precision |
| 4 | **ExtraTrees Classifier** | 81.93% | 0.8123 | Robust Feature Splitter |
| 5 | **XGBoost Classifier** | 80.67% | 0.8001 | Solid Baseline |

---

## ⚡ How to Run Benchmarks & Retrain Models

### 1. Run Robust Cross-Validated Benchmark:
```bash
python backend/ml/train_regularized_robust_models.py
```

### 2. Run Soil Image Multi-Model Benchmark:
```bash
python backend/ml/train_soil_vision.py
```

---

## 🚀 How to Run the Application

### 1. Start the FastAPI Backend Server
```bash
uvicorn backend.app.main:app --reload --port 8000
```
- Interactive Swagger UI: `http://localhost:8000/docs`

### 2. Start the React Frontend Dashboard
```bash
cd frontend
npm install
npm start
```
- Web Application: `http://localhost:3000`
