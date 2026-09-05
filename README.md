# Multimodal Generative AI System for Adaptive Crop Planning and Proactive Farm Intelligence

A complete, end-to-end intelligent agricultural decision-support system built for Major Project Submission (2026-2027).

---

## 🌟 Key Features

1. **Multimodal Soil Image Verification (CNN)**
   - Classifies soil images across 7 key classes (`Alluvial_Soil`, `Arid_Soil`, `Black_Soil`, `Laterite_Soil`, `Mountain_Soil`, `Red_Soil`, `Yellow_Soil`) using Deep Transfer Learning.
2. **Precision Crop & Fertilizer Recommendation (Hybrid ML)**
   - Allows farmers to provide manual lab soil test properties ($N, P, K, \text{pH}$, Moisture, Zinc, Sulphur, EC).
   - Combines manual values with live weather (OpenWeather API) and detected soil types to rank the best high-yield crops.
3. **Automated Crop Lifecycle Timeline & Stage-Aware Planner**
   - Dynamic schedule generator spanning Sowing, Vegetative, Tillering/Flowering, and Maturity stages.
   - Tailored irrigation routines, fertilizer top-dressing intervals, and pest/disease warnings.
4. **Proactive Notification Engine**
   - Tracks crop age and alerts farmers ahead of time for upcoming field operations or extreme weather events.
5. **Generative AI Farm Knowledge Assistant (Gemini API + RAG)**
   - Conversational assistant powered by LangChain and Retrieval-Augmented Generation (RAG) over verified agronomic documents.

---

## 🛠️ Project Structure

```
.
├── backend/
│   ├── app/
│   │   ├── api/            # REST API Routes (Soil, Crop, Timeline, Notifications, Chat, Weather)
│   │   ├── schemas/        # Pydantic Request & Response Data Models
│   │   ├── services/       # Core Business Logic (Soil CNN, Crop ML, RAG Assistant, Weather)
│   │   ├── config.py       # Configuration & Environment Settings
│   │   └── main.py         # FastAPI App Entrypoint
│   ├── data/
│   │   ├── knowledge/      # Agronomy Documents for RAG Knowledge Retrieval
│   │   └── *.csv           # Soil, Crop, and Fertilizer Datasets
│   ├── ml/
│   │   ├── train_soil_cnn.py               # CNN Soil Image Classifier Training
│   │   ├── train_crop_recommender.py       # ML Crop Classifier Training
│   │   └── train_fertilizer_recommender.py # ML Fertilizer Classifier Training
│   └── models/             # Exported Model Artifacts (.joblib, .keras, .json)
├── frontend/               # Modern React.js Farmer Dashboard
│   ├── public/
│   ├── src/
│   │   ├── components/     # Navbar, Weather, SoilAnalysis, Timeline, Chat, Alerts
│   │   ├── App.js
│   │   └── App.css
│   └── package.json
├── Orignal-Dataset/        # Soil Image Dataset (7 Classes)
├── requirements.txt        # Python Dependencies
└── README.md
```

---

## ⚡ Installation & Training Guide

### 1. Install Python Dependencies
```bash
pip install -r requirements.txt
```

### 2. Train the Machine Learning & Deep Learning Models

- **Train Soil Image CNN Classifier:**
  ```bash
  python backend/ml/train_soil_cnn.py
  ```
- **Train Crop Recommendation Model:**
  ```bash
  python backend/ml/train_crop_recommender.py
  ```
- **Train Fertilizer Recommendation Model:**
  ```bash
  python backend/ml/train_fertilizer_recommender.py
  ```

---

## 🚀 Running the Application

### 1. Start the FastAPI Backend Server
```bash
uvicorn backend.app.main:app --reload --port 8000
```
- API Docs & Swagger UI: `http://localhost:8000/docs`

### 2. Start the React Frontend Dashboard
```bash
cd frontend
npm install
npm start
```
- Frontend UI: `http://localhost:3000`
