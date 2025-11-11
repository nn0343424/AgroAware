# AgroAware: ML-Powered Crop Recommendation System

## Overview
AgroAware is an end-to-end platform that leverages machine learning to recommend optimal crops and fertilizers to farmers based on soil, weather, and seasonal data. It features a modern web interface, RESTful backend, and a Python-based ML microservice.

---

## Features
- **Crop Recommendation** using ensemble ML models (DecisionTree, RandomForest, LogisticRegression, XGBoost, Weighted Soft Voting)
- **Fertilizer Recommendation** based on soil and crop data
- **Weather & Seasonal Advisory**
- **User Authentication** (signup/login)
- **Advisory Log** for user queries
- **Modern React Frontend**
- **RESTful Node.js Backend**
- **Python FastAPI ML Microservice**
- **MongoDB Database**

---

## System Architecture
```
[React Client] ⇄ [Node.js Server] ⇄ [Python ML Service] ⇄ [ML Models/Data]
                        │
                  [MongoDB Database]
```

---

## Folder Structure
```
AgroAware/
├── client/         # React frontend
├── server/         # Node.js backend
├── ml_service/     # Python ML microservice
```

---

## Setup Instructions

### 1. Prerequisites
- Git
- Python 3.10+
- Node.js (v18+ recommended)
- MongoDB
- (Optional) WSL for Linux-like environment on Windows

### 2. Clone the Repository
```bash
git clone https://github.com/BhuvaneshAdithya45/AgroAware.git
cd AgroAware
```

### 3. Python ML Service
```bash
cd ml_service
python -m venv venv
# Windows:
venv\Scripts\activate
# Linux/WSL:
source venv/bin/activate
pip install -r requirements.txt
uvicorn api:app --host 0.0.0.0 --port 8800 --reload
```

### 4. Node.js Backend
```bash
cd server
npm install
npm start
```

### 5. React Frontend
```bash
cd client
npm install
npm start
```

### 6. MongoDB
- Start MongoDB service (or use MongoDB Atlas)
- Default connection: `mongodb://localhost:27017/agroaware`

---

## Usage
- Frontend: http://localhost:5173
- Backend: http://localhost:5000
- ML API: http://localhost:8800/docs (Swagger UI)

---

## Dataset Source
- [Kaggle Crop Recommendation Dataset](https://www.kaggle.com/datasets/atharvaingle/crop-recommendation-dataset)
- Curated fertilizer and weather data (see `ml_service/data/`)

---

## Results & Reports
- Model training and comparison reports: `ml_service/reports/`
- Ensemble model is used for predictions by default

---

## Security & .gitignore
- Sensitive files, large data, models, and reports are ignored by `.gitignore`
- Never commit personal access tokens or credentials

---

## Contributors
- Bhuvanesh Adithya Gowda


---

## License
[MIT License](LICENSE)

---

## Contact
For queries, contact: bhuvaneshadithya294@gmail.com
