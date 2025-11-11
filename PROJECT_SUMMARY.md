# AgroAware – A Generative AI-Based Farming Advisor and Awareness Platform

## Problem Statement
Indian farmers often lack access to scientific, data-driven advice for crop selection, fertilizer use, and government schemes. This leads to suboptimal yields, wasted resources, and missed opportunities. AgroAware bridges this gap by providing a digital platform that gives personalized, actionable recommendations using machine learning and generative AI.

---

## Project Overview
AgroAware is a full-stack platform that helps farmers make better decisions by:
- Recommending the best crop for their soil and climate
- Suggesting fertilizer dosages
- Providing weather-based advice
- Summarizing government schemes
- Supporting both expert and beginner users

---

## System Architecture (A to Z)

### 1. Data Layer
- **Datasets:** Agricultural CSVs with N, P, K, pH, temperature, rainfall, crop label
- **Preprocessing:** Cleaning, scaling, encoding, train/test split

### 2. Machine Learning Layer
- **Models Trained:**
  - Decision Tree
  - Random Forest
  - XGBoost
  - Logistic Regression
- **Ensemble:** Weighted Soft Voting Ensemble (all 4 models)
- **Artifacts Saved:** model.pkl, scaler.pkl, label_encoder.pkl, ensemble_model.pkl
- **Metrics:** Accuracy, Precision, Recall, F1-score (macro, per-crop)

### 3. ML Microservice (Python FastAPI)
- **Endpoints:**
  - `/predict` (crop recommendation)
  - `/fertilizer` (fertilizer recommendation)
  - `/weather` (fetch real-time weather)
- **Input:** N, P, K, pH, Temperature, Rainfall
- **Output:** Recommended Crop, Fertilizer Dosage

### 4. Backend API (Node.js + Express)
- **Bridges frontend and ML microservice**
- **Handles:**
  - User authentication (MongoDB)
  - Logging and analytics
  - Routing requests to ML service

### 5. Frontend (React + Tailwind)
- **Dual-Mode UI:**
  - Expert Mode: For users with soil test data
  - Beginner Mode: For users without soil test (State → District → Season → Crops)
- **Features:**
  - Login/Signup
  - Crop recommendation form
  - Fertilizer advisory
  - Weather auto-fill
  - Results dashboard

### 6. Database (MongoDB)
- **Stores:**
  - User accounts
  - Advisory logs
  - Saved recommendations

### 7. Integrations
- **Weather API:** Real-time district weather
- **Fertilizer DB:** Crop- and soil-specific fertilizer lookup

---

## Features Implemented (Till Now)
- Data cleaning, scaling, encoding
- Training and evaluation of 4 ML models
- Weighted ensemble model for robust predictions
- Fertilizer recommendation system
- Weather API integration
- Full-stack integration (React, Node.js, Python, MongoDB)
- Dual-mode UI (Expert/Beginner)
- Secure authentication
- User advisory history

---

## Results (Sample)
- **Best Model Accuracy:** ~95% (Decision Tree, Random Forest, XGBoost)
- **Ensemble Model:** Slightly higher stability and macro-F1
- **API Response Time:** <1s for prediction
- **User Feedback:** UI is simple, both expert and beginner flows tested
- **Reports Generated:** Model comparison, parameters, detailed metrics

---

## What We Are Doing Next (Upcoming Features)
- Voice-based assistance (STT/TTS)
- Govt. scheme summarizer (RAG-based)
- Awareness poster generator (GenAI)
- Mobile app and offline support

---

## Project Value
AgroAware empowers farmers with:
- Scientific, data-backed crop recommendations
- Personalized advice based on soil, climate, and season
- Easy access to government schemes and best practices
- A user-friendly interface for both experts and beginners

---

## One-Line Summary (For Reports/Viva)
> “AgroAware helps farmers choose the most suitable crop and farming practices using machine learning, ensemble models, and a dual-mode advisory interface.”

---

## System Flow Diagram (Textual)
1. **User opens web app**
2. **Logs in / signs up**
3. **Chooses mode:**
   - Expert: enters N, P, K, pH, Temp, Rainfall
   - Beginner: selects State, District, Season
4. **Frontend sends request to backend**
5. **Backend authenticates and logs request**
6. **Backend calls ML microservice**
7. **ML service preprocesses input, runs ensemble model**
8. **Returns crop recommendation (+ fertilizer, weather)**
9. **Frontend displays result and saves to user history**

---

## Key Takeaways
- End-to-end working system (data → ML → API → UI)
- Ensemble model for robust, reliable crop prediction
- Fertilizer and weather integration for actionable advice
- Dual-mode UI for all user types
- Modular, extensible architecture for future GenAI features

---

## For More Details
- See code in `/ml_service`, `/server`, `/client`
- Reports and model artifacts in `/ml_service/reports` and `/ml_service/models`
- This summary can be used for reports, presentations, and viva answers.

---

## Dataset Source
- The primary crop recommendation dataset was taken from Kaggle (https://www.kaggle.com/datasets/atharvaingle/crop-recommendation-dataset).
- Additional fertilizer and weather mapping data was curated and cleaned as needed for the project.
