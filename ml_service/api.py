# ml_service/api.py
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import joblib
import numpy as np
from pathlib import Path

from fertilizer import get_fertilizer_recommendation   # ✅ NEW

# ---------- Paths ----------
APP_DIR = Path(__file__).resolve().parent
MODELS_DIR = APP_DIR / "models"

MODEL_FILE = MODELS_DIR / "ensemble_model.pkl"
SCALER_FILE = MODELS_DIR / "scaler.pkl"
ENCODER_FILE = MODELS_DIR / "label_encoder.pkl"

# ---------- Load Artifacts ----------
model = joblib.load(MODEL_FILE)
scaler = joblib.load(SCALER_FILE)
label_encoder = joblib.load(ENCODER_FILE)

# ---------- App Init ----------
app = FastAPI(title="AgroAware ML Service", version="2.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------- Schemas ----------
class CropInput(BaseModel):
    N: float
    P: float
    K: float
    ph: float
    temperature: float
    rainfall: float

class FertilizerInput(BaseModel):
    crop: str
    N: float
    P: float
    K: float


# ---------- Routes ----------
@app.get("/")
def home():
    return {"status": "✅ ML Service Running", "model": "Ensemble Soft Voting"}


@app.post("/predict")
def predict(data: CropInput):
    arr = np.array([[data.N, data.P, data.K, data.ph, data.temperature, data.rainfall]])
    arr_scaled = scaler.transform(arr)
    
    # Get probabilities for all classes (soft voting gives us probs)
    probs = model.predict_proba(arr_scaled)[0]  # shape: (num_classes,)
    
    # Get top-3 predictions
    top3_indices = np.argsort(-probs)[:3]  # descending order
    
    top3_crops = []
    for idx in top3_indices:
        crop_name = label_encoder.inverse_transform([idx])[0]
        confidence = float(probs[idx]) * 100  # Convert to percentage
        top3_crops.append({
            "crop": crop_name,
            "confidence": round(confidence, 1),
            "reason": f"Optimal for NPK={data.N:.0f},{data.P:.0f},{data.K:.0f}, pH={data.ph}, Temp={data.temperature}°C, Rain={data.rainfall}mm"
        })
    
    # Return top-1 as primary + all top-3 for explainability
    return {
        "predicted_crop": top3_crops[0]["crop"],
        "confidence": top3_crops[0]["confidence"],
        "top_3": top3_crops  # New field for explainability
    }


@app.post("/fertilizer")
def fertilizer(data: FertilizerInput):
    result = get_fertilizer_recommendation(data.crop, data.N, data.P, data.K)
    return result
