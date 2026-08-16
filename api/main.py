from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import joblib
import pandas as pd
import geopandas as gpd

app = FastAPI(title="Neuron Sentinel - Flood Risk Prediction API")

model = joblib.load("models/flood_risk_model.joblib")
label_map = joblib.load("models/label_map.joblib")
reverse_map = {v: k for k, v in label_map.items()}

fri_lookup = gpd.read_file("data/processed/lome_fri.gpkg")[['canton_nom', 'FRI']].set_index('canton_nom')['FRI'].to_dict()

class PredictRequest(BaseModel):
    zone_id: str
    rainfall_1h: float
    rainfall_6h: float
    humidity: float

class PredictResponse(BaseModel):
    zone_id: str
    flood_probability: float
    risk_level: str
    prediction_horizon: str

@app.post("/predict", response_model=PredictResponse)
def predict(req: PredictRequest):
    if req.zone_id not in fri_lookup:
        raise HTTPException(status_code=404, detail=f"Zone inconnue: {req.zone_id}")

    X = pd.DataFrame([{
        "FRI": fri_lookup[req.zone_id],
        "rainfall_1h": req.rainfall_1h,
        "rainfall_6h": req.rainfall_6h,
        "humidity": req.humidity
    }])

    proba = model.predict_proba(X)[0]
    pred_class = model.predict(X)[0]

    return PredictResponse(
        zone_id=req.zone_id,
        flood_probability=round(float(max(proba)), 2),
        risk_level=reverse_map[pred_class].upper(),
        prediction_horizon="6h"
    )
