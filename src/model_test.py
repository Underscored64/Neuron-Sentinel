import joblib
import pandas as pd

model = joblib.load("models/flood_risk_model.joblib")
label_map = joblib.load("models/label_map.joblib")
reverse_map = {v: k for k, v in label_map.items()}

# Cas 1 : zone à fort risque structurel + grosse pluie → Notre attente ici c'est : high
sample_high = pd.DataFrame([{"FRI": 0.634, "rainfall_1h": 45, "rainfall_6h": 120, "humidity": 91}])
pred = model.predict(sample_high)[0]
proba = model.predict_proba(sample_high)[0]
print("Cas fort risque:", reverse_map[pred], dict(zip([reverse_map[i] for i in range(3)], proba)))

# Cas 2 : zone à faible risque structurel + pas de pluie → Notre attente : Low
sample_low = pd.DataFrame([{"FRI": 0.22, "rainfall_1h": 0, "rainfall_6h": 0, "humidity": 60}])
pred = model.predict(sample_low)[0]
print("Cas faible risque:", reverse_map[pred])
