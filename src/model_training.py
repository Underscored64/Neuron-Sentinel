import pandas as pd
import geopandas as gpd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
from xgboost import XGBClassifier
import joblib
import numpy as np


df = pd.read_csv("data/processed/training_data.csv")

lome_fri = gpd.read_file("data/processed/lome_fri.gpkg")[['canton_nom', 'FRI']]
df = df.merge(lome_fri, on='canton_nom')

np.random.seed(42)
noise = np.random.normal(0, 0.05, size=len(df))
df['dynamic_score'] = df['FRI'] + (df['rainfall_6h'] / df['rainfall_6h'].max()) * 0.3 + noise
df['dynamic_risk'] = pd.qcut(df['dynamic_score'], q=3, labels=['low', 'medium', 'high'])

test_cantons = ['Baguida', 'Aflao-Gakli', 'Vakpossito']
train_df = df[~df['canton_nom'].isin(test_cantons)]
test_df = df[df['canton_nom'].isin(test_cantons)]

features = ['FRI', 'rainfall_1h', 'rainfall_6h', 'humidity']
X_train, y_train = train_df[features], train_df['dynamic_risk']
X_test, y_test = test_df[features], test_df['dynamic_risk']

rf = RandomForestClassifier(n_estimators=200, class_weight='balanced', random_state=42)
rf.fit(X_train, y_train)
print("=== Random Forest ===")
print(classification_report(y_test, rf.predict(X_test)))

label_map = {'low': 0, 'medium': 1, 'high': 2}
xgb = XGBClassifier(n_estimators=200, random_state=42)
xgb.fit(X_train, y_train.map(label_map))
print("=== XGBoost ===")
print(classification_report(y_test.map(label_map), xgb.predict(X_test)))

xgb.fit(X_train, y_train.map(label_map))
joblib.dump(xgb, "models/flood_risk_model.joblib")
joblib.dump(label_map, "models/label_map.joblib")
print("Modèle exporté !")
