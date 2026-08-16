import requests
import pandas as pd
import time
import geopandas as gpd

def get_weather_history(lat, lon, start_date, end_date):
    url = "https://archive-api.open-meteo.com/v1/archive"
    params = {
        "latitude": lat,
        "longitude": lon,
        "start_date": start_date,
        "end_date": end_date,
        "hourly": "rain,relative_humidity_2m",
        "timezone": "Africa/Lome"
    }
    response = requests.get(url, params=params)
    response.raise_for_status()
    data = response.json()["hourly"]
    return pd.DataFrame(data)

# Boucle sur les 13 cantons 
lome = gpd.read_file("data/processed/lome_fri.gpkg")
all_data = []
for _, row in lome.iterrows():
    df = get_weather_history(row['lat'], row['lon'], "2023-01-01", "2024-12-31")
    df['canton_nom'] = row['canton_nom']
    df['risk_level'] = row['risk_level']
    all_data.append(df)
    time.sleep(1)  

weather_full = pd.concat(all_data, ignore_index=True)
weather_full.to_csv("data/processed/weather_raw.csv", index=False)
print(weather_full.shape)
print(weather_full.head())