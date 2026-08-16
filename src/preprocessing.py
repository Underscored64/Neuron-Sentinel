import pandas as pd

weather_full = pd.read_csv("data/processed/weather_raw.csv")
weather_full['time'] = pd.to_datetime(weather_full['time'])
weather_full = weather_full.sort_values(['canton_nom', 'time'])

weather_full['rainfall_1h'] = weather_full['rain']
weather_full['rainfall_6h'] = (
    weather_full.groupby('canton_nom')['rain']
    .rolling(window=6, min_periods=1)
    .sum()
    .reset_index(level=0, drop=True)
)

weather_full = weather_full.rename(columns={'relative_humidity_2m': 'humidity'})

print(weather_full[['canton_nom', 'time', 'rain', 'rainfall_6h']].head(10))


features = weather_full[['canton_nom', 'time', 'rainfall_1h', 'rainfall_6h', 'humidity', 'risk_level']]
features.to_csv("data/processed/training_data.csv", index=False)
print(features.shape)
print(features['risk_level'].value_counts())
