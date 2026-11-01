
import requests
import pandas as pd
import numpy as np
import joblib

# 1. Load your trained AI models
rain_model = joblib.load("models/rain_model.joblib")
temp_model = joblib.load("models/temp_model.joblib")

# 2. Fetch recent daily weather history.
# We need several days to build the model's rolling features.
url = "https://archive-api.open-meteo.com/v1/archive"

params = {
    "latitude": 17.6868,
    "longitude": 83.2185,
    "start_date": (pd.Timestamp.now(tz="Asia/Kolkata").date()
                   - pd.Timedelta(days=7)).isoformat(),
    "end_date": (pd.Timestamp.now(tz="Asia/Kolkata").date()
                 - pd.Timedelta(days=2)).isoformat(),
    "daily": (
        "temperature_2m_max,"
        "temperature_2m_min,"
        "relative_humidity_2m_mean,"
        "precipitation_sum,"
        "wind_speed_10m_max"
    ),
    "timezone": "Asia/Kolkata"
}

response = requests.get(url, params=params, timeout=30)
response.raise_for_status()

daily = response.json()["daily"]
history = pd.DataFrame({
    "date": pd.to_datetime(daily["time"]),
    "temp_max": daily["temperature_2m_max"],
    "temp_min": daily["temperature_2m_min"],
    "humidity_mean": daily["relative_humidity_2m_mean"],
    "precipitation_sum": daily["precipitation_sum"],
    "wind_speed_max": daily["wind_speed_10m_max"]
})

# 3. Check whether the data is usable
columns = [
    "temp_max", "temp_min", "humidity_mean",
    "precipitation_sum", "wind_speed_max"
]

if len(history) < 4 or history[columns].isna().any().any():
    raise ValueError("Not enough complete daily weather data.")

# 4. Build the same feature names used during training
row = {}

for col in columns:
    row[col] = history[col].iloc[-1]
    row[f"{col}_previous_day"] = history[col].iloc[-2]
    row[f"{col}_previous_3day_mean"] = history[col].iloc[-4:-1].mean()

today = pd.Timestamp.now(tz="Asia/Kolkata")
row["month_sin"] = np.sin(2 * np.pi * today.month / 12)
row["month_cos"] = np.cos(2 * np.pi * today.month / 12)

# 5. Arrange inputs in the exact order expected by each model
features = [
    "temp_max", "temp_min", "humidity_mean",
    "precipitation_sum", "wind_speed_max",
    "temp_max_previous_day", "temp_min_previous_day",
    "humidity_mean_previous_day", "precipitation_sum_previous_day",
    "wind_speed_max_previous_day", "temp_max_previous_3day_mean",
    "temp_min_previous_3day_mean", "humidity_mean_previous_3day_mean",
    "precipitation_sum_previous_3day_mean",
    "wind_speed_max_previous_3day_mean", "month_sin", "month_cos"
]

X = pd.DataFrame([row])[features]

# 6. Generate AI predictions
rain_probability = rain_model.predict_proba(X)[0][1]
rain_prediction = rain_probability >= 0.35
temp_prediction = temp_model.predict(X)[0]

print("\nVIZAG AI WEATHER PREDICTION")
print("Latest daily data date:", history["date"].iloc[-1].date())
print("AI predicted maximum temperature tomorrow:",
      round(float(temp_prediction), 1), "°C")
print("AI estimated rain probability:",
      round(float(rain_probability) * 100, 1), "%")
print("AI rainfall classification:",
      "Rain expected" if rain_prediction else "Rain not expected")
print("\nNote: These are experimental model predictions, not verified forecasts.")
