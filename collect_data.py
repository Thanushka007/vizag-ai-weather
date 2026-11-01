
import requests
import pandas as pd
from datetime import date, timedelta
from pathlib import Path

# 1. Visakhapatnam location
LATITUDE = 17.6868
LONGITUDE = 83.2185
TIMEZONE = "Asia/Kolkata"

# 2. Collect 2016 data up to the latest safely available date
START_DATE = date(2016, 1, 1)
END_DATE = date.today() - timedelta(days=5)

if END_DATE < START_DATE:
    raise ValueError("The selected end date is before the start date.")

# 3. Open-Meteo Historical Weather API
URL = "https://archive-api.open-meteo.com/v1/archive"

params = {
    "latitude": LATITUDE,
    "longitude": LONGITUDE,
    "start_date": START_DATE.isoformat(),
    "end_date": END_DATE.isoformat(),
    "hourly": (
        "temperature_2m,"
        "relative_humidity_2m,"
        "precipitation,"
        "wind_speed_10m"
    ),
    "timezone": TIMEZONE,
    "temperature_unit": "celsius",
    "wind_speed_unit": "kmh",
    "precipitation_unit": "mm"
}

print(f"Downloading Vizag weather data from {START_DATE} to {END_DATE}...")

# 4. Download data
response = requests.get(URL, params=params, timeout=60)
response.raise_for_status()
weather = response.json()

if "hourly" not in weather:
    raise ValueError(f"Unexpected API response: {weather}")

# 5. Convert the hourly data into a DataFrame
df = pd.DataFrame(weather["hourly"])
df["time"] = pd.to_datetime(df["time"])
df = df.sort_values("time")

# 6. Save hourly data with timestamps
Path("data").mkdir(exist_ok=True)
df.to_csv("data/vizag_weather_hourly_2026.csv", index=False)

# 7. Calculate daily weather measurements
hourly = df.set_index("time")

daily = hourly.resample("D").agg({
    "temperature_2m": ["max", "min"],
    "relative_humidity_2m": "mean",
    "precipitation": "sum",
    "wind_speed_10m": "max"
})

daily.columns = [
    "temp_max",
    "temp_min",
    "humidity_mean",
    "precipitation_sum",
    "wind_speed_max"
]

daily = daily.dropna().reset_index()
daily = daily.rename(columns={"time": "date"})

# 8. Save daily data for machine learning
daily.to_csv("data/vizag_weather_2026.csv", index=False)

print("\nDownload completed successfully!")
print("Hourly records:", len(df))
print("Daily records:", len(daily))
print("\nDaily dataset preview:")
print(daily.head())

print("\nFiles saved:")
print("data/vizag_weather_hourly_2026.csv")
print("data/vizag_weather_2026.csv")

