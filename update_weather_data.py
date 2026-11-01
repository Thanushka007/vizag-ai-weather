
import requests
import pandas as pd
from pathlib import Path
from datetime import date, timedelta

DATA_FILE = Path("data/vizag_weather_2026.csv")

LATITUDE = 17.6868
LONGITUDE = 83.2185

def update_weather_data():
    DATA_FILE.parent.mkdir(parents=True, exist_ok=True)

    # Open-Meteo historical data can have a delay.
    end_date = date.today() - timedelta(days=5)

    if DATA_FILE.exists():
        existing = pd.read_csv(DATA_FILE, parse_dates=["date"])
        last_date = existing["date"].max().date()
        start_date = last_date + timedelta(days=1)
    else:
        existing = pd.DataFrame()

        # Initial download if the CSV does not exist.
        start_date = date(2016, 1, 1)

    if start_date > end_date:
        return existing, "Weather data is already up to date."

    url = "https://archive-api.open-meteo.com/v1/archive"

    params = {
        "latitude": LATITUDE,
        "longitude": LONGITUDE,
        "start_date": start_date.isoformat(),
        "end_date": end_date.isoformat(),
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

    daily = response.json().get("daily")
    if not daily or not daily.get("time"):
        return existing, "No new weather records were returned."

    new_data = pd.DataFrame({
        "date": pd.to_datetime(daily["time"]),
        "temp_max": daily["temperature_2m_max"],
        "temp_min": daily["temperature_2m_min"],
        "humidity_mean": daily["relative_humidity_2m_mean"],
        "precipitation_sum": daily["precipitation_sum"],
        "wind_speed_max": daily["wind_speed_10m_max"]
    })

    if not existing.empty:
        combined = pd.concat(
            [existing, new_data],
            ignore_index=True
        )
    else:
        combined = new_data

    combined["date"] = pd.to_datetime(combined["date"])
    combined = (
        combined.sort_values("date")
        .drop_duplicates(subset=["date"], keep="last")
        .reset_index(drop=True)
    )

    combined.to_csv(DATA_FILE, index=False)

    return combined, (
        f"Weather data updated through "
        f"{combined['date'].max().date()}."
    )
