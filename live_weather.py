
import requests

url = "https://api.open-meteo.com/v1/forecast"

params = {
    "latitude": 17.6868,
    "longitude": 83.2185,
    "current": (
        "temperature_2m,"
        "relative_humidity_2m,"
        "precipitation,"
        "wind_speed_10m"
    ),
    "daily": (
        "temperature_2m_max,"
        "temperature_2m_min,"
        "precipitation_sum,"
        "precipitation_probability_max"
    ),
    "forecast_days": 2,
    "timezone": "Asia/Kolkata"
}

try:
    response = requests.get(url, params=params, timeout=20)
    response.raise_for_status()
    weather = response.json()

    current = weather["current"]
    daily = weather["daily"]

    print("\nLIVE WEATHER — VISAKHAPATNAM")
    print("Updated:", current["time"])
    print("Current temperature:", current["temperature_2m"], "°C")
    print("Humidity:", current["relative_humidity_2m"], "%")
    print("Current precipitation:", current["precipitation"], "mm")
    print("Wind speed:", current["wind_speed_10m"], "km/h")

    print("\nFORECAST")
    print("Forecast date:", daily["time"][1])
    print("Maximum temperature:", daily["temperature_2m_max"][1], "°C")
    print("Minimum temperature:", daily["temperature_2m_min"][1], "°C")
    print("Expected precipitation:", daily["precipitation_sum"][1], "mm")
    print(
        "Forecast precipitation probability:",
        daily["precipitation_probability_max"][1],
        "%"
    )

except requests.RequestException as error:
    print("Could not fetch live weather:", error)
