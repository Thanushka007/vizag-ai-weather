from joblib import load
from update_weather_data import update_weather_data

import streamlit as st
import pandas as pd
import numpy as np
import requests
import joblib

# ---------------- AUTOMATIC WEATHER UPDATE ----------------
try:
    updated_df, update_message = update_weather_data()
    st.session_state["weather_update_message"] = update_message
except Exception as e:
    st.session_state["weather_update_message"] = (
        f"Weather update unavailable: {e}"
    )

st.caption(
    "Historical data: "
    + st.session_state["weather_update_message"]
)





# ---------------- PAGE SETTINGS ----------------
st.set_page_config(
    page_title="Vizag AI Weather",
    page_icon="🌦️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ---------------- CUSTOM DESIGN ----------------
st.markdown("""
<style>
.stApp {
    background: linear-gradient(180deg, #edf6ff 0%, #f8fbff 55%, #ffffff 100%);
    color: #183153;
}
[data-testid="stHeader"] {
    background: rgba(237, 246, 255, 0.9);
}
.block-container {
    padding-top: 2rem;
    padding-bottom: 2rem;
    max-width: 1250px;
}
.hero {
    background: linear-gradient(120deg, #103d70, #167ac6, #45b5e8);
    padding: 32px;
    border-radius: 22px;
    color: white;
    margin-bottom: 24px;
    box-shadow: 0 10px 30px rgba(22, 90, 150, 0.18);
}
.hero h1 {
    color: white;
    font-size: 38px;
    margin: 0;
}
.hero p {
    color: #e5f4ff;
    font-size: 16px;
    margin-bottom: 0;
}
.section-title {
    color: #153f6c;
    font-size: 23px;
    font-weight: 700;
    margin-top: 20px;
    margin-bottom: 12px;
}
.metric-card {
    background: white;
    border: 1px solid #d9eaff;
    border-radius: 18px;
    padding: 22px;
    min-height: 145px;
    box-shadow: 0 5px 18px rgba(35, 89, 140, 0.07);
}
.metric-label {
    color: #637c98;
    font-size: 14px;
    font-weight: 600;
}
.metric-value {
    color: #123f70;
    font-size: 32px;
    font-weight: 750;
    margin-top: 10px;
}
.metric-note {
    color: #70859d;
    font-size: 12px;
    margin-top: 7px;
}
.status-card {
    background: #e7f4ff;
    border-left: 5px solid #1685d4;
    padding: 17px 20px;
    border-radius: 12px;
    color: #16446e;
    margin-top: 12px;
}
div.stButton > button {
    background: linear-gradient(100deg, #1267b1, #2196df);
    color: white;
    border: none;
    border-radius: 10px;
    padding: 0.65rem 1.1rem;
    font-weight: 600;
    width: 100%;
}
div.stButton > button:hover {
    background: #0d5596;
    color: white;
    border: none;
}
[data-testid="stSidebar"] {
    background: #e5f2ff;
}
footer {
    visibility: hidden;
}
</style>
""", unsafe_allow_html=True)

# ---------------- HEADER ----------------
st.markdown("""
<div class="hero">
    <h1>🌦️ Vizag AI Weather</h1>
    <p>Intelligent rainfall and temperature prediction for Visakhapatnam</p>
    <p>Machine Learning • Weather Analytics • Open-Meteo Data</p>
</div>
""", unsafe_allow_html=True)

# ---------------- SIDEBAR ----------------
with st.sidebar:
    st.markdown("## 🌤️ Project Menu")
    st.write("**Location:** Visakhapatnam, Andhra Pradesh")
    st.write("**Models:** Random Forest")
    st.write("**Predictions:**")
    st.write("- Maximum temperature")
    st.write("- Rain probability")
    st.divider()
    st.caption("AI & Data Science Project")
    st.caption("Predictions are experimental, not official forecasts.")



# ---------------- LOAD MODELS ----------------
@st.cache_resource
def load_models():
    rain_model = load("models/rain_model.joblib")
    temp_model = load("models/temp_model.joblib")
    return rain_model, temp_model



# ---------------- FETCH WEATHER DATA ----------------
def fetch_weather_data():
    today = pd.Timestamp.now(tz="Asia/Kolkata").date()
    end_date = today - pd.Timedelta(days=2)
    start_date = today - pd.Timedelta(days=7)

    url = "https://archive-api.open-meteo.com/v1/archive"

    params = {
        "latitude": 17.6868,
        "longitude": 83.2185,
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
    data = response.json()["daily"]

    history = pd.DataFrame({
        "date": pd.to_datetime(data["time"]),
        "temp_max": data["temperature_2m_max"],
        "temp_min": data["temperature_2m_min"],
        "humidity_mean": data["relative_humidity_2m_mean"],
        "precipitation_sum": data["precipitation_sum"],
        "wind_speed_max": data["wind_speed_10m_max"]
    })

    history = history.dropna().reset_index(drop=True)
    return history


# ---------------- CREATE MODEL FEATURES ----------------
def create_features(history):
    if len(history) < 4:
        raise ValueError("Not enough weather data to make a prediction.")

    current = history.iloc[-1]
    previous = history.iloc[-2]
    previous_three = history.iloc[-4:-1]

    features = {
        "temp_max": current["temp_max"],
        "temp_min": current["temp_min"],
        "humidity_mean": current["humidity_mean"],
        "precipitation_sum": current["precipitation_sum"],
        "wind_speed_max": current["wind_speed_max"],

        "temp_max_previous_day": previous["temp_max"],
        "temp_min_previous_day": previous["temp_min"],
        "humidity_mean_previous_day": previous["humidity_mean"],
        "precipitation_sum_previous_day": previous["precipitation_sum"],
        "wind_speed_max_previous_day": previous["wind_speed_max"],

        "temp_max_previous_3day_mean":
            previous_three["temp_max"].mean(),
        "temp_min_previous_3day_mean":
            previous_three["temp_min"].mean(),
        "humidity_mean_previous_3day_mean":
            previous_three["humidity_mean"].mean(),
        "precipitation_sum_previous_3day_mean":
            previous_three["precipitation_sum"].mean(),
        "wind_speed_max_previous_3day_mean":
            previous_three["wind_speed_max"].mean(),

        "month_sin": np.sin(
            2 * np.pi * current["date"].month / 12
        ),
        "month_cos": np.cos(
            2 * np.pi * current["date"].month / 12
        )
    }

    return pd.DataFrame([features]), current

# ---------------- 7-DAY WEATHER FORECAST ----------------
def fetch_7_day_forecast():
    url = "https://api.open-meteo.com/v1/forecast"

    params = {
        "latitude": 17.6868,
        "longitude": 83.2185,
        "daily": (
            "weather_code,"
            "temperature_2m_max,"
            "temperature_2m_min,"
            "precipitation_probability_max,"
            "precipitation_sum"
        ),
        "timezone": "Asia/Kolkata",
        "forecast_days": 7
    }

    response = requests.get(url, params=params, timeout=30)
    response.raise_for_status()

    daily = response.json()["daily"]

    forecast = pd.DataFrame({
        "Date": pd.to_datetime(daily["time"]),
        "Max Temperature (°C)": daily["temperature_2m_max"],
        "Min Temperature (°C)": daily["temperature_2m_min"],
        "Rain Probability (%)": daily["precipitation_probability_max"],
        "Rainfall (mm)": daily["precipitation_sum"],
        "Weather Code": daily["weather_code"]
    })

    return forecast


def weather_description(code):
    if code == 0:
        return "Clear sky"
    elif code in [1, 2]:
        return "Mainly clear / partly cloudy"
    elif code == 3:
        return "Overcast"
    elif code in [45, 48]:
        return "Fog"
    elif code in [51, 53, 55, 56, 57]:
        return "Drizzle"
    elif code in [61, 63, 65, 66, 67]:
        return "Rain"
    elif code in [71, 73, 75, 77]:
        return "Snow"
    elif code in [80, 81, 82]:
        return "Rain showers"
    elif code in [85, 86]:
        return "Snow showers"
    elif code in [95, 96, 99]:
        return "Thunderstorm"
    else:
        return "Weather conditions vary"
    


    
# ---------------- AI MODEL PERFORMANCE ----------------


@st.cache_data(ttl=3600)
def get_model_performance():
    import numpy as np
    import pandas as pd
    from sklearn.metrics import (
        accuracy_score,
        f1_score,
        mean_absolute_error
    )

    df = pd.read_csv(
        "data/vizag_weather_2026.csv",
        parse_dates=["date"]
    )

    df = df.sort_values("date").reset_index(drop=True)

    weather_cols = [
        "temp_max",
        "temp_min",
        "humidity_mean",
        "precipitation_sum",
        "wind_speed_max"
    ]

    # Previous-day features
    for col in weather_cols:
        df[f"{col}_previous_day"] = df[col].shift(1)

    # Previous three-day averages
    for col in weather_cols:
        df[f"{col}_previous_3day_mean"] = (
            df[col].shift(1).rolling(3).mean()
        )

    # Seasonal features
    df["month"] = df["date"].dt.month
    df["month_sin"] = np.sin(2 * np.pi * df["month"] / 12)
    df["month_cos"] = np.cos(2 * np.pi * df["month"] / 12)

    # Actual next-day weather
    df["actual_rain"] = (
        df["precipitation_sum"].shift(-1) > 0.1
    ).astype(int)

    df["actual_temp"] = df["temp_max"].shift(-1)

    # All 17 model features
    features = (
        weather_cols
        + [f"{col}_previous_day" for col in weather_cols]
        + [f"{col}_previous_3day_mean" for col in weather_cols]
        + ["month_sin", "month_cos"]
    )

    # Include all available 2026 records
    test = df[
        df["date"] >= pd.Timestamp("2026-01-01")
    ].dropna(
        subset=features + ["actual_rain", "actual_temp"]
    ).copy()

    if test.empty:
        raise ValueError("No evaluation records available.")

    X_test = test[features]

    # Load trained models
    rain_model = load("models/rain_model.joblib")
    temp_model = load("models/temp_model.joblib")

    # AI predictions
    rain_prob = rain_model.predict_proba(X_test)[:, 1]
    rain_pred = (rain_prob >= 0.35).astype(int)
    temp_pred = temp_model.predict(X_test)

    # Baseline predictions
    rain_baseline = (
        test["precipitation_sum"] > 0.1
    ).astype(int)

    temp_baseline = test["temp_max"]

    actual_rain = test["actual_rain"]
    actual_temp = test["actual_temp"]

    return {
        "rows": len(test),
        "rain_ai_accuracy": accuracy_score(
            actual_rain, rain_pred
        ),
        "rain_ai_f1": f1_score(
            actual_rain, rain_pred, zero_division=0
        ),
        "rain_baseline_accuracy": accuracy_score(
            actual_rain, rain_baseline
        ),
        "temp_ai_mae": mean_absolute_error(
            actual_temp, temp_pred
        ),
        "temp_baseline_mae": mean_absolute_error(
            actual_temp, temp_baseline
        ),
        "comparison": pd.DataFrame({
            "Date": test["date"].dt.strftime("%Y-%m-%d"),
            "Actual Rain": actual_rain.map({
                0: "No rain", 1: "Rain"
            }).values,
            "AI Rain": pd.Series(
                rain_pred, index=test.index
            ).map({
                0: "No rain", 1: "Rain"
            }).values,
            "Actual Max Temp (°C)": actual_temp.round(1).values,
            "AI Max Temp (°C)": np.round(temp_pred, 1)
        })
    }



# Display the performance dashboard
st.header("📊 AI Model Performance")
st.caption(
    "Historical evaluation: January–September 2026. "
    "These results are retrospective, not a live forecast."
)

if st.button("Evaluate My AI Models"):
    try:
        performance = get_model_performance()

        col1, col2, col3 = st.columns(3)

        col1.metric(
            "Rain AI Accuracy",
            f"{performance['rain_ai_accuracy']:.1%}"
        )

        col2.metric(
            "Rain Baseline Accuracy",
            f"{performance['rain_baseline_accuracy']:.1%}"
        )

        col3.metric(
            "Rain AI F1 Score",
            f"{performance['rain_ai_f1']:.3f}"
        )

        col4, col5 = st.columns(2)

        col4.metric(
            "Temperature AI MAE",
            f"{performance['temp_ai_mae']:.2f} °C",
            help="Lower is better"
        )

        col5.metric(
            "Temperature Baseline MAE",
            f"{performance['temp_baseline_mae']:.2f} °C",
            help="Lower is better"
        )

        st.subheader("Actual Weather vs AI Predictions")
        st.dataframe(
            performance["comparison"],
            use_container_width=True,
            hide_index=True
        )

        st.caption(
            "Rain predictions use your training threshold of 0.35. "
            "This evaluation uses historical weather observations "
            "and does not prove future forecast accuracy."
        )

    except Exception as e:
        st.error(f"Could not evaluate the models: {e}")





# ---------------- MAIN DASHBOARD ----------------
try:
    rain_model, temp_model = load_models()

    st.markdown(
        '<div class="section-title">📊 Weather Prediction Dashboard</div>',
        unsafe_allow_html=True
    )

    
    # Seven-day forecast section
    st.markdown(
        '<div class="section-title">📅 7-Day Weather Forecast</div>',
        unsafe_allow_html=True
    )

    st.write(
        "View the weather provider's forecast separately from "
        "your experimental AI model predictions."
    )

    if st.button("🌤️ Load 7-Day Forecast"):
        with st.spinner("Loading Vizag weather forecast..."):
            forecast = fetch_7_day_forecast()
            st.session_state["seven_day_forecast"] = forecast

    if "seven_day_forecast" in st.session_state:
        forecast = st.session_state["seven_day_forecast"].copy()

        # Summary cards for the first forecast day
        first_day = forecast.iloc[0]
        c1, c2, c3 = st.columns(3)

        with c1:
            st.metric(
                "Today's Forecast High",
                f"{first_day['Max Temperature (°C)']:.1f} °C"
            )

        with c2:
            st.metric(
                "Today's Forecast Low",
                f"{first_day['Min Temperature (°C)']:.1f} °C"
            )

        with c3:
            probability = first_day["Rain Probability (%)"]
            if pd.notna(probability):
                st.metric(
                    "Maximum Rain Probability",
                    f"{probability:.0f}%"
                )
            else:
                st.metric("Maximum Rain Probability", "Unavailable")

        # Weather descriptions
        forecast["Conditions"] = forecast["Weather Code"].apply(
            lambda code: weather_description(int(code))
            if pd.notna(code) else "Unavailable"
        )

        # Format dates and display table
        display_forecast = forecast.copy()
        display_forecast["Date"] = display_forecast["Date"].dt.strftime(
            "%a, %d %b"
        )
        display_forecast = display_forecast.drop(
            columns=["Weather Code"]
        )

        st.dataframe(
            display_forecast,
            use_container_width=True,
            hide_index=True
        )

        st.markdown("#### 🌡️ 7-Day Temperature Trend")

        chart = forecast.set_index("Date")[
            ["Max Temperature (°C)", "Min Temperature (°C)"]
        ]
        st.line_chart(chart)

        st.markdown("#### 🌧️ Expected Daily Rainfall")
        st.bar_chart(
            forecast.set_index("Date")[["Rainfall (mm)"]]
        )

        
        # ---------------- FORECAST INSIGHTS ----------------
        st.markdown("#### 🔍 Forecast Insights")

        valid_temps = forecast.dropna(
            subset=["Max Temperature (°C)"]
        )
        valid_rain = forecast.dropna(
            subset=["Rain Probability (%)"]
        )

        if not valid_temps.empty:
            warmest = valid_temps.loc[
                valid_temps["Max Temperature (°C)"].idxmax()
            ]

            total_rainfall = forecast["Rainfall (mm)"].sum()

            insight1, insight2, insight3 = st.columns(3)

            with insight1:
                st.metric(
                    "🔥 Warmest Day",
                    warmest["Date"].strftime("%a, %d %b"),
                    f"{warmest['Max Temperature (°C)']:.1f} °C"
                )

            with insight2:
                st.metric(
                    "🌧️ Expected 7-Day Rainfall",
                    f"{total_rainfall:.1f} mm"
                )

            with insight3:
                if not valid_rain.empty:
                    rainiest = valid_rain.loc[
                        valid_rain["Rain Probability (%)"].idxmax()
                    ]

                    st.metric(
                        "☔ Highest Rain Chance",
                        f"{rainiest['Rain Probability (%)']:.0f}%",
                        rainiest["Date"].strftime("%a, %d %b")
                    )
                else:
                    st.metric(
                        "☔ Highest Rain Chance",
                        "Unavailable"
                    )



    if st.button("🔄 Generate Latest Prediction"):
        with st.spinner("Fetching weather data and running AI models..."):
            history = fetch_weather_data()
            X, current = create_features(history)

            # Keep the existing model prediction logic
            rain_probability = rain_model.predict_proba(X)[0][1]
            rain_expected = rain_probability >= 0.35
            predicted_temp = temp_model.predict(X)[0]

            st.session_state["weather_result"] = {
                "history": history,
                "data_date": current["date"].strftime("%Y-%m-%d"),
                "rain_probability": float(rain_probability),
                "rain_expected": bool(rain_expected),
                "predicted_temp": float(predicted_temp),
                "humidity": float(current["humidity_mean"]),
                "wind": float(current["wind_speed_max"]),
                "rainfall": float(current["precipitation_sum"])
            }

    if "weather_result" in st.session_state:
        result = st.session_state["weather_result"]

        st.markdown(
            '<div class="section-title">✨ AI Prediction Results</div>',
            unsafe_allow_html=True
        )

        col1, col2, col3 = st.columns(3)

        with col1:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-label">🌡️ Predicted Maximum Temperature</div>
                <div class="metric-value">{result['predicted_temp']:.1f} °C</div>
                <div class="metric-note">Experimental model estimate</div>
            </div>
            """, unsafe_allow_html=True)

        with col2:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-label">🌧️ Estimated Rain Probability</div>
                <div class="metric-value">{result['rain_probability'] * 100:.1f}%</div>
                <div class="metric-note">Model probability, not a verified forecast</div>
            </div>
            """, unsafe_allow_html=True)

        with col3:
            rain_label = "Rain expected" if result["rain_expected"] else "Rain not expected"
            rain_icon = "🌧️" if result["rain_expected"] else "☀️"

            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-label">☁️ AI Rain Classification</div>
                <div class="metric-value" style="font-size:25px;">
                    {rain_icon} {rain_label}
                </div>
                <div class="metric-note">Classification threshold: 35%</div>
            </div>
            """, unsafe_allow_html=True)

        st.markdown("#### Rain Probability")
        st.progress(
            min(max(result["rain_probability"], 0.0), 1.0),
            text=f"{result['rain_probability'] * 100:.1f}% estimated probability"
        )

        st.markdown(
            f"""
            <div class="status-card">
                <b>📅 Latest daily data used:</b> {result['data_date']}<br>
                <b>📍 Location:</b> Visakhapatnam, Andhra Pradesh
            </div>
            """,
            unsafe_allow_html=True
        )

        st.markdown(
            '<div class="section-title">🌤️ Recent Weather Conditions</div>',
            unsafe_allow_html=True
        )

        history = result["history"].copy()
        history["date"] = history["date"].dt.strftime("%d %b %Y")
        history = history.rename(columns={
            "date": "Date",
            "temp_max": "Max Temp (°C)",
            "temp_min": "Min Temp (°C)",
            "humidity_mean": "Humidity (%)",
            "precipitation_sum": "Rainfall (mm)",
            "wind_speed_max": "Max Wind (km/h)"
        })

        st.dataframe(
            history.sort_values("Date", ascending=False),
            use_container_width=True,
            hide_index=True
        )

        st.markdown(
            '<div class="section-title">📈 Temperature Trend</div>',
            unsafe_allow_html=True
        )

        chart_data = result["history"].set_index("date")[
            ["temp_max", "temp_min"]
        ].rename(columns={
            "temp_max": "Maximum temperature",
            "temp_min": "Minimum temperature"
        })

        st.line_chart(chart_data)

        st.info(
            "Important: These are experimental model predictions, not "
            "verified forecasts. Data is obtained from the Open-Meteo "
            "historical archive, which may contain reanalysis estimates. "
            "The displayed rain probability has not been established as "
            "a calibrated real-world probability."
        )

    else:
        st.markdown("""
        <div class="status-card">
            👋 Welcome! Click <b>Generate Latest Prediction</b> to load
            weather data and view your AI model results.
        </div>
        """, unsafe_allow_html=True)

except FileNotFoundError:
    st.error(
        "Model files not found. Please check that these files exist: "
        "models/rain_model.joblib and models/temp_model.joblib"
    )

except requests.RequestException as error:
    st.error(f"Could not retrieve weather data. Check your internet connection. Details: {error}")

except Exception as error:
    st.error(f"Something went wrong: {error}")
