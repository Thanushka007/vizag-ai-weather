
import pandas as pd
import numpy as np

from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    mean_absolute_error
)

# Load the dataset
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

# Previous-day and previous-three-day features
for col in weather_cols:
    df[f"{col}_previous_day"] = df[col].shift(1)
    df[f"{col}_previous_3day_mean"] = (
        df[col].shift(1).rolling(3).mean()
    )

# Seasonal features
df["month_sin"] = np.sin(
    2 * np.pi * df["date"].dt.month / 12
)
df["month_cos"] = np.cos(
    2 * np.pi * df["date"].dt.month / 12
)

# Next-day targets
df["rain_tomorrow"] = (
    df["precipitation_sum"].shift(-1) > 0.1
).astype(int)

df["temp_tomorrow"] = df["temp_max"].shift(-1)

df = df.dropna().reset_index(drop=True)

features = weather_cols.copy()
features += [f"{c}_previous_day" for c in weather_cols]
features += [f"{c}_previous_3day_mean" for c in weather_cols]
features += ["month_sin", "month_cos"]

# Evaluate separate future periods
periods = [
    ("2023", "2023-01-01", "2024-01-01"),
    ("2024", "2024-01-01", "2025-01-01"),
    ("2025", "2025-01-01", "2026-01-01"),
    ("2026", "2026-01-01", "2027-01-01")
]

print("\n--- Rolling Time-Based Validation ---")

for name, start, end in periods:
    train_mask = df["date"] < start
    test_mask = (
        (df["date"] >= start) &
        (df["date"] < end)
    )

    X_train = df.loc[train_mask, features]
    X_test = df.loc[test_mask, features]

    y_rain_train = df.loc[train_mask, "rain_tomorrow"]
    y_rain_test = df.loc[test_mask, "rain_tomorrow"]

    y_temp_train = df.loc[train_mask, "temp_tomorrow"]
    y_temp_test = df.loc[test_mask, "temp_tomorrow"]

    if len(X_train) == 0 or len(X_test) == 0:
        print(f"{name}: insufficient data")
        continue

    rain_model = RandomForestClassifier(
        n_estimators=200,
        random_state=42,
        class_weight="balanced",
        min_samples_leaf=3
    )

    temp_model = RandomForestRegressor(
        n_estimators=200,
        random_state=42,
        min_samples_leaf=3
    )

    rain_model.fit(X_train, y_rain_train)
    temp_model.fit(X_train, y_temp_train)

    rain_pred = rain_model.predict(X_test)
    temp_pred = temp_model.predict(X_test)

    # Baselines
    rain_baseline = (
        X_test["precipitation_sum"] > 0.1
    ).astype(int)

    temp_baseline = X_test["temp_max"]

    print(f"\nTest period: {name}")
    print("Training rows:", len(X_train))
    print("Testing rows:", len(X_test))

    print(
        "Rain ML accuracy:",
        round(accuracy_score(y_rain_test, rain_pred), 3)
    )
    print(
        "Rain ML F1:",
        round(f1_score(y_rain_test, rain_pred, zero_division=0), 3)
    )
    print(
        "Rain baseline accuracy:",
        round(accuracy_score(y_rain_test, rain_baseline), 3)
    )

    print(
        "Temperature ML MAE:",
        round(mean_absolute_error(y_temp_test, temp_pred), 2)
    )
    print(
        "Temperature baseline MAE:",
        round(mean_absolute_error(y_temp_test, temp_baseline), 2)
    )
