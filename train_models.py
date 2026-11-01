
import pandas as pd
import numpy as np
from pathlib import Path
from joblib import dump

from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    mean_absolute_error,
    mean_squared_error,
    confusion_matrix,
    average_precision_score
)

# --------------------------------------------------
# 1. Load and prepare the weather dataset
# --------------------------------------------------

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

# Mean weather values from the previous three days
for col in weather_cols:
    df[f"{col}_previous_3day_mean"] = (
        df[col].shift(1).rolling(window=3).mean()
    )

# Seasonal features
df["month"] = df["date"].dt.month
df["month_sin"] = np.sin(2 * np.pi * df["month"] / 12)
df["month_cos"] = np.cos(2 * np.pi * df["month"] / 12)

# Targets: weather on the following day
df["rain_tomorrow"] = (
    df["precipitation_sum"].shift(-1) > 0.1
).astype(int)

df["temp_tomorrow"] = df["temp_max"].shift(-1)

# Remove rows with missing features or targets
df = df.dropna().reset_index(drop=True)

# --------------------------------------------------
# 2. Select features and targets
# --------------------------------------------------

features = weather_cols.copy()

features += [
    f"{col}_previous_day"
    for col in weather_cols
]

features += [
    f"{col}_previous_3day_mean"
    for col in weather_cols
]

features += ["month_sin", "month_cos"]

X = df[features]
print("\n--- Features used by the models ---")
print(features)

print("\n--- Dataset date range ---")
print("First date:", df["date"].min())
print("Last date:", df["date"].max())

print("\n--- Missing values in features ---")
print(X.isnull().sum())
y_rain = df["rain_tomorrow"]
y_temp = df["temp_tomorrow"]

# --------------------------------------------------
# 3. Chronological train/test split
# Train before 2026; test on 2026
# --------------------------------------------------

train_mask = df["date"] < pd.Timestamp("2026-01-01")
test_mask = df["date"] >= pd.Timestamp("2026-01-01")

X_train = X.loc[train_mask]
X_test = X.loc[test_mask]

y_rain_train = y_rain.loc[train_mask]
y_rain_test = y_rain.loc[test_mask]

y_temp_train = y_temp.loc[train_mask]
y_temp_test = y_temp.loc[test_mask]

if X_train.empty or X_test.empty:
    raise ValueError(
        "Training or test data is empty. Check the dataset dates."
    )

print("Training dates:",
      df.loc[train_mask, "date"].min().date(),
      "to",
      df.loc[train_mask, "date"].max().date())

print("Testing dates:",
      df.loc[test_mask, "date"].min().date(),
      "to",
      df.loc[test_mask, "date"].max().date())

print("Training records:", len(X_train))
print("Testing records:", len(X_test))

# --------------------------------------------------
# 4. Train and evaluate rainfall model
# --------------------------------------------------

rain_model = RandomForestClassifier(
    n_estimators=200,
    random_state=42,
    class_weight="balanced",
    min_samples_leaf=3
)

rain_model.fit(X_train, y_rain_train)

# IMPORTANT: Generate predictions before evaluating them
rain_probabilities = rain_model.predict_proba(X_test)[:, 1]
rain_predictions = (rain_probabilities >= 0.35).astype(int)
rain_probabilities = rain_model.predict_proba(X_test)[:, 1]

print("\n--- Random Forest Rainfall Results ---")
print(
    "Accuracy:",
    round(accuracy_score(y_rain_test, rain_predictions), 3)
)

print("\nClassification report:")
print(
    classification_report(
        y_rain_test,
        rain_predictions,
        zero_division=0
    )
)

print("Confusion matrix:")
print(confusion_matrix(y_rain_test, rain_predictions))

if y_rain_test.nunique() == 2:
    print(
        "Rainfall average precision (PR-AUC summary):",
        round(
            average_precision_score(
                y_rain_test, rain_probabilities
            ),
            3
        )
    )

# Rainfall baseline: predict tomorrow's rain status
# using today's observed rainfall
rain_baseline_predictions = (
    X_test["precipitation_sum"] > 0.1
).astype(int)

print("\n--- Rainfall Baseline Results ---")
print(
    "Baseline accuracy:",
    round(
        accuracy_score(
            y_rain_test, rain_baseline_predictions
        ),
        3
    )
)

print(
    classification_report(
        y_rain_test,
        rain_baseline_predictions,
        zero_division=0
    )
)

# --------------------------------------------------
# 5. Train and evaluate temperature model
# --------------------------------------------------

temp_model = RandomForestRegressor(
    n_estimators=200,
    random_state=42,
    min_samples_leaf=3
)

temp_model.fit(X_train, y_temp_train)

temp_predictions = temp_model.predict(X_test)

rf_mae = mean_absolute_error(
    y_temp_test, temp_predictions
)

rf_rmse = mean_squared_error(
    y_temp_test, temp_predictions
) ** 0.5

# Persistence baseline: tomorrow equals today's maximum
persistence_predictions = X_test["temp_max"]

baseline_mae = mean_absolute_error(
    y_temp_test, persistence_predictions
)

baseline_rmse = mean_squared_error(
    y_temp_test, persistence_predictions
) ** 0.5

print("\n--- Temperature Results ---")
print("Random Forest MAE:", round(rf_mae, 2), "°C")
print("Baseline MAE:", round(baseline_mae, 2), "°C")
print("Random Forest RMSE:", round(rf_rmse, 2), "°C")
print("Baseline RMSE:", round(baseline_rmse, 2), "°C")

# Monthly temperature comparison
results = pd.DataFrame({
    "date": df.loc[test_mask, "date"].values,
    "actual_temp": y_temp_test.values,
    "rf_prediction": temp_predictions,
    "baseline_prediction": persistence_predictions.values
})

results["month"] = results["date"].dt.month

print("\n--- Temperature MAE by Month ---")

for month, group in results.groupby("month"):
    month_rf_mae = mean_absolute_error(
        group["actual_temp"], group["rf_prediction"]
    )

    month_baseline_mae = mean_absolute_error(
        group["actual_temp"], group["baseline_prediction"]
    )

    print(
        f"Month {month:02d}: "
        f"Random Forest = {month_rf_mae:.2f} °C, "
        f"Baseline = {month_baseline_mae:.2f} °C"
    )

# --------------------------------------------------
# 6. Save trained models
# --------------------------------------------------

Path("models").mkdir(exist_ok=True)

dump(rain_model, "models/rain_model.joblib")
dump(temp_model, "models/temp_model.joblib")

print("\nModels saved successfully in the models folder!")

