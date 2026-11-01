
import pandas as pd
import numpy as np

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, classification_report

# Load and prepare data
df = pd.read_csv("data/vizag_weather_2026.csv", parse_dates=["date"])
df = df.sort_values("date").reset_index(drop=True)

df["rain_tomorrow"] = (
    df["precipitation_sum"].shift(-1) > 0.1
).astype(int)

for col in [
    "temp_max", "temp_min", "humidity_mean",
    "precipitation_sum", "wind_speed_max"
]:
    df[f"{col}_yesterday"] = df[col].shift(1)

df["rain_3day_total"] = (
    df["precipitation_sum"].shift(1).rolling(3).sum()
)
df["humidity_3day_mean"] = (
    df["humidity_mean"].shift(1).rolling(3).mean()
)
df["month_sin"] = np.sin(2 * np.pi * df["date"].dt.month / 12)
df["month_cos"] = np.cos(2 * np.pi * df["date"].dt.month / 12)

df = df.dropna().reset_index(drop=True)

features = [
    c for c in df.columns
    if c not in ["date", "rain_tomorrow", "temp_tomorrow"]
]

train = df["date"] < "2026-01-01"
test = df["date"] >= "2026-01-01"

X_train, y_train = df.loc[train, features], df.loc[train, "rain_tomorrow"]
X_test, y_test = df.loc[test, features], df.loc[test, "rain_tomorrow"]

model = RandomForestClassifier(
    n_estimators=300,
    random_state=42,
    class_weight="balanced",
    min_samples_leaf=4
)
model.fit(X_train, y_train)

pred = model.predict(X_test)

print("CONFUSION MATRIX")
print("Rows = actual; columns = predicted")
print(confusion_matrix(y_test, pred))

print("\nCLASSIFICATION REPORT")
print(classification_report(y_test, pred, zero_division=0))

results = df.loc[test, ["date", "precipitation_sum", "rain_tomorrow"]].copy()
results["predicted_rain"] = pred
results["prediction_correct"] = (
    results["rain_tomorrow"] == results["predicted_rain"]
)

print("\nWRONG PREDICTIONS (first 20)")
print(
    results.loc[~results["prediction_correct"]]
    .head(20)
    .to_string(index=False)
)

print("\nTOTAL WRONG:", int((~results["prediction_correct"]).sum()))
