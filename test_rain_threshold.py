
import pandas as pd
import numpy as np

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score
)

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

X_train = df.loc[train, features]
y_train = df.loc[train, "rain_tomorrow"]
X_test = df.loc[test, features]
y_test = df.loc[test, "rain_tomorrow"]

model = RandomForestClassifier(
    n_estimators=300,
    random_state=42,
    class_weight="balanced",
    min_samples_leaf=4
)

model.fit(X_train, y_train)
probabilities = model.predict_proba(X_test)[:, 1]

print("THRESHOLD COMPARISON")
print("Threshold | Accuracy | Precision | Recall | F1")

for threshold in [0.30, 0.35, 0.40, 0.45, 0.50, 0.55, 0.60]:
    predictions = (probabilities >= threshold).astype(int)

    print(
        f"{threshold:8.2f} | "
        f"{accuracy_score(y_test, predictions):8.3f} | "
        f"{precision_score(y_test, predictions, zero_division=0):9.3f} | "
        f"{recall_score(y_test, predictions, zero_division=0):6.3f} | "
        f"{f1_score(y_test, predictions, zero_division=0):.3f}"
    )
