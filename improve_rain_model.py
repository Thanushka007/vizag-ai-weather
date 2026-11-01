
import pandas as pd
import numpy as np

from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier, HistGradientBoostingClassifier
from sklearn.metrics import accuracy_score, f1_score, classification_report
from sklearn.impute import SimpleImputer
from sklearn.pipeline import make_pipeline

# Load weather data
df = pd.read_csv("data/vizag_weather_2026.csv", parse_dates=["date"])
df = df.sort_values("date").reset_index(drop=True)

# Create rainfall target: will it rain tomorrow?
df["rain_tomorrow"] = (
    df["precipitation_sum"].shift(-1) > 0.1
).astype(int)

# Create previous-day features
for col in ["temp_max", "temp_min", "humidity_mean",
            "precipitation_sum", "wind_speed_max"]:
    df[f"{col}_yesterday"] = df[col].shift(1)

# Create rolling features using only previous days
df["rain_3day_total"] = (
    df["precipitation_sum"].shift(1).rolling(3).sum()
)

df["humidity_3day_mean"] = (
    df["humidity_mean"].shift(1).rolling(3).mean()
)

# Month seasonality
df["month_sin"] = np.sin(2 * np.pi * df["date"].dt.month / 12)
df["month_cos"] = np.cos(2 * np.pi * df["date"].dt.month / 12)

# Remove rows with missing values
df = df.dropna().reset_index(drop=True)

# Features
drop_cols = ["date", "rain_tomorrow", "temp_tomorrow"]
X = df.drop(columns=[c for c in drop_cols if c in df.columns])
y = df["rain_tomorrow"]

# Chronological split: test on 2026
train_mask = df["date"] < "2026-01-01"
test_mask = df["date"] >= "2026-01-01"

X_train, X_test = X.loc[train_mask], X.loc[test_mask]
y_train, y_test = y.loc[train_mask], y.loc[test_mask]

# Baseline: assume tomorrow matches today's rainfall status
baseline = (df.loc[test_mask, "precipitation_sum"] > 0.1).astype(int)

models = {
    "Random Forest": RandomForestClassifier(
        n_estimators=300,
        random_state=42,
        class_weight="balanced",
        min_samples_leaf=4
    ),
    "Extra Trees": ExtraTreesClassifier(
        n_estimators=300,
        random_state=42,
        class_weight="balanced",
        min_samples_leaf=4
    ),
    "Gradient Boosting": HistGradientBoostingClassifier(
        max_iter=150,
        learning_rate=0.08,
        max_leaf_nodes=15,
        l2_regularization=1.0,
        random_state=42
    )
}

print("RAIN MODEL COMPARISON")
print("Training rows:", len(X_train))
print("Testing rows:", len(X_test))
print()

print("Baseline accuracy:", round(accuracy_score(y_test, baseline), 3))
print("Baseline F1:", round(f1_score(y_test, baseline, zero_division=0), 3))
print()

for name, model in models.items():
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)

    print("=" * 40)
    print(name)
    print("Accuracy:", round(accuracy_score(y_test, predictions), 3))
    print("F1 score:", round(f1_score(y_test, predictions, zero_division=0), 3))
    print(classification_report(y_test, predictions, zero_division=0))
