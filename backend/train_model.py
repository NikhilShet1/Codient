import pandas as pd
from pathlib import Path
from sklearn.ensemble import IsolationForest


# ----------------------------------
# Load cleaned data
# ----------------------------------
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_PATH = BASE_DIR / "data" / "student_tracking_data_cleaned.csv"

df = pd.read_csv(DATA_PATH)
print(df.columns.tolist())

# ----------------------------------
# Select features for training
# ----------------------------------
features = [
    "attendance_pct",
    "assignment_delay_days",
    "average_grade"
]

X = df[features]

# Drop rows with NULLs (required for training)
X = X.dropna()

# ----------------------------------
# Train Isolation Forest model
# ----------------------------------
model = IsolationForest(
    n_estimators=100,
    contamination=0.15,
    random_state=42
)

model.fit(X)

print("✅ Model trained successfully")

# ----------------------------------
# Generate anomaly scores
# ----------------------------------
df.loc[X.index, "anomaly_score"] = model.decision_function(X)
df.loc[X.index, "risk_flag"] = model.predict(X)

# Convert output to readable form
df["risk_flag"] = df["risk_flag"].map({1: "Normal", -1: "At Risk"})

print("\n🔍 Sample predictions:")
print(df[["attendance_pct", "assignment_delay_days", "average_grade", "risk_flag"]].head())

# ----------------------------------
# Save results
# ----------------------------------
output_path = BASE_DIR / "data" / "student_risk_predictions.csv"
df.to_csv(output_path, index=False)

print(f"\n✅ Predictions saved to: {output_path}")
