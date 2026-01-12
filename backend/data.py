import pandas as pd
from pathlib import Path

# --------------------------------------------------
# Load CSV safely from project structure
# --------------------------------------------------
BASE_DIR = Path(__file__).resolve().parent.parent
CSV_PATH = BASE_DIR / "data" / "student_tracking_data.csv"

df = pd.read_csv(CSV_PATH)

# --------------------------------------------------
# Standardize column names
# --------------------------------------------------
df.columns = df.columns.str.lower().str.strip()

# --------------------------------------------------
# student_id → clean, missing/invalid → NULL
# --------------------------------------------------
if "student_id" in df.columns:
    df["student_id"] = (
        df["student_id"]
        .astype(str)
        .str.strip()
        .str.upper()
    )
    df.loc[df["student_id"].isin(["", "NAN", "NONE"]), "student_id"] = pd.NA

# --------------------------------------------------
# date → convert, invalid/missing → NULL
# --------------------------------------------------
if "date" in df.columns:
    df["date"] = pd.to_datetime(df["date"], errors="coerce")

# numeric columns → convert, invalid/missing → NULL
# --------------------------------------------------
for col in ["attendance_pct", "assignment_delay_days", "average_grade"]:
    if col in df.columns:
        df[col] = pd.to_numeric(df[col], errors="coerce")
df = df.dropna()

# --------------------------------------------------
# Preview + save cleaned file
# --------------------------------------------------
print("✅ Cleaned data preview:")
print(df.head())

output_path = BASE_DIR / "data" / "student_tracking_data_cleaned.csv"
df.to_csv(output_path, index=False)

print(f"\n✅ Cleaned file saved at: {output_path}")
print("\n🔍 NULL counts:")
print(df.isna().sum())


