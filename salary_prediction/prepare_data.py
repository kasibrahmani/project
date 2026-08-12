"""Prepare the provided source Excel file for this salary-prediction project."""

from pathlib import Path
import pandas as pd

BASE_DIR = Path(__file__).resolve().parent
# The source workbook supplied with this project stores daily hours as Working_Hours.
SOURCE_PATH = BASE_DIR.parent / "enhanced_salary_prediction_dataset_1500.xlsx"
OUTPUT_PATH = BASE_DIR / "data.xlsx"

if not SOURCE_PATH.exists():
    raise FileNotFoundError(f"Put the source workbook at: {SOURCE_PATH}")

source = pd.read_excel(SOURCE_PATH)
data = source[["Position", "Experience_Years", "Working_Hours", "Salary"]].copy()
data = data.rename(columns={"Working_Hours": "Working_Hours_Per_Day"})
# The source dataset reports daily hours. This project uses a five-day work week.
data["Weekly_Hours"] = (data["Working_Hours_Per_Day"] * 5).round(1)

numeric_columns = ["Experience_Years", "Working_Hours_Per_Day", "Weekly_Hours", "Salary"]
for column in numeric_columns:
    data[column] = pd.to_numeric(data[column], errors="coerce")

data = data.dropna().drop_duplicates()
data = data[
    (data["Salary"] > 0)
    & (data["Experience_Years"] >= 0)
    & (data["Working_Hours_Per_Day"] > 0)
    & (data["Working_Hours_Per_Day"] <= 24)
    & (data["Weekly_Hours"] > 0)
]
data.to_excel(OUTPUT_PATH, index=False)
print(f"Saved {len(data)} clean records to {OUTPUT_PATH.name}")
