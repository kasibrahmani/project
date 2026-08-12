"""Flask application for employee salary prediction.

Run ``python train_model.py`` once before starting this application.
"""

from pathlib import Path

import joblib
import matplotlib
import pandas as pd
from flask import Flask, render_template, request

matplotlib.use("Agg")  # Lets Matplotlib create images without opening a window.
import matplotlib.pyplot as plt

BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR / "data.xlsx"
MODEL_PATH = BASE_DIR / "salary_model.pkl"
CHART_PATH = BASE_DIR / "static" / "salary_by_position.png"

app = Flask(__name__)


def load_data() -> pd.DataFrame:
    """Read the cleaned project dataset."""
    return pd.read_excel(DATA_PATH)


def create_chart(data: pd.DataFrame) -> None:
    """Create the dashboard chart if it does not exist yet."""
    if CHART_PATH.exists():
        return

    averages = data.groupby("Position")["Salary"].mean().sort_values()
    fig, axis = plt.subplots(figsize=(9, 4.5))
    axis.barh(averages.index, averages.values, color="#183b5b")
    axis.set_title("Average Salary by Position", color="#102a43", fontweight="bold")
    axis.set_xlabel("Annual salary (INR)")
    axis.ticklabel_format(style="plain", axis="x")
    axis.grid(axis="x", alpha=0.2)
    fig.tight_layout()
    fig.savefig(CHART_PATH, dpi=140, bbox_inches="tight")
    plt.close(fig)


def dashboard_values(data: pd.DataFrame) -> dict[str, str]:
    return {
        "employees": f"{len(data):,}",
        "average_salary": f"₹{data['Salary'].mean():,.0f}",
        "highest_salary": f"₹{data['Salary'].max():,.0f}",
        "lowest_salary": f"₹{data['Salary'].min():,.0f}",
        "average_experience": f"{data['Experience_Years'].mean():.1f} years",
    }


def performance_multiplier(score: float) -> float:
    """Return a modest, explainable adjustment for a performance score."""
    # A score of 3 is neutral; each step shifts the base estimate by 3%.
    return 1 + ((score - 3) * 0.03)


@app.route("/", methods=["GET", "POST"])
def index():
    if not DATA_PATH.exists() or not MODEL_PATH.exists():
        return render_template(
            "index.html",
            error="Project files are missing. Run prepare_data.py and then train_model.py first.",
            positions=[], stats={}, result=None, values={},
        )

    data = load_data()
    create_chart(data)
    positions = sorted(data["Position"].dropna().unique())
    result, error = None, None
    values = {
        "position": "", "age": "", "experience": "", "performance_score": "3",
        "daily_hours": "", "weekly_hours": "",
    }

    if request.method == "POST":
        values = {
            "position": request.form.get("position", "").strip(),
            "age": request.form.get("age", "").strip(),
            "experience": request.form.get("experience", "").strip(),
            "performance_score": request.form.get("performance_score", "").strip(),
            "daily_hours": request.form.get("daily_hours", "").strip(),
            "weekly_hours": request.form.get("weekly_hours", "").strip(),
        }
        try:
            if values["position"] not in positions:
                raise ValueError("Please select a valid position.")
            age = int(values["age"])
            experience = float(values["experience"])
            performance_score = float(values["performance_score"])
            daily_hours = float(values["daily_hours"])
            weekly_hours = float(values["weekly_hours"])
            if age < 18 or age > 75:
                raise ValueError("Age must be between 18 and 75.")
            if experience < 0:
                raise ValueError("Years of experience cannot be negative.")
            if experience > age - 14:
                raise ValueError("Experience should be realistic for the selected age.")
            if performance_score < 1 or performance_score > 5:
                raise ValueError("Performance score must be between 1 and 5.")
            if daily_hours <= 0 or daily_hours > 24:
                raise ValueError("Working hours per day must be between 0 and 24.")
            if weekly_hours <= 0 or weekly_hours > 168:
                raise ValueError("Weekly working hours must be between 0 and 168.")

            model = joblib.load(MODEL_PATH)
            input_data = pd.DataFrame([{
                "Position": values["position"],
                "Experience_Years": experience,
                "Working_Hours_Per_Day": daily_hours,
                "Weekly_Hours": weekly_hours,
            }])
            base_prediction = model.predict(input_data)[0]
            prediction = max(0, base_prediction * performance_multiplier(performance_score))
            result = f"₹{prediction:,.0f}"
        except ValueError as exc:
            error = str(exc)
        except Exception:
            error = "The prediction could not be calculated. Please check the entered values."

    return render_template(
        "index.html", positions=positions, stats=dashboard_values(data),
        result=result, error=error, values=values,
    )


if __name__ == "__main__":
    app.run(debug=True)
