"""Train and save the salary prediction model."""

from pathlib import Path
import joblib
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

BASE_DIR = Path(__file__).resolve().parent
data = pd.read_excel(BASE_DIR / "data.xlsx")

features = ["Position", "Experience_Years", "Working_Hours_Per_Day", "Weekly_Hours"]
X = data[features]
y = data["Salary"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42)

preprocessor = ColumnTransformer([
    ("position", OneHotEncoder(handle_unknown="ignore"), ["Position"]),
    ("numbers", "passthrough", ["Experience_Years", "Working_Hours_Per_Day", "Weekly_Hours"]),
])
model = Pipeline([
    ("preprocessor", preprocessor),
    ("regression", LinearRegression()),
])
model.fit(X_train, y_train)

predictions = model.predict(X_test)
mse = mean_squared_error(y_test, predictions)
print(f"MAE:  {mean_absolute_error(y_test, predictions):,.2f}")
print(f"MSE:  {mse:,.2f}")
print(f"RMSE: {mse ** 0.5:,.2f}")
print(f"R²:   {r2_score(y_test, predictions):.3f}")

joblib.dump(model, BASE_DIR / "salary_model.pkl")
print("Saved salary_model.pkl")
