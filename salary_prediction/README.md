# Employee Salary Prediction

This beginner-friendly Flask project predicts an employee's annual salary in INR.

## Install and run (Python 3.14)

1. Install Python 3.14 from [python.org](https://www.python.org/downloads/) and create a virtual environment:
   ```powershell
   py -3.14 -m venv .venv
   .\.venv\Scripts\Activate.ps1
   ```
2. From this folder, install only the project requirements:
   ```powershell
   python -m pip install --upgrade pip
   python -m pip install -r requirements.txt
   ```
3. Place `enhanced_salary_prediction_dataset_1500.xlsx` one folder above this project, then prepare and train:
   ```powershell
   python prepare_data.py
   python train_model.py
   ```
4. Start the website and open the shown local address (normally http://127.0.0.1:5000):
   ```powershell
   python app.py
   ```

## Dataset preparation

The supplied source workbook contains `Working_Hours`, which is daily working hours. `prepare_data.py` renames it to `Working_Hours_Per_Day` and derives `Weekly_Hours` as daily hours × 5. It keeps only the five needed fields, converts numeric values, removes empty/duplicate rows, and rejects invalid values.

## How prediction works

`train_model.py` uses an 80/20 train/test split. A scikit-learn pipeline one-hot encodes the job position and passes the three numeric fields into a Linear Regression model. It prints MAE, MSE, RMSE, and R², then saves the complete pipeline to `salary_model.pkl`. The Flask app loads that same pipeline for each prediction.

## Notebook

Open `salary_prediction.ipynb` in Jupyter with `jupyter notebook`, run the cells from top to bottom, and it will prepare the data, show checks and four Matplotlib graphs, evaluate the model, and save it.

## Presentation talking points

- The model learns from past salary records, not a fixed salary table.
- Position is categorical, so one-hot encoding converts each position into machine-readable columns.
- Experience and work-hours fields stay numeric.
- MAE describes the average prediction error in rupees; R² indicates how much salary variation the model explains.
- The web form validates inputs before asking the saved model to make an estimate.
