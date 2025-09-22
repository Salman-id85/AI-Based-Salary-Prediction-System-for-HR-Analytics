# AI-Based-Salary-Prediction-System-for-HR-Analytics
A Streamlit app for predicting employee salaries based on HR features

[![License](https://img.shields.io/badge/License-Apache-2.0-green.svg)](LICENSE)
[![Python](https://img.shields.io/badge/Python-3.8%2B-blue.svg)]()
[![Streamlit](https://img.shields.io/badge/Framework-Streamlit-red)]()
[![scikit-learn](https://img.shields.io/badge/ML-scikit--learn-orange)]()

This project predicts employee salaries based on HR-related features like job role, experience, and department using machine learning. It includes a Streamlit app for real-time predictions and exploratory data analysis (EDA) visualizations to enhance HR decision-making.

## Features
-  **Exploratory Data Analysis (EDA)**: Visualizations of salary distribution, correlations, and salary by job role/department.
-  **Data Cleaning and Preprocessing**: Handles missing data and calculates experience from hire date.
-  **Label Encoding**: Converts categorical variables (job_id, department_id, manager_id) to numeric.
-  **ML Model Training**: Linear Regression, Decision Tree, and Random Forest models, with the best model saved.
-  **Model Evaluation**: Metrics include R² Score, Mean Absolute Error (MAE), and Root Mean Squared Error (RMSE).
-  **Salary Prediction**: Predicts continuous salary values via a user-friendly Streamlit app.
-  **Model Persistence**: Saves and loads models/encoders for real-time predictions.

---

## Dataset
The dataset (`employee_salary.csv`) includes the following columns:
- **employee_id**: Unique identifier
- **first_name**, **last_name**: Employee name
- **email**: Company email
- **phone_number**: Contact number
- **hire_date**: Date of joining
- **job_id**: Job role (e.g., IT_PROG, HR_REP, SALES_REP)
- **salary**: Target variable (continuous, in USD)
- **commission_pct**: Commission percentage (0.0 to 0.5)
- **manager_id**: Reporting manager ID
- **department_id**: Department (e.g., IT, HR, SALES)

---

## Setup
1. **Clone the repository**:
   ```bash
   git clone https://github.com/Salman-id85/AI-Based-Salary-Prediction-System-for-HR-Analytics.git
   cd AI-Based-Salary-Prediction-System-for-HR-Analytics
   ```
2. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```
3. **Run EDA (optional, generates visualizations):**
   ```bash
   python eda.py
   ```
   Outputs plots to eda_plots/ (e.g., salary distribution, salary by job role).
4. **Train the model:**
   ```bash
   python train_model.py
   ```
   Trains Linear Regression, Decision Tree, and Random Forest models, saving the best model and encoders to saved_models/.
5. **Run the Streamlit app:**
  ```bash
streamlit run app.py
```
Access at http://localhost:8501. Enter employee details to predict salary.

---

**Model Performance**
Example metrics (based on sample data):

- R² Score: ~91.3%
- MAE: ~1320.4
- RMSE: ~2550.9

Note: Performance depends on the actual dataset. The sample dataset is small; real-world data may yield different results.

---

**Files**
- app.py: Streamlit app for real-time salary predictions.
- data_preprocessing.py: Loads and preprocesses the dataset, including experience calculation and label encoding.
- train_model.py: Trains multiple regression models and saves the best one.
- eda.py: Generates EDA visualizations (saved to eda_plots/).
- employee_salary.csv: Sample dataset (replace with your actual data if available).
- requirements.txt: Lists required Python packages.
- LICENSE: MIT License file.
- README.md: Project documentation.
- saved_models/: Directory for saved model and encoders (populated after running train_model.py).
- eda_plots/: Directory for EDA visualizations (populated after running eda.py).

---

**Requirements**
Listed in requirements.txt:
```bash
numpy
pandas
scikit-learn
matplotlib
seaborn
streamlit
joblib
```

---

**Notes**

**Dataset:** Replace employee_salary.csv with your actual HR dataset if available, ensuring column names match.
Deployment: For production, deploy using Streamlit Sharing, Heroku, or ngrok (e.g., ngrok http 8501 after setting up an authtoken).
Enhancements:

Add SHAP for model explainability.
Include more models (e.g., XGBoost, SVR) in train_model.py.
Enhance the app with visualizations or a form-based input system.

---

**Performance:** The claimed metrics (R²=91.3%, MAE=1320.4, RMSE=2550.9) are based on sample data. Validate with your dataset.
GitHub: The repository may need updating to match this project (currently uses UCI Adult dataset).

---

**Example Usage**

Run python eda.py to explore data (check eda_plots/).
Run python train_model.py to train models and view performance metrics.
Run streamlit run app.py, then input details (e.g., Job Role=IT_PROG, Experience=5 years) to predict salary.

---

**Future Scope**

Integrate SHAP for explainable AI.
Add more advanced models (e.g., XGBoost, Gradient Boosting).
Expand dataset with additional features (e.g., performance ratings, education).
Deploy as a web service for HR teams.

---

**License**

This project is licensed under the Apache License 2.0 - see the LICENSE file for details.

For issues or contributions, contact the repository owner or open a GitHub issue
