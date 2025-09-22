import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

def load_and_preprocess():
    df = pd.read_csv("employee_salary.csv")
    df = df.dropna()

    # Calculate experience years from hire_date
    df["hire_date"] = pd.to_datetime(df["hire_date"])
    df["experience_years"] = (pd.Timestamp.now() - df["hire_date"]).dt.days / 365.25

    # Label encode categorical features
    categorical_cols = ["job_id", "manager_id", "department_id"]
    encoders = {}
    for col in categorical_cols:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col])
        encoders[col] = le

    # Features and target
    X = df[["job_id", "commission_pct", "manager_id", "department_id", "experience_years"]]
    y = df["salary"]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    return X_train, X_test, y_train, y_test, encoders
