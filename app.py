import streamlit as st
import pandas as pd
import joblib

st.title("AI-Powered Employee Salary Prediction System")
st.markdown("Predict employee salaries based on HR-related attributes using machine learning models.")

# Load saved model and encoders
model = joblib.load("saved_models/best_model.pkl")
encoders = joblib.load("saved_models/encoders.pkl")

# User inputs
st.sidebar.header("Employee Details")
employee_id = st.sidebar.text_input("Employee ID", "EMP001")
first_name = st.sidebar.text_input("First Name", "John")
last_name = st.sidebar.text_input("Last Name", "Doe")
email = st.sidebar.text_input("Email", "john.doe@company.com")
phone_number = st.sidebar.text_input("Phone Number", "123-456-7890")
hire_date = st.sidebar.date_input("Hire Date")
job_id = st.sidebar.selectbox("Job Role", encoders["job_id"].classes_)
commission_pct = st.sidebar.slider("Commission Percentage", 0.0, 0.5, 0.0, step=0.01)
manager_id = st.sidebar.text_input("Manager ID", "MGR001")
department_id = st.sidebar.selectbox("Department", encoders["department_id"].classes_)
experience_years = st.sidebar.slider("Years of Experience", 0, 30, 5)

# Prepare input data
input_data = pd.DataFrame({
    "employee_id": [employee_id],
    "first_name": [first_name],
    "last_name": [last_name],
    "email": [email],
    "phone_number": [phone_number],
    "hire_date": [hire_date],
    "job_id": [encoders["job_id"].transform([job_id])[0]],
    "commission_pct": [commission_pct],
    "manager_id": [encoders["manager_id"].transform([manager_id])[0] if manager_id in encoders["manager_id"].classes_ else 0],
    "department_id": [encoders["department_id"].transform([department_id])[0]],
    "experience_years": [experience_years]
})

# Drop non-model features
model_features = ["job_id", "commission_pct", "manager_id", "department_id", "experience_years"]
input_data = input_data[model_features]

if st.button("Predict Salary"):
    prediction = model.predict(input_data)[0]
    st.success(f"💰 Predicted Salary: ${prediction:,.2f}")
