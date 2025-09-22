import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

def perform_eda():
    df = pd.read_csv("employee_salary.csv")
    os.makedirs("eda_plots", exist_ok=True)

    # Salary distribution
    plt.figure(figsize=(10, 6))
    sns.histplot(df["salary"], bins=30, kde=True)
    plt.title("Salary Distribution")
    plt.savefig("eda_plots/salary_distribution.png")
    plt.close()

    # Salary by job role
    plt.figure(figsize=(12, 6))
    sns.boxplot(x="job_id", y="salary", data=df)
    plt.title("Salary by Job Role")
    plt.xticks(rotation=45)
    plt.savefig("eda_plots/salary_by_job.png")
    plt.close()

    # Salary by department
    plt.figure(figsize=(12, 6))
    sns.boxplot(x="department_id", y="salary", data=df)
    plt.title("Salary by Department")
    plt.savefig("eda_plots/salary_by_department.png")
    plt.close()

    # Correlation matrix
    numeric_cols = df.select_dtypes(include=["float64", "int64"]).columns
    plt.figure(figsize=(10, 8))
    sns.heatmap(df[numeric_cols].corr(), annot=True, cmap="coolwarm")
    plt.title("Correlation Matrix")
    plt.savefig("eda_plots/correlation_matrix.png")
    plt.close()

if __name__ == "__main__":
    perform_eda()
