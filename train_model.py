import joblib
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
import numpy as np
from data_preprocessing import load_and_preprocess

def train_and_save_models():
    X_train, X_test, y_train, y_test, encoders = load_and_preprocess()
    models = {
        "LinearRegression": LinearRegression(),
        "DecisionTree": DecisionTreeRegressor(random_state=42),
        "RandomForest": RandomForestRegressor(random_state=42)
    }
    best_model = None
    best_r2 = -float("inf")
    for name, model in models.items():
        model.fit(X_train, y_train)
        preds = model.predict(X_test)
        r2 = r2_score(y_test, preds)
        mae = mean_absolute_error(y_test, preds)
        rmse = np.sqrt(mean_squared_error(y_test, preds))
        print(f"\n{name} Performance:")
        print(f"R² Score: {r2:.4f}")
        print(f"MAE: {mae:.2f}")
        print(f"RMSE: {rmse:.2f}")
        if r2 > best_r2:
            best_r2 = r2
            best_model = model
    joblib.dump(best_model, "saved_models/best_model.pkl")
    joblib.dump(encoders, "saved_models/encoders.pkl")
    print(f"\nBest model saved with R² Score: {best_r2:.4f}")

if __name__ == "__main__":
    import os
    os.makedirs("saved_models", exist_ok=True)
    train_and_save_models()
