import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error
import mlflow
import joblib
import os

def train_model(data_dir="data/processed/"):
    # Load processed data
    X_train = pd.read_csv(f"{data_dir}/X_train.csv")
    y_train = pd.read_csv(f"{data_dir}/y_train.csv")
    X_test = pd.read_csv(f"{data_dir}/X_test.csv")
    y_test = pd.read_csv(f"{data_dir}/y_test.csv")
    
    # Set up MLflow
    mlflow.set_experiment("Real_Estate_Valuation")
    mlflow.sklearn.autolog() # Automatically logs params and metrics!
    
    with mlflow.start_run():
        print("Training Random Forest model...")
        model = RandomForestRegressor(n_estimators=100, max_depth=10, random_state=42)
        model.fit(X_train, y_train.values.ravel())
        
        preds = model.predict(X_test)
        mae = mean_absolute_error(y_test, preds)
        print(f"Model trained. Mean Absolute Error: ${mae:,.2f}")
        
        # Save model locally for the FastAPI app
        os.makedirs("models", exist_ok=True)
        joblib.dump(model, "models/rf_model.pkl")

if __name__ == "__main__":
    train_model()