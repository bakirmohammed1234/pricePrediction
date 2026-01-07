# src/training/mlflow_pipeline.py
import mlflow
import mlflow.sklearn
from utils import load_data
from train import train_model

EXPERIMENT_NAME = "Housing-Price-Prediction"


def run_pipeline():
    mlflow.set_experiment(EXPERIMENT_NAME)

    with mlflow.start_run(run_name="LinearRegression"):

        # Tags (innovation 💡)
        mlflow.set_tags({
            "project": "housing-system",
            "model": "LinearRegression",
            "author": "Abdelaziz",
            "stage": "training"
        })

        # Load data
        X_train, X_test, y_train, y_test = load_data("Housing.csv")

        # Params
        mlflow.log_param("test_size", 0.2)
        mlflow.log_param("random_state", 42)

        # Train
        model, mse, r2 = train_model(
            X_train, y_train, X_test, y_test
        )

        # Metrics
        mlflow.log_metric("mse", mse)
        mlflow.log_metric("r2", r2)

        # Model registry (🔥 très important)
        mlflow.sklearn.log_model(
            model,
            artifact_path="model",
            registered_model_name="HousingLinearModel"
        )

        print("✅ Training completed")
        print(f"MSE: {mse}")
        print(f"R2 : {r2}")


if __name__ == "__main__":
    run_pipeline()
