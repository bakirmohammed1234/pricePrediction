import mlflow
import mlflow.sklearn
from utils import load_data
from train import train_model

EXPERIMENT_NAME = "Housing-Price-Prediction"


def run_pipeline():
    mlflow.set_experiment(EXPERIMENT_NAME)

    with mlflow.start_run(run_name="LinearRegression"):

        mlflow.set_tags({
            "project": "price-prediction",
            "owner": "Abdelaziz",
            "framework": "scikit-learn",
            "type": "regression"
        })

        X_train, X_test, y_train, y_test = load_data()

        mlflow.log_param("model_type", "LinearRegression")
        mlflow.log_param("test_size", 0.2)

        model, mse, r2 = train_model(
            X_train, y_train, X_test, y_test
        )

        mlflow.log_metric("mse", mse)
        mlflow.log_metric("r2", r2)

        mlflow.sklearn.log_model(
            model,
            artifact_path="model",
            registered_model_name="HousingModel"
        )

        print("✅ MLflow pipeline finished successfully")


if __name__ == "__main__":
    run_pipeline()
