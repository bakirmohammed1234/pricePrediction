import mlflow
import mlflow.sklearn
from train import train_model

# Experiment name
mlflow.set_experiment("housing-price-prediction")


def run_pipeline():
    with mlflow.start_run():
        model, rmse = train_model()

        # Log parameters
        mlflow.log_param("model_type", "LinearRegression")

        # Log metrics
        mlflow.log_metric("rmse", rmse)

        # Log model
        mlflow.sklearn.log_model(model, artifact_path="model")

        print("MLflow pipeline completed successfully")


if __name__ == "__main__":
    run_pipeline()
