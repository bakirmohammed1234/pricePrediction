import pandas as pd
import joblib
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error


def train_model(csv_path="data/processed/housing.csv"):
    """
    Train a Linear Regression model on housing data
    """

    # Load dataset
    df = pd.read_csv(csv_path)

    # Split features / target
    X = df.drop(columns=["price"])
    y = df["price"]

    # Train / test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # Model
    model = LinearRegression()
    model.fit(X_train, y_train)

    # Evaluation
    predictions = model.predict(X_test)
    rmse = mean_squared_error(y_test, predictions, squared=False)

    # Save model
    joblib.dump(model, "model.pkl")

    return model, rmse


if __name__ == "__main__":
    model, rmse = train_model()
    print(f"Training done | RMSE = {rmse}")
