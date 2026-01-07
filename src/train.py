import pickle
from typing import Dict, Tuple

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_absolute_error, r2_score

from preprocess import load_and_preprocess
from config import TEST_SIZE, RANDOM_STATE


def train_models() -> Tuple[
    object,   # best_model
    str,      # best_model_name
    float,    # best_mae
    float,    # best_r2
    Dict[str, Dict]
]:
    """
    Train models and return all artifacts needed for tracking.
    MLflow-agnostic training logic.
    """

    # =========================
    # 1. Load & split data
    # =========================
    X, y = load_and_preprocess()

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE
    )

    # =========================
    # 2. Models
    # =========================
    models = {
        "LinearRegression": LinearRegression(),
        "DecisionTree": DecisionTreeRegressor(
            max_depth=5,
            random_state=RANDOM_STATE
        )
    }

    results = {}
    best_model = None
    best_model_name = None
    best_mae = float("inf")
    best_r2 = float("-inf")

    # =========================
    # 3. Train & evaluate
    # =========================
    for model_name, model in models.items():
        model.fit(X_train, y_train)
        preds = model.predict(X_test)

        mae = mean_absolute_error(y_test, preds)
        r2 = r2_score(y_test, preds)

        results[model_name] = {
            "model": model,
            "mae": mae,
            "r2": r2,
            "params": model.get_params()
        }

        print(f"{model_name} | MAE={mae:.2f} | R2={r2:.3f}")

        # Selection based on MAE (explicit & documented)
        if mae < best_mae:
            best_mae = mae
            best_r2 = r2
            best_model = model
            best_model_name = model_name

    return best_model, best_model_name, best_mae, best_r2, results


# =========================
# Standalone usage
# =========================
if __name__ == "__main__":
    best_model, best_name, best_mae, best_r2, _ = train_models()

    with open("best_model.pkl", "wb") as f:
        pickle.dump(best_model, f)

    print(
        f"\nBest model: {best_name} "
        f"(MAE={best_mae:.2f}, R2={best_r2:.3f})"
    )
    print("Saved → best_model.pkl")
