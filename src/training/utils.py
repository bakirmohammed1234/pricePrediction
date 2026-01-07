import pandas as pd
import os
from sklearn.model_selection import train_test_split


def load_data(filename="Housing.csv"):
    BASE_DIR = os.path.abspath(
        os.path.join(os.path.dirname(__file__), "../../")
    )

    data_path = os.path.join(BASE_DIR, filename)

    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Dataset not found: {data_path}")

    df = pd.read_csv(data_path)

    X = df.drop("price", axis=1)
    y = df["price"]

    return train_test_split(X, y, test_size=0.2, random_state=42)
