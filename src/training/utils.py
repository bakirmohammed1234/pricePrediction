# src/training/utils.py
import pandas as pd
from sklearn.model_selection import train_test_split


def load_data(path="Housing.csv"):
    df = pd.read_csv(path)

    X = df.drop("price", axis=1)
    y = df["price"]

    return train_test_split(X, y, test_size=0.2, random_state=42)
