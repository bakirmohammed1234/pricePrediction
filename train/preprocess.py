import pandas as pd
from config import DATA_PATH, TARGET_COL, FEATURES

def load_and_preprocess():
    df = pd.read_csv(DATA_PATH)

    # Sélection des features
    X = df[FEATURES]
    y = df[TARGET_COL]

    return X, y
