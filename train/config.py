from pathlib import Path
# DATA_PATH = "data/cleaned_housing_data.csv"
# Racine du projet (1 niveau au-dessus de config.py)
PROJECT_ROOT = Path(__file__).resolve().parent

DATA_PATH = PROJECT_ROOT / "data" / "cleaned_housing_data.csv"
TARGET_COL = "price"

FEATURES = [
    "area",
    "bedrooms",
    "bathrooms",
    "stories",
    "parking",
    "mainroad"
]

TEST_SIZE = 0.2
RANDOM_STATE = 42
EXPERIMENT_NAME = "House_Price_Training_Pipeline"
