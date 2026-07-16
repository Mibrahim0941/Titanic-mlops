from pathlib import Path
import os

ROOT_DIR = Path(__file__).resolve().parent.parent

DATA_DIR = ROOT_DIR / "data"
MODEL_DIR = ROOT_DIR / "models"

TRAIN_DATA = DATA_DIR / "train.csv"
NEW_DATA = DATA_DIR / "new_data.csv"

MODEL_FILE = MODEL_DIR / "titanic_rf.pkl"
PRODUCTION_MODEL = MODEL_DIR / "production_model.pkl"
CANDIDATE_MODEL = MODEL_DIR / "candidate_model.pkl"

MIN_IMPROVEMENT = 0.001

TARGET = "survived"

NUMERICAL_FEATURES = [
    "age",
    "fare"
]

CATEGORICAL_FEATURES = [
    "sex",
    "embarked",
    "pclass"
]

FEATURES = [
    "pclass",
    "sex",
    "age",
    "sibsp",
    "parch",
    "fare",
    "embarked"
]

TEST_SIZE = 0.20
RANDOM_STATE = 42

RF_PARAMS = {
    "n_estimators": 200,
    "max_depth": 8,
    "min_samples_split": 4,
    "min_samples_leaf": 2,
    "random_state": RANDOM_STATE,
    "n_jobs": -1
}

AGE_BINS = 4

MIN_ACCURACY = 0.80

MODEL_NAME = "TitanicRandomForest"

ENDPOINT_NAME = "titanic-endpoint-2"

DEPLOYMENT_NAME = "blue"

EXPERIMENT_NAME = "Titanic-RandomForest"