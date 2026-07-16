import shutil
import joblib
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

from config import (
    TRAIN_DATA,
    TARGET,
    FEATURES,
    TEST_SIZE,
    RANDOM_STATE,
    CANDIDATE_MODEL,
    PRODUCTION_MODEL,
    MIN_IMPROVEMENT,
)

def load_test_data():
    df = pd.read_csv(TRAIN_DATA)
    df["pclass"] = df["pclass"].astype(str)
    X = df[FEATURES]
    y = df[TARGET]
    _, X_test, _, y_test = train_test_split(
        X,
        y,
        test_size=TEST_SIZE,
        stratify=y,
        random_state=RANDOM_STATE,
    )
    return X_test, y_test


def evaluate():

    X_test, y_test = load_test_data()
    candidate = joblib.load(CANDIDATE_MODEL)
    candidate_preds = candidate.predict(X_test)
    candidate_acc = accuracy_score(
        y_test,
        candidate_preds,
    )

    print(f"\nCandidate Accuracy: {candidate_acc:.4f}")
    if not PRODUCTION_MODEL.exists():
        print("\nNo production model found.")
        shutil.copy(
            CANDIDATE_MODEL,
            PRODUCTION_MODEL,
        )
        print("Candidate promoted to production.")
        return True

    production = joblib.load(PRODUCTION_MODEL)
    production_preds = production.predict(X_test)
    production_acc = accuracy_score(
        y_test,
        production_preds,
    )

    print(f"Production Accuracy: {production_acc:.4f}")
    improvement = candidate_acc - production_acc
    print(f"Improvement: {improvement:.4f}")

    if improvement >= MIN_IMPROVEMENT:
        print("\nNew model is better.")
        shutil.copy(
            CANDIDATE_MODEL,
            PRODUCTION_MODEL,
        )
        print("Production model updated.")
        return True

    else:
        print("\nCandidate rejected.")
        return False
    
if __name__ == "__main__":
    evaluate()