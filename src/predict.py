import joblib
import pandas as pd
from config import (
    PRODUCTION_MODEL,
    FEATURES,
)

def load_model():
    if not PRODUCTION_MODEL.exists():
        raise FileNotFoundError(
            "Production model not found."
        )
    return joblib.load(PRODUCTION_MODEL)


def predict(data):
    model = load_model()
    df = pd.DataFrame(data)
    df["pclass"] = df["pclass"].astype(str)
    df = df[FEATURES]
    predictions = model.predict(df)
    probabilities = model.predict_proba(df)
    return predictions, probabilities


if __name__ == "__main__":
    sample = {
        "pclass": [3],
        "sex": ["male"],
        "age": [25],
        "sibsp": [0],
        "parch": [0],
        "fare": [8.05],
        "embarked": ["S"],
    }

    pred, prob = predict(sample)
    print()
    print(
        f"Prediction : {'Survived' if pred[0] else 'Died'}"
    )
    print(
        f"Probability: {prob[0][1]:.4f}"
    )