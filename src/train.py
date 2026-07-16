import joblib
import pandas as pd
import mlflow
import mlflow.sklearn

from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
)

from pipeline import create_pipeline

from config import (
    MLFLOW_TRACKING_URI,
    TRAIN_DATA,
    NEW_DATA,
    MODEL_DIR,
    TARGET,
    FEATURES,
    TEST_SIZE,
    RANDOM_STATE,
    RF_PARAMS,
    EXPERIMENT_NAME,
)


def load_training_data():
    train = pd.read_csv(TRAIN_DATA)
    if NEW_DATA.exists():
        print("New training data found.")
        new = pd.read_csv(NEW_DATA)
        train = pd.concat(
            [train, new],
            ignore_index=True,
        )
    return train


def main():
    df = load_training_data()
    df["pclass"] = df["pclass"].astype(str)

    X = df[FEATURES]
    y = df[TARGET]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=TEST_SIZE,
        stratify=y,
        random_state=RANDOM_STATE,
    )

    pipe = create_pipeline()
    print("\nTraining model...\n")

    mlflow.set_experiment(EXPERIMENT_NAME)

    with mlflow.start_run():
        mlflow.log_params(RF_PARAMS)
        mlflow.log_param(
            "training_samples",
            len(X_train),
        )

        mlflow.log_param(
            "testing_samples",
            len(X_test),
        )

        mlflow.log_param(
            "features",
            ",".join(FEATURES),
        )
        pipe.fit(X_train, y_train)

        preds = pipe.predict(X_test)

        accuracy = accuracy_score(
            y_test,
            preds,
        )

        precision = precision_score(
            y_test,
            preds,
        )

        recall = recall_score(
            y_test,
            preds,
        )

        f1 = f1_score(
            y_test,
            preds,
        )

        mlflow.log_metric(
            "accuracy",
            accuracy,
        )

        mlflow.log_metric(
            "precision",
            precision,
        )

        mlflow.log_metric(
            "recall",
            recall,
        )

        mlflow.log_metric(
            "f1_score",
            f1,
        )

        mlflow.sklearn.log_model(
            sk_model=pipe,
            artifact_path="model",
        )

    MODEL_DIR.mkdir(exist_ok=True)

    model_path = MODEL_DIR / "candidate_model.pkl"

    joblib.dump(
        pipe,
        model_path,
    )

    print("Training Complete\n")

    print(f"Accuracy : {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall   : {recall:.4f}")
    print(f"F1 Score : {f1:.4f}")

    print(f"\nCandidate model saved to:\n{model_path}")


if __name__ == "__main__":
    main()