import joblib
import pandas as pd
import json
import boto3
from config import (
    PRODUCTION_MODEL,
    FEATURES,
    ENDPOINT_NAME
)

def load_local_model():
    if not PRODUCTION_MODEL.exists():
        raise FileNotFoundError(
            "Production model not found."
        )
    return joblib.load(PRODUCTION_MODEL)

def predict_local(data):
    model = load_local_model()
    df = pd.DataFrame(data)
    df["pclass"] = df["pclass"].astype(str)
    df = df[FEATURES]
    predictions = model.predict(df)
    probabilities = model.predict_proba(df)
    return predictions, probabilities

def predict_sagemaker(data):
    """
    Invokes the deployed AWS SageMaker endpoint for inference.
    Requires AWS credentials to be configured in your environment.
    """
    # sagemaker-runtime is the client used specifically for invoking endpoints
    client = boto3.client('sagemaker-runtime')
    
    df = pd.DataFrame(data)
    df["pclass"] = df["pclass"].astype(str)
    df = df[FEATURES]
    
    # SageMaker's default Scikit-Learn container accepts CSV data natively
    csv_data = df.to_csv(header=False, index=False)
    
    response = client.invoke_endpoint(
        EndpointName=ENDPOINT_NAME,
        ContentType='text/csv',
        Body=csv_data
    )
    
    # The response body will contain the predictions array
    result = json.loads(response['Body'].read().decode())
    return result

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

    print("--- LOCAL INFERENCE ---")
    try:
        pred, prob = predict_local(sample)
        print(f"Prediction : {'Survived' if pred[0] else 'Died'}")
        print(f"Probability: {prob[0][1]:.4f}")
    except FileNotFoundError:
        print("Local model not found. Pipeline might not have completed a local run.")
    except Exception as e:
        print(f"Local prediction failed: {e}")

    print("\n--- SAGEMAKER INFERENCE ---")
    print(f"Attempting to call Endpoint: {ENDPOINT_NAME}")
    try:
        sm_result = predict_sagemaker(sample)
        print(f"SageMaker Prediction Result: {sm_result}")
    except Exception as e:
        print(f"Could not connect to SageMaker Endpoint. Reason:\n{e}")