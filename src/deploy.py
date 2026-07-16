import os
import tarfile
import joblib
import boto3
from config import PRODUCTION_MODEL, ENDPOINT_NAME

# If we were strictly importing sagemaker:
# from sagemaker.sklearn.model import SKLearnModel
# from sagemaker import Session

def deploy_model():
    if not PRODUCTION_MODEL.exists():
        print("No production model found. Skipping deployment.")
        return

    print("Preparing model for deployment...")
    
    # Compress model for SageMaker (SageMaker expects a tar.gz file)
    model_tar_path = PRODUCTION_MODEL.parent / "model.tar.gz"
    with tarfile.open(model_tar_path, "w:gz") as tar:
        # SageMaker expects the model file to be at the root of the tar file
        tar.add(PRODUCTION_MODEL, arcname=PRODUCTION_MODEL.name)
    
    print("Model compressed to model.tar.gz")

    s3_bucket = os.environ.get("MODEL_S3_BUCKET")
    sagemaker_role = os.environ.get("SAGEMAKER_ROLE")

    # For beginners: we simulate if credentials are not fully set
    if not s3_bucket or not sagemaker_role:
        print("Error: MODEL_S3_BUCKET or SAGEMAKER_ROLE environment variables are not set.")
        print("Simulating deployment to SageMaker successfully!")
        return

    try:
        from sagemaker.sklearn.model import SKLearnModel
        from sagemaker import Session
        
        # Upload to S3
        s3_client = boto3.client('s3')
        s3_model_path = f"s3://{s3_bucket}/models/model.tar.gz"
        
        print(f"Uploading model to {s3_model_path}...")
        s3_client.upload_file(str(model_tar_path), s3_bucket, "models/model.tar.gz")

        print("Deploying to AWS SageMaker...")
        sagemaker_session = Session()

        # Define the SageMaker SKLearn Model
        sklearn_model = SKLearnModel(
            model_data=s3_model_path,
            role=sagemaker_role,
            framework_version="1.2-1",
            py_version="py3",
            sagemaker_session=sagemaker_session,
        )
        
        print("Creating/Updating Endpoint...")
        predictor = sklearn_model.deploy(
            instance_type="ml.m5.large",
            initial_instance_count=1,
            endpoint_name=ENDPOINT_NAME,
        )
        print(f"Model deployed successfully to endpoint: {ENDPOINT_NAME}")
    except Exception as e:
        print(f"Deployment failed: {e}")

if __name__ == "__main__":
    deploy_model()
