from tensorflow import keras
from google.cloud import storage
import os
import params
import time

import mlflow
from mlflow.tracking import MlflowClient
import tensorflow as tf


def mlflow_run(func):
    """
    Generic function to log params and results to MLflow along with TensorFlow auto-logging

    Args:
        - func (function): Function you want to run within the MLflow run
        - params (dict, optional): Params to add to the run in MLflow. Defaults to None.
        - context (str, optional): Param describing the context of the run. Defaults to "Train".
    """
    def wrapper(*args, **kwargs):
        mlflow.end_run()
        mlflow.set_tracking_uri(params.MLFLOW_TRACKING_URI)
        mlflow.set_experiment(experiment_name=params.MLFLOW_EXPERIMENT)

        with mlflow.start_run():
            mlflow.tensorflow.autolog()
            results = func(*args, **kwargs)

        print("✅ mlflow_run auto-log done")

        return results
    return wrapper


def load_model():
    print("Load latest model from GCS...")
    client = storage.Client()
    blobs = list(client.get_bucket(params.BUCKET_NAME).list_blobs(prefix="model"))

    print(f"Blobs: {blobs}")

    try:
        latest_blob = max(blobs, key=lambda x: x.updated)
        latest_model_path_to_save = os.path.join(params.LOCAL_MODEL_PATH, latest_blob.name)
        latest_blob.download_to_filename(latest_model_path_to_save)
        latest_model = keras.models.load_model(latest_model_path_to_save)

        print("✅ Latest model downloaded from cloud storage")

        return latest_model
    except:
        print(f"\n❌ No model found in GCS bucket {params.BUCKET_NAME}")

        return None

@mlflow_run
def save_model(model):
    timestamp = time.strftime("%Y%m%d-%H%M%S")

    # Save model locally
    # model_path = os.path.join(params.LOCAL_MODEL_PATH, f"{timestamp}.h5")
    # model.save(model_path)
    # print("✅ Model saved locally")

    # Save on google cloud
    # model_filename = model_path.split("/")[-1] # e.g. "20230208-161047.h5" for instance
    # client = storage.Client()
    # bucket = client.bucket(params.BUCKET_NAME)
    # blob = bucket.blob(f"models/{model_filename}")
    # blob.upload_from_filename(model_path)
    # print("✅ Model saved to GCS")

    # Safe to ml flow
    print(model.summary())
    mlflow.tensorflow.log_model(
        model=model,
        artifact_path="model",
        registered_model_name=params.MLFLOW_MODEL_NAME
    )
    print("✅ Model saved to mlflow")

    return None



if __name__ == '__main__':
    from pathlib import Path
    modelpath="/Users/fredi/code/fgeb/08-blood-cancer-prediction-model/leuk-detect/models/20250312-144801.h5"
    loaded_model = keras.models.load_model(Path(modelpath))
    save_model(loaded_model)
