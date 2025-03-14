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


def load_model(model_name):
    # Check if model is locally available
    if not os.path.exists(os.path.join(params.LOCAL_MODEL_PATH, model_name)):
        print("Model not available locally")
    print("Load model from GCS...")
    client = storage.Client()
    blobs = list(client.get_bucket(params.BUCKET_NAME).list_blobs(prefix="model"))
    for blob in blobs:
        if blob.name.split('/')[-1] == model_name:
            print(f"Found model in bucket: {blob.name}")
    exit()

    try:
        model_path_to_save = os.path.join(params.LOCAL_MODEL_PATH, model_name)
        # latest_blob.download_to_filename(latest_model_path_to_save)
        # latest_model = keras.models.load_model(latest_model_path_to_save)

        print("✅ Latest model downloaded from cloud storage")
    except Exception as e:
        print(f"\n❌ No model found in GCS bucket {params.BUCKET_NAME}")
        print(f"Exception: {e}")
        return None


def load_latest_model():
    print("Load latest model from GCS...")
    client = storage.Client()
    blobs = list(client.get_bucket(params.BUCKET_NAME).list_blobs(prefix="model"))
    print(f"Blobs: {blobs}")

    try:
        latest_blob = max(blobs, key=lambda x: x.updated)
        latest_model_path_to_save = os.path.join(params.LOCAL_MODEL_PATH, latest_blob.name.split('/')[-1])
        latest_blob.download_to_filename(latest_model_path_to_save)
        latest_model = keras.models.load_model(latest_model_path_to_save)

        print("✅ Latest model downloaded from cloud storage")
        return latest_model
    except Exception as e:
        print(f"\n❌ No model found in GCS bucket {params.BUCKET_NAME}")
        print(f"Exception: {e}")
        return None

@mlflow_run
def save_model(model, history, duration_sec):
    timestamp = time.strftime("%Y%m%d-%H%M%S")
    client = MlflowClient()

    # Save model locally
    model_path = os.path.join(params.LOCAL_MODEL_PATH, f"{timestamp}.keras")
    model.save(model_path)
    print("✅ Model saved locally")

    # Save on google cloud
    model_filename = model_path.split("/")[-1]
    client = storage.Client()
    bucket = client.bucket(params.BUCKET_NAME)
    blob = bucket.blob(f"models/{model_filename}")
    blob.upload_from_filename(model_path)
    print("✅ Model saved to GCS")

    # Safe to ml flow
    p = {
        "optimizer": model.optimizer.get_config(),
        "loss": model.loss,
        "epochs": len(history.epoch),
    }
    print(f"Log params: {p}")
    mlflow.log_params({
        "optimizer": model.optimizer.get_config()['name'],
        "optimizer": model.optimizer.get_config()['learning_rate'],
        "optimizer": model.optimizer.get_config()['name'],
        "loss": model.loss,
        "epochs": len(history.epoch),
    })
    mlflow.log_text(model.to_json(), "model_architecture.json")

    # Save history
    for metric in history.history.keys():
        for epoch, value in enumerate(history.history[metric]):
            mlflow.log_metric(metric, value, step=epoch)
    mlflow.log_param('model_filename', model_filename)

    minutes, seconds = divmod(duration_sec, 60)
    hours, minutes = divmod(minutes, 60)
    mlflow.log_param("training_duration_hms", f"{int(hours)}h {int(minutes)}m {int(seconds)}s")

    print("✅ Metadata saved to mlflow")

    # Register a new version with only metadata (no actual model file)
    # model_version = client.create_model_version(
    #     name=params.MLFLOW_MODEL_NAME,
    #     source=blob
    # )

    return None



if __name__ == '__main__':
    from pathlib import Path
    model = load_model()
    if model is not None:
        model.save('./models/test.keras')
