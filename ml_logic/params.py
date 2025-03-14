import os

##################  VARIABLES  ##################
DATA_DIR = os.environ.get("DATA_DIR")
LOCAL_MODEL_PATH = os.environ.get("LOCAL_MODEL_PATH")

BUCKET_NAME = os.environ.get("BUCKET_NAME")
BUCKET = os.environ.get("BUCKET")
INSTANCE = os.environ.get("INSTANCE")

MLFLOW_TRACKING_URI = os.environ.get('MLFLOW_TRACKING_URI')
MLFLOW_EXPERIMENT = os.environ.get('MLFLOW_EXPERIMENT')
MLFLOW_MODEL_NAME = os.environ.get('MLFLOW_MODEL_NAME')

BATCH_SIZE = int(os.environ.get('BATCH_SIZE'))
EPOCHS = int(os.environ.get('EPOCHS'))
VALIDATION_SPLIT = float(os.environ.get('VALIDATION_SPLIT'))
