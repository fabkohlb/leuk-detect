import os
import numpy as np

##################  VARIABLES  ##################
DATA_DIR = os.environ.get("DATA_DIR")
LOCAL_MODEL_PATH = os.environ.get("LOCAL_MODEL_PATH")

BUCKET_NAME = os.environ.get("BUCKET_NAME")
INSTANCE = os.environ.get("INSTANCE")

MLFLOW_TRACKING_URI = os.environ.get('MLFLOW_TRACKING_URI')
MLFLOW_EXPERIMENT = os.environ.get('MLFLOW_EXPERIMENT')
MLFLOW_MODEL_NAME = os.environ.get('MLFLOW_MODEL_NAME')
