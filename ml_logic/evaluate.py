import registry
from keras.utils import image_dataset_from_directory
import tensorflow as tf


def evaluate_latest_model():
    print("Start evaluation of the model")
    model = registry.load_latest_model()

def _eval_model(model):



if __name__ == '__main__':
    run_evaluation()
