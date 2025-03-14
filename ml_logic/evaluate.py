import registry
import params
import os
from keras.utils import image_dataset_from_directory
import tensorflow as tf


def evaluate_model(model_name):
    print("Start evaluation of the model")
    model = registry.load_model(model_name)
    #_eval_model(model)

def _eval_model(model):
    data = image_dataset_from_directory(
        directory=os.path.join(params.DATA_DIR, 'evaluation'),
        labels='inferred',
        label_mode='categorical',
        batch_size=params.BATCH_SIZE,
        image_size=(224, 224),
    )
    print("✅ Evaluation data loaded.")



if __name__ == '__main__':
    evaluate_model('20250314-005854.keras')
