import model
from tensorflow.keras.callbacks import EarlyStopping
from keras.utils import image_dataset_from_directory
import params
import registry
import time
import cv2
import os


def run_train():
    print(f"▶️ Start training")
    start_time = time.time()

    # Create model
    m = model.create_compile_model_fredi()

    # Load data
    data_train, data_val = load_dataset()

    # Train
    es = EarlyStopping(patience=2)
    history = m.fit(
        data_train,
        batch_size=params.BATCH_SIZE,
        epochs=params.EPOCHS,
        validation_data=data_val,
    )
    training_duration = time.time() - start_time
    print(f"✅ Training complete in {(training_duration/60):.2f} minutes")
    registry.save_model(m, history, training_duration)


def load_dataset():
    print(f"### Load Dataset")

    data_train = image_dataset_from_directory(
        directory=os.path.join(params.DATA_DIR, 'train'),
        labels='inferred',
        label_mode='categorical',
        batch_size=params.BATCH_SIZE,
        image_size=(224, 224),
    )

    # Validation dataset
    data_val = image_dataset_from_directory(
        directory=os.path.join(params.DATA_DIR, 'validation'),
        labels='inferred',
        label_mode='categorical',
        batch_size=params.BATCH_SIZE,
        image_size=(224, 224),
    )

    return data_train, data_val


if __name__ == '__main__':
    run_train()
