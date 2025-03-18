import model
from tensorflow.keras.callbacks import EarlyStopping
from keras.utils import image_dataset_from_directory
import params
import registry
import time
import cv2
import os
import numpy as np
from sklearn.utils.class_weight import compute_class_weight
import pathlib


def run_train():
    print(f"▶️ Start training")
    start_time = time.time()

    # Create model
    m = model.create_compile_model_fredi()

    # Load data
    data_train, data_val = load_dataset()

    data_dir = pathlib.Path(os.path.join(params.DATA_DIR, 'train')
)
    class_names = [item.name for item in data_dir.glob('*') if item.is_dir()]
    class_counts = []

    for class_name in class_names:
        class_path = data_dir / class_name
        count = len(list(class_path.glob('*')))
        class_counts.append(count)
        print(f"Class {class_name}: {count} images")

    # Calculate weights inversely proportional to class frequencies
    total_images = sum(class_counts)
    class_weights = {i: total_images / (len(class_names) * count)
                            for i, count in enumerate(class_counts)}
    class_weight_dict = dict(zip(range(len(class_names)), class_weights))
    print(class_weight_dict)

    # Train
    es = EarlyStopping(patience=2)
    history = m.fit(
        data_train,
        batch_size=params.BATCH_SIZE,
        epochs=params.EPOCHS,
        validation_data=data_val,
        class_weight=class_weight_dict
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
