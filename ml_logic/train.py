import model
from tensorflow.keras.callbacks import EarlyStopping
from keras.utils import image_dataset_from_directory
from sklearn.utils.class_weight import compute_class_weight
import params
import registry
import time
import cv2
import os
import numpy as np


def run_train():
    print(f"▶️ Start training")
    start_time = time.time()

    # Load data
    data_train, data_val = load_dataset()

    print(f"Len {len(data_train)}")
    print(data_train.class_names)
    labels_list = []
    for images, labels in data_train:  # Loop through batches
        labels_list.extend(np.argmax(labels.numpy(), axis=1))
    print(f"Labels list {np.unique(labels_list)}")

    # Compute class weights
    class_weights = compute_class_weight(
        'balanced',
        classes=np.arange(len(data_train.class_names)),
        y=labels_list
    )
    class_weights = {i: class_weights[i] for i in range(len(data_train.class_names))}
    print(f"Class weights: {class_weights}")

    # Create model
    m = model.create_compile_model_fredi(len(data_train.class_names))

    # Train
    es = EarlyStopping(patience=3, restore_best_weights=True)
    history = m.fit(
        data_train,
        batch_size=params.BATCH_SIZE,
        epochs=params.EPOCHS,
        validation_data=data_val,
        class_weight=class_weights,
        callbacks=[es]
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
