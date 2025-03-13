import model
from tensorflow.keras.callbacks import EarlyStopping
from keras.utils import image_dataset_from_directory
import params
import registry
import time


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
        batch_size=32,
        epochs=1,
        validation_data=data_val,
    )
    training_duration = time.time() - start_time
    print(f"✅ Training complete in {(training_duration/60):.2f} minutes")
    registry.save_model(m, history, training_duration)


def load_dataset():
    print(f"### Load Dataset")

    val_split = 0.3

    data_train = image_dataset_from_directory(
        directory=params.DATA_DIR,
        labels='inferred',
        label_mode='categorical',
        batch_size=32,
        image_size=(224, 224),
        validation_split=val_split,
        subset='training',
        seed=123
    )

    # Validation dataset
    data_val = image_dataset_from_directory(
        directory=params.DATA_DIR,
        labels='inferred',
        label_mode='categorical',
        batch_size=32,
        image_size=(224, 224),
        validation_split=val_split,  # Same 30% validation split
        subset='validation',  # Explicitly specify validation subset
        seed=123  # Same seed ensures consistency
    )

    return data_train, data_val


if __name__ == '__main__':
    run_train()
