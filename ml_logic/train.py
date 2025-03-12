import model
from tensorflow.keras.callbacks import EarlyStopping
from keras.utils import image_dataset_from_directory
import params
import os
import registry


def run_train():
    print(f"### Start training")
    print(f"Models: {os.listdir(params.LOCAL_MODEL_PATH)}")

    # Create model
    m = model.create_compile_model_fredi()

    # Load data
    data_train, data_val = load_dataset()

    # Train
    es = EarlyStopping(patience=2)
    result = m.fit(
        data_train,
        batch_size=32,
        epochs=1,
        validation_data=data_val
    )
    #m.save(os.path.join(params.LOCAL_MODEL_PATH, '0001.h5'))
    registry.save_model(m)
    print(result)


def load_dataset():
    print(f"### Load Dataset")
    data_train = image_dataset_from_directory(
        directory=params.DATA_DIR,
        labels='inferred',
        label_mode='categorical',
        batch_size=32,
        image_size=(224, 224),
        validation_split=0.3,  # 30% of data will be used for validation
        subset='training',  # Explicitly specify training subset
        seed=123  # Ensure the same split each time
    )

    # Validation dataset
    data_val = image_dataset_from_directory(
        directory=params.DATA_DIR,
        labels='inferred',
        label_mode='categorical',
        batch_size=32,
        image_size=(224, 224),
        validation_split=0.3,  # Same 30% validation split
        subset='validation',  # Explicitly specify validation subset
        seed=123  # Same seed ensures consistency
    )

    return data_train, data_val


if __name__ == '__main__':
    run_train()
