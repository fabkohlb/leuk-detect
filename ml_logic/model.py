import tensorflow as tf
from tensorflow.keras import models, layers



def create_compile_model_fredi():
    augmenting = models.Sequential([
        layers.experimental.preprocessing.Rescaling(1./255),
        layers.experimental.preprocessing.RandomFlip("horizontal_and_vertical"),
        layers.experimental.preprocessing.RandomRotation(0.2),
        layers.experimental.preprocessing.RandomZoom(0.05),
        layers.experimental.preprocessing.RandomTranslation(0.05, 0.05),
    ])

    base_model = tf.keras.applications.ResNet50(
        include_top=False,
        weights="imagenet",
        input_shape=(224, 224, 3),
    )

    fully_connected = models.Sequential([
        layers.GlobalAveragePooling2D(),
        layers.Dense(1024, activation='relu'),
        layers.BatchNormalization(),
        layers.Dropout(0.3),
        layers.Dense(512, activation='relu'),
        layers.BatchNormalization(),
        layers.Dropout(0.5),
        layers.Dense(15, activation='softmax')
    ])
    model = models.Sequential([
        augmenting,
        base_model,                                         # Add the base model
        fully_connected
    ])

    # Step 4: Compile the model
    model.compile(optimizer='adam',
                loss='categorical_crossentropy',  # Change this depending on your problem (e.g., 'binary_crossentropy' for binary classification)
                metrics=['accuracy', 'precision', 'recall'])

    return model
