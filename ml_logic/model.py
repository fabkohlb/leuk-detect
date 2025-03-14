import tensorflow as tf
from tensorflow.keras import models, layers


def create_compile_model_fredi():
    base_model = tf.keras.applications.ResNet50(
        include_top=False,
        weights="imagenet",
        input_shape=(224, 224, 3),
    )
    model = models.Sequential([
        base_model,                                        # Add the ResNet50 base model
        layers.GlobalAveragePooling2D(),                    # Add pooling layer to reduce feature map size
        layers.Dense(1024, activation='relu'),              # Add a fully connected hidden layer (optional)
        layers.Dropout(0.5),                                # Add a dropout layer to reduce overfitting
        layers.Dense(15, activation='softmax')              # Custom output layer (e.g., for 10 classes, use softmax for multi-class classification)
    ])

    # Step 4: Compile the model
    model.compile(optimizer='adam',
                loss='categorical_crossentropy',  # Change this depending on your problem (e.g., 'binary_crossentropy' for binary classification)
                metrics=['accuracy', 'precision', 'recall'])

    return model
