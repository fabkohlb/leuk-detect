import tensorflow as tf
from tensorflow.keras import models, layers
from tensorflow.keras.applications import EfficientNetB3


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


def create_model_fredi_2(num_classes):
    # Load EfficientNetB3 as the feature extractor
    base_model = EfficientNetB3(
        include_top=False,
        weights="imagenet",
        input_shape=(224, 224, 3)
    )

    base_model.trainable = False  # Freeze pre-trained weights initially

    model = models.Sequential([
        base_model,
        layers.GlobalAveragePooling2D(),
        layers.BatchNormalization(),  # Helps stabilize training
        layers.Dense(1024, activation='swish'),
        layers.Dropout(0.5),  # Reduces overfitting
        layers.Dense(num_classes, activation='softmax')  # Output layer for multi-class classification
    ])

    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001),
                  loss='categorical_crossentropy',
                  metrics=['accuracy', 'Precision', 'Recall'])

    return model
