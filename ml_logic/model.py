import tensorflow as tf
from tensorflow.keras import models, layers
#input format (224, 224, 3)

def create_compile_model_fredi():
    base_model = tf.keras.applications.EfficientNetB3(
        include_top=False,   # Removes default classification head (we add our own)
        weights="imagenet",  # Use pre-trained weights (better for small datasets)
        input_shape=(224, 224, 3),  # Match input image size
        classifier_activation=None  # Avoids auto softmax (we define our own layers)
    )

    model = models.Sequential([
        base_model,
        layers.GlobalAveragePooling2D(),
        layers.Dense(1024, activation='swish'),
        layers.Dropout(0.4),  # Regularization to prevent overfitting
        layers.Dense(512, activation='swish'),
        layers.Dropout(0.3),
        layers.Dense(256, activation='swish'),  # Intermediate layer for better feature extraction
        layers.Dense(15, activation='softmax')  # Output layer for classification
    ])
    optimizer = tf.keras.optimizers.AdamW(learning_rate=0.0005, weight_decay=1e-4)

    # Compile model with label smoothing & AdamW
    model.compile(
        optimizer=optimizer,
        loss=tf.keras.losses.CategoricalCrossentropy(label_smoothing=0.1),
        metrics=['accuracy', 'precision', 'recall'])

    return model
