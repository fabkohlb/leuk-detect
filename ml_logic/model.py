import tensorflow as tf
from tensorflow.keras import models, layers



def create_compile_model_fredi():
    # Data augmentation with more aggressive transformations
    augmenting = models.Sequential([
        layers.Rescaling(1./255),
        layers.RandomFlip("horizontal_and_vertical"),
        layers.RandomRotation(0.3),  # Increased rotation
        layers.RandomZoom(0.2),      # Increased zoom
        layers.RandomTranslation(0.1, 0.1),  # Increased translation
        layers.RandomBrightness(0.2),  # Added brightness adjustment
        layers.RandomContrast(0.2),    # Added contrast adjustment
    ])

    # Using a different EfficientNet version with custom input size
    base_model = tf.keras.applications.EfficientNetB5(  # B5 instead of B7 - better speed/accuracy trade-off
        include_top=False,
        weights="imagenet",
        input_shape=(224, 224, 3),  # Slightly larger input size
    )
    base_model.trainable = False  # Freeze base model initially

    # Modified fully connected layers with L2 regularization
    fully_connected = models.Sequential([
        layers.GlobalAveragePooling2D(),
        layers.Dense(1024, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.01)),
        layers.BatchNormalization(),
        layers.Dropout(0.4),
        layers.Dense(512, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.01)),
        layers.BatchNormalization(),
        layers.Dropout(0.5),
        layers.Dense(15, activation='softmax')
    ])

    model = models.Sequential([
        augmenting,
        base_model,
        fully_connected
    ])

    # Using a learning rate scheduler and different optimizer
    initial_learning_rate = 0.001
    lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
        initial_learning_rate, decay_steps=1000, decay_rate=0.9
    )
    optimizer = tf.keras.optimizers.Adam(learning_rate=lr_schedule)

    model.compile(
        optimizer=optimizer,
        loss='categorical_crossentropy',
        metrics=['accuracy', 'precision', 'recall']
    )
    return model
