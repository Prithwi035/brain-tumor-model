"""
Classification model: EfficientNetB3 fine-tuned for 4-class brain tumor classification.
Also includes a lightweight custom CNN as a baseline.
"""
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, Model
from config import IMG_SIZE, NUM_CLASSES, CHANNELS, LEARNING_RATE


# ─────────────────────────────────────────────────────────────────────────────
# EfficientNetB3 Transfer-learning model  (recommended)
# ─────────────────────────────────────────────────────────────────────────────

def build_efficientnet_classifier(img_size: tuple = IMG_SIZE,
                                   num_classes: int = NUM_CLASSES,
                                   freeze_base: bool = True,
                                   dropout_rate: float = 0.4) -> Model:
    """
    EfficientNetB3 backbone (ImageNet weights) with a custom classification head.
    Phase 1: train only the head (freeze_base=True)
    Phase 2: fine-tune last N layers (freeze_base=False)
    """
    inputs = keras.Input(shape=(*img_size, CHANNELS), name="mri_input")

    # EfficientNetB3 expects [0, 255] uint8-like but works fine with [0,1] float
    # We scale back to [0, 255] via Rescaling so the backbone stats match.
    x = layers.Rescaling(255.0)(inputs)

    base = keras.applications.EfficientNetB3(
        include_top=False,
        weights="imagenet",
        input_tensor=x,
        pooling=None
    )
    base.trainable = not freeze_base

    x = base.output
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dense(512, activation="relu")(x)
    x = layers.Dropout(dropout_rate)(x)
    x = layers.Dense(256, activation="relu")(x)
    x = layers.Dropout(dropout_rate / 2)(x)
    outputs = layers.Dense(num_classes, activation="softmax", name="predictions")(x)

    model = Model(inputs=inputs, outputs=outputs, name="EfficientNetB3_Classifier")
    return model


def compile_classifier(model: Model,
                        lr: float = LEARNING_RATE,
                        label_smoothing: float = 0.1) -> Model:
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=lr),
        loss=keras.losses.CategoricalCrossentropy(label_smoothing=label_smoothing),
        metrics=["accuracy",
                 keras.metrics.AUC(name="auc"),
                 keras.metrics.Precision(name="precision"),
                 keras.metrics.Recall(name="recall")]
    )
    return model


# ─────────────────────────────────────────────────────────────────────────────
# Lightweight custom CNN  (baseline / fallback)
# ─────────────────────────────────────────────────────────────────────────────

def build_custom_cnn(img_size: tuple = IMG_SIZE,
                      num_classes: int = NUM_CLASSES) -> Model:
    inputs = keras.Input(shape=(*img_size, CHANNELS))

    def conv_block(x, filters, pool=True):
        x = layers.Conv2D(filters, 3, padding="same", use_bias=False)(x)
        x = layers.BatchNormalization()(x)
        x = layers.Activation("relu")(x)
        x = layers.Conv2D(filters, 3, padding="same", use_bias=False)(x)
        x = layers.BatchNormalization()(x)
        x = layers.Activation("relu")(x)
        if pool:
            x = layers.MaxPooling2D(2)(x)
        return x

    x = conv_block(inputs, 32)
    x = conv_block(x, 64)
    x = conv_block(x, 128)
    x = conv_block(x, 256)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dense(256, activation="relu")(x)
    x = layers.Dropout(0.5)(x)
    outputs = layers.Dense(num_classes, activation="softmax")(x)

    model = Model(inputs=inputs, outputs=outputs, name="CustomCNN_Classifier")
    return model


# ─────────────────────────────────────────────────────────────────────────────
# Callbacks factory
# ─────────────────────────────────────────────────────────────────────────────

def get_callbacks(model_path: str, patience: int = 7):
    return [
        keras.callbacks.ModelCheckpoint(
            model_path,
            monitor="val_accuracy",
            save_best_only=True,
            verbose=1
        ),
        keras.callbacks.EarlyStopping(
            monitor="val_accuracy",
            patience=patience,
            restore_best_weights=True,
            verbose=1
        ),
        keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.5,
            patience=4,
            min_lr=1e-7,
            verbose=1
        ),
        # TensorBoard – only add if tensorboard package is available
        *([keras.callbacks.TensorBoard(log_dir="logs/classifier", histogram_freq=0)]
          if __import__("importlib").util.find_spec("tensorboard") else []),
    ]
