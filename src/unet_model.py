"""
U-Net segmentation model for brain tumor region detection.
Architecture: Encoder-Decoder with skip connections.
Input:  (256, 256, 3)  grayscale MRI replicated to 3 channels
Output: (256, 256, 1)  binary tumor mask
"""
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, Model
from config import SEG_IMG_SIZE, SEG_LR, CHANNELS


# ─────────────────────────────────────────────────────────────────────────────
# Custom metrics for segmentation
# ─────────────────────────────────────────────────────────────────────────────

def dice_coefficient(y_true, y_pred, smooth: float = 1e-6):
    y_true_f = tf.reshape(tf.cast(y_true, tf.float32), [-1])
    y_pred_f = tf.reshape(tf.cast(y_pred, tf.float32), [-1])
    intersection = tf.reduce_sum(y_true_f * y_pred_f)
    return (2.0 * intersection + smooth) / (
        tf.reduce_sum(y_true_f) + tf.reduce_sum(y_pred_f) + smooth
    )


def dice_loss(y_true, y_pred):
    return 1.0 - dice_coefficient(y_true, y_pred)


def combined_loss(y_true, y_pred, bce_weight: float = 0.5):
    bce = keras.losses.binary_crossentropy(y_true, y_pred)
    dl  = dice_loss(y_true, y_pred)
    return bce_weight * bce + (1 - bce_weight) * dl


def iou_score(y_true, y_pred, smooth: float = 1e-6, threshold: float = 0.5):
    y_pred_bin = tf.cast(y_pred > threshold, tf.float32)
    y_true_f   = tf.reshape(tf.cast(y_true, tf.float32), [-1])
    y_pred_f   = tf.reshape(y_pred_bin, [-1])
    intersection = tf.reduce_sum(y_true_f * y_pred_f)
    union        = tf.reduce_sum(y_true_f) + tf.reduce_sum(y_pred_f) - intersection
    return (intersection + smooth) / (union + smooth)


# ─────────────────────────────────────────────────────────────────────────────
# U-Net building blocks
# ─────────────────────────────────────────────────────────────────────────────

def conv_block(x, filters: int, dropout_rate: float = 0.0):
    x = layers.Conv2D(filters, 3, padding="same", use_bias=False)(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)
    if dropout_rate > 0:
        x = layers.Dropout(dropout_rate)(x)
    x = layers.Conv2D(filters, 3, padding="same", use_bias=False)(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)
    return x


def encoder_block(x, filters: int, dropout_rate: float = 0.0):
    skip = conv_block(x, filters, dropout_rate)
    down = layers.MaxPooling2D(2)(skip)
    return skip, down


def decoder_block(x, skip, filters: int):
    x = layers.Conv2DTranspose(filters, 2, strides=2, padding="same")(x)
    x = layers.Concatenate()([x, skip])
    x = conv_block(x, filters)
    return x


# ─────────────────────────────────────────────────────────────────────────────
# U-Net model
# ─────────────────────────────────────────────────────────────────────────────

def build_unet(img_size: tuple = SEG_IMG_SIZE,
               channels: int = CHANNELS,
               base_filters: int = 64,
               depth: int = 4,
               dropout_rate: float = 0.3) -> Model:
    """
    Configurable U-Net.
    depth=4 → 4 encoder stages, bridge, 4 decoder stages.
    base_filters doubles each level: 64 → 128 → 256 → 512.
    """
    inputs = keras.Input(shape=(*img_size, channels), name="mri_seg_input")

    skips = []
    x = inputs
    # Encoder
    for i in range(depth):
        f = base_filters * (2 ** i)
        dr = dropout_rate if i >= depth - 2 else 0.0
        skip, x = encoder_block(x, f, dr)
        skips.append(skip)

    # Bridge
    x = conv_block(x, base_filters * (2 ** depth), dropout_rate)

    # Decoder
    for i in reversed(range(depth)):
        f = base_filters * (2 ** i)
        x = decoder_block(x, skips[i], f)

    # Output head
    outputs = layers.Conv2D(1, 1, activation="sigmoid", name="seg_output")(x)

    model = Model(inputs=inputs, outputs=outputs, name="UNet_Segmentation")
    return model


def compile_unet(model: Model, lr: float = SEG_LR) -> Model:
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=lr),
        loss=combined_loss,
        metrics=[dice_coefficient, iou_score,
                 keras.metrics.BinaryAccuracy(name="bin_acc")]
    )
    return model


# ─────────────────────────────────────────────────────────────────────────────
# Callbacks
# ─────────────────────────────────────────────────────────────────────────────

def get_seg_callbacks(model_path: str, patience: int = 8):
    return [
        keras.callbacks.ModelCheckpoint(
            model_path,
            monitor="val_dice_coefficient",
            save_best_only=True,
            mode="max",
            verbose=1
        ),
        keras.callbacks.EarlyStopping(
            monitor="val_dice_coefficient",
            patience=patience,
            mode="max",
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
        *([keras.callbacks.TensorBoard(log_dir="logs/unet", histogram_freq=0)]
          if __import__("importlib").util.find_spec("tensorboard") else []),
    ]
