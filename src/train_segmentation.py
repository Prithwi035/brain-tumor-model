"""
Training pipeline for U-Net brain tumor segmentation.

Dataset expected layout:
    data_root/
        images/   *.jpg|*.png   (MRI slices)
        masks/    *.jpg|*.png   (binary tumor masks, same filename)

Usage:
    python train_segmentation.py --data_root d:/My Files/Dev/brain_tumor_dev/data/raw
"""
import os, sys, argparse, numpy as np, cv2
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
sys.path.insert(0, os.path.dirname(__file__))

from glob import glob
from sklearn.model_selection import train_test_split
import tensorflow as tf
from config import *
from unet_model import build_unet, compile_unet, get_seg_callbacks
from evaluation import (
    evaluate_segmentation, visualize_segmentation,
    plot_training_history
)


def load_segmentation_dataset(data_root: str,
                               img_size: tuple = SEG_IMG_SIZE):
    """
    Load paired (image, mask) for segmentation.
    Looks for lgg-mri-segmentation layout OR flat images/masks directories.
    """
    img_paths, mask_paths = [], []

    # Try LGG dataset structure: patient folders with *_mask.tif
    lgg_imgs = glob(os.path.join(data_root, "**", "*.tif"), recursive=True)
    lgg_imgs = [p for p in lgg_imgs if "_mask" not in p]
    if lgg_imgs:
        for p in lgg_imgs:
            mask_p = p.replace(".tif", "_mask.tif")
            if os.path.exists(mask_p):
                img_paths.append(p)
                mask_paths.append(mask_p)
        print(f"Found {len(img_paths)} LGG-style paired images")
    else:
        # Flat images/masks directories
        img_dir  = os.path.join(data_root, "images")
        mask_dir = os.path.join(data_root, "masks")
        for ext in ["*.jpg", "*.jpeg", "*.png"]:
            for p in glob(os.path.join(img_dir, ext)):
                fname  = os.path.basename(p)
                mask_p = os.path.join(mask_dir, fname)
                if os.path.exists(mask_p):
                    img_paths.append(p)
                    mask_paths.append(mask_p)
        print(f"Found {len(img_paths)} paired images in images/masks directories")

    if not img_paths:
        raise FileNotFoundError(
            "\n\n❌  No paired image/mask files found under: " + data_root + "\n"
            "    For segmentation you need the LGG MRI Segmentation dataset:\n"
            "      https://www.kaggle.com/datasets/mateuszbuda/lgg-mri-segmentation\n\n"
            "    Expected layout (LGG style):\n"
            "      data/raw/<patient_folder>/<slice>.tif\n"
            "      data/raw/<patient_folder>/<slice>_mask.tif\n\n"
            "    OR flat layout:\n"
            "      data/raw/images/*.jpg\n"
            "      data/raw/masks/*.jpg   (same filenames)\n\n"
            "    ➡  Run:  python src/setup_dataset.py\n"
            "       to download both datasets automatically (needs Kaggle credentials),\n"
            "       OR see README.md for manual download instructions.\n"
        )

    X, M = [], []
    for ip, mp in zip(img_paths, mask_paths):
        img  = cv2.imread(ip) if ip.endswith((".jpg", ".jpeg", ".png")) else \
               _read_tif(ip)
        mask = cv2.imread(mp, cv2.IMREAD_GRAYSCALE) if mp.endswith((".jpg",".jpeg",".png")) else \
               _read_tif(mp, gray=True)
        if img is None or mask is None:
            continue
        img  = cv2.resize(cv2.cvtColor(img, cv2.COLOR_BGR2RGB), img_size)
        mask = cv2.resize(mask, img_size)
        X.append(img.astype(np.float32) / 255.0)
        M.append((mask > 127).astype(np.float32)[..., np.newaxis])

    X = np.array(X, dtype=np.float32)
    M = np.array(M, dtype=np.float32)
    print(f"Loaded {len(X)} image-mask pairs  |  X: {X.shape}  M: {M.shape}")
    return X, M


def _read_tif(path: str, gray: bool = False) -> np.ndarray:
    """Read .tif using cv2 (supports multi-page; take first channel)."""
    flags = cv2.IMREAD_GRAYSCALE if gray else cv2.IMREAD_COLOR
    img = cv2.imread(path, flags)
    return img


def build_seg_tf_datasets(X, M, batch_size: int = SEG_BATCH_SIZE):
    X_tv, X_test, M_tv, M_test = train_test_split(
        X, M, test_size=0.15, random_state=RANDOM_SEED)
    X_train, X_val, M_train, M_val = train_test_split(
        X_tv, M_tv, test_size=0.15, random_state=RANDOM_SEED)

    AUTOTUNE = tf.data.AUTOTUNE

    def augment(img, mask):
        # Random horizontal flip
        seed = tf.random.uniform([], maxval=2, dtype=tf.int32)
        img  = tf.cond(seed == 1, lambda: tf.image.flip_left_right(img),  lambda: img)
        mask = tf.cond(seed == 1, lambda: tf.image.flip_left_right(mask), lambda: mask)
        # Random vertical flip
        seed2 = tf.random.uniform([], maxval=2, dtype=tf.int32)
        img  = tf.cond(seed2 == 1, lambda: tf.image.flip_up_down(img),  lambda: img)
        mask = tf.cond(seed2 == 1, lambda: tf.image.flip_up_down(mask), lambda: mask)
        img  = tf.image.random_brightness(img, 0.1)
        img  = tf.clip_by_value(img, 0.0, 1.0)
        return img, mask

    train_ds = (tf.data.Dataset.from_tensor_slices((X_train, M_train))
                .shuffle(len(X_train), seed=RANDOM_SEED)
                .map(augment, num_parallel_calls=AUTOTUNE)
                .batch(batch_size).prefetch(AUTOTUNE))

    val_ds = (tf.data.Dataset.from_tensor_slices((X_val, M_val))
              .batch(batch_size).prefetch(AUTOTUNE))

    return train_ds, val_ds, X_test, M_test, X_train, X_val, M_train, M_val


def parse_args():
    p = argparse.ArgumentParser(description="Train U-Net Brain Tumor Segmentation")
    p.add_argument("--data_root",   default=RAW_DIR)
    p.add_argument("--epochs",      type=int, default=SEG_EPOCHS)
    p.add_argument("--batch_size",  type=int, default=SEG_BATCH_SIZE)
    p.add_argument("--lr",          type=float, default=SEG_LR)
    p.add_argument("--model_path",  default=UNET_PATH)
    return p.parse_args()


def run_training(args):
    os.makedirs(MODELS_DIR, exist_ok=True)
    os.makedirs(PLOTS_DIR, exist_ok=True)
    os.makedirs(PREDICTIONS_DIR, exist_ok=True)

    print("\n[1/4] Loading segmentation dataset ...")
    X, M = load_segmentation_dataset(args.data_root, SEG_IMG_SIZE)

    print("\n[2/4] Building tf.data pipelines ...")
    train_ds, val_ds, X_test, M_test, _, _, _, _ = build_seg_tf_datasets(X, M, args.batch_size)

    print("\n[3/4] Building and training U-Net ...")
    model = build_unet()
    model = compile_unet(model, lr=args.lr)
    model.summary(line_length=90)

    history = model.fit(
        train_ds,
        validation_data = val_ds,
        epochs          = args.epochs,
        callbacks       = get_seg_callbacks(args.model_path),
        verbose         = 1
    )
    plot_training_history(history, PLOTS_DIR, prefix="unet")

    print("\n[4/4] Evaluating U-Net on test set ...")
    model = tf.keras.models.load_model(
        args.model_path,
        custom_objects={
            "combined_loss": __import__("unet_model").combined_loss,
            "dice_coefficient": __import__("unet_model").dice_coefficient,
            "iou_score": __import__("unet_model").iou_score,
        }
    )
    preds, dices, ious = evaluate_segmentation(model, X_test, M_test)

    visualize_segmentation(
        images      = X_test,
        true_masks  = M_test,
        pred_masks  = preds,
        n           = 6,
        save_path   = os.path.join(PREDICTIONS_DIR, "segmentation_results.png")
    )

    print("\n✅ Segmentation training complete!")
    print(f"   Model   → {args.model_path}")
    print(f"   Results → {PREDICTIONS_DIR}")


if __name__ == "__main__":
    args = parse_args()
    run_training(args)
