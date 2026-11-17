"""
MRI image preprocessing utilities.
- Skull-stripping approximation via Otsu + morphology
- Normalisation, resizing
- Memory-efficient tf.data pipeline (reads images from disk on-the-fly)
"""
import os, cv2, numpy as np
from glob import glob
from sklearn.model_selection import train_test_split
import tensorflow as tf
from config import *


# ─────────────────────────────────────────────────────────────────────────────
# Low-level helpers
# ─────────────────────────────────────────────────────────────────────────────

def skull_strip(img_bgr: np.ndarray) -> np.ndarray:
    """
    Approximate skull-stripping:
    1. Convert to grayscale → Otsu threshold
    2. Morphological closing to fill holes
    3. Keep largest connected component (the brain)
    4. Apply mask to original image
    """
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
    closed = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel, iterations=3)

    n_labels, labels, stats, _ = cv2.connectedComponentsWithStats(closed)
    if n_labels > 1:
        largest = 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])
        mask = (labels == largest).astype(np.uint8) * 255
    else:
        mask = closed

    return cv2.bitwise_and(img_bgr, img_bgr, mask=mask)


def is_brain_mri(img_bgr: np.ndarray) -> tuple:
    """
    Validate whether an uploaded image has the structural characteristics of a cranial Brain MRI scan.
    Rejects non-MRI images (photos of people, pets, scenes, documents, blank/solid images, drawings).
    Returns:
        (is_valid: bool, reason: str)
    """
    if img_bgr is None or img_bgr.size == 0:
        return False, "File is empty or cannot be decoded as an image."

    h, w = img_bgr.shape[:2]
    if h < 64 or w < 64:
        return False, "Resolution is too low to be a diagnostic brain MRI scan."

    # 1. Color check (Medical cranial MRI scans are monochromatic / grayscale)
    if len(img_bgr.shape) == 3 and img_bgr.shape[2] == 3:
        b, g, r = img_bgr[:, :, 0], img_bgr[:, :, 1], img_bgr[:, :, 2]
        color_diff = np.mean(np.abs(r.astype(float) - g.astype(float)) + np.abs(g.astype(float) - b.astype(float)))
        if color_diff > 18.0:
            return False, "The uploaded image contains distinct color. Medical brain MRI scans are monochromatic (grayscale)."

    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY) if len(img_bgr.shape) == 3 else img_bgr

    # 2. Dynamic range / contrast
    std_dev = float(np.std(gray))
    mean_val = float(np.mean(gray))
    if std_dev < 12.0:
        return False, "Image lacks contrast or is a solid/blank color."
    if mean_val > 235.0:
        return False, "Image background is bright white (e.g. document, diagram, or sketch), not a dark-field MRI."
    if mean_val < 6.0:
        return False, "Image is completely dark/blank."

    # 3. Head / Brain anatomical structure check
    _, thresh = cv2.threshold(gray, 20, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return False, "No cranial anatomical structures detected."

    largest_c = max(contours, key=cv2.contourArea)
    area = cv2.contourArea(largest_c)
    total_area = h * w
    area_ratio = area / total_area

    if area_ratio < 0.10:
        return False, "Detected anatomical mass is too small to be a brain MRI scan."

    x, y, bw, bh = cv2.boundingRect(largest_c)
    aspect = float(bw) / max(1, bh)
    if aspect < 0.45 or aspect > 2.2:
        return False, "Anatomical aspect ratio does not match standard cranial proportions."

    cx = x + bw / 2.0
    cy = y + bh / 2.0
    if cx / w < 0.15 or cx / w > 0.85 or cy / h < 0.15 or cy / h > 0.85:
        return False, "Anatomical structure is not centered in the image frame."

    return True, "Valid Brain MRI"


def preprocess_image(path: str,
                     size: tuple = IMG_SIZE,
                     apply_skull_strip: bool = True,
                     normalize: bool = True) -> np.ndarray:
    """Read → skull-strip (optional) → resize → normalize."""
    img = cv2.imread(path)
    if img is None:
        raise FileNotFoundError(f"Cannot read: {path}")
    if apply_skull_strip:
        img = skull_strip(img)
    img = cv2.resize(img, size)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    if normalize:
        img = img.astype(np.float32) / 255.0
    return img


# ─────────────────────────────────────────────────────────────────────────────
# Path-only dataset builder  (memory-efficient)
# ─────────────────────────────────────────────────────────────────────────────

def collect_image_paths(data_root: str):
    """
    Scan Training/ and Testing/ sub-folders and return (paths, labels) lists.
    Does NOT load any images into RAM.
    """
    paths, labels = [], []
    for split in ["Training", "Testing"]:
        split_dir = os.path.join(data_root, split)
        if not os.path.isdir(split_dir):
            continue
        for label_idx, cls in enumerate(CLASS_NAMES):
            cls_dir = os.path.join(split_dir, cls)
            if not os.path.isdir(cls_dir):
                continue
            for ext in ["*.jpg", "*.jpeg", "*.png"]:
                for p in glob(os.path.join(cls_dir, ext)):
                    paths.append(p)
                    labels.append(label_idx)

    if len(paths) == 0:
        raise RuntimeError(
            "\n\n❌  No images loaded from: " + data_root +
            "\n    Expected layout:"
            "\n      data/raw/Training/glioma/*.jpg"
            "\n      data/raw/Training/meningioma/*.jpg"
            "\n      data/raw/Training/notumor/*.jpg"
            "\n      data/raw/Training/pituitary/*.jpg"
            "\n      data/raw/Testing/  (same sub-folders)"
            "\n\n    ➡  See README.md for download instructions.\n"
        )

    paths  = np.array(paths)
    labels = np.array(labels, dtype=np.int32)
    print(f"Found {len(paths)} images across {len(set(labels))} classes")
    return paths, labels


def split_paths(paths, labels):
    """Train / val / test path splits (stratified)."""
    p_tv, p_test, l_tv, l_test = train_test_split(
        paths, labels,
        test_size=TEST_SPLIT, stratify=labels, random_state=RANDOM_SEED
    )
    val_frac = VAL_SPLIT / (1 - TEST_SPLIT)
    p_train, p_val, l_train, l_val = train_test_split(
        p_tv, l_tv,
        test_size=val_frac, stratify=l_tv, random_state=RANDOM_SEED
    )
    print(f"Split → train:{len(p_train)}  val:{len(p_val)}  test:{len(p_test)}")
    return p_train, p_val, p_test, l_train, l_val, l_test


# ─────────────────────────────────────────────────────────────────────────────
# tf.data streaming pipeline  (images loaded per-batch, not all at once)
# ─────────────────────────────────────────────────────────────────────────────

def _make_load_fn(img_size: tuple, apply_skull_strip: bool, num_classes: int):
    """Return a tf.py_function wrapper that reads one image from disk."""
    h, w = img_size

    def load_and_preprocess(path_tensor, label_tensor):
        def _load(path_bytes, label_int):
            path = path_bytes.numpy().decode("utf-8")
            try:
                img = preprocess_image(
                    path, size=(w, h),
                    apply_skull_strip=apply_skull_strip
                )
            except Exception:
                img = np.zeros((h, w, 3), dtype=np.float32)
            lbl = tf.one_hot(label_int, num_classes)
            return img.astype(np.float32), lbl.numpy()

        img, lbl = tf.py_function(
            func=_load,
            inp=[path_tensor, label_tensor],
            Tout=[tf.float32, tf.float32]
        )
        img.set_shape((h, w, 3))
        lbl.set_shape((num_classes,))
        return img, lbl

    return load_and_preprocess


def _augment(image, label):
    image = tf.image.random_flip_left_right(image)
    image = tf.image.random_flip_up_down(image)
    image = tf.image.random_brightness(image, max_delta=0.15)
    image = tf.image.random_contrast(image, lower=0.8, upper=1.2)
    image = tf.clip_by_value(image, 0.0, 1.0)
    return image, label


def build_tf_datasets_from_paths(p_train, p_val, p_test,
                                  l_train, l_val, l_test,
                                  img_size: tuple = IMG_SIZE,
                                  apply_skull_strip: bool = True,
                                  num_classes: int = NUM_CLASSES,
                                  batch_size: int = BATCH_SIZE):
    """
    Memory-efficient pipeline:
    - Stores only file paths in RAM
    - Loads images from disk in parallel, batch by batch
    - Never loads the full dataset into RAM at once
    """
    AUTOTUNE = tf.data.AUTOTUNE
    load_fn  = _make_load_fn(img_size, apply_skull_strip, num_classes)

    train_ds = (
        tf.data.Dataset.from_tensor_slices((p_train, l_train))
        .shuffle(len(p_train), seed=RANDOM_SEED)
        .map(load_fn,   num_parallel_calls=AUTOTUNE)
        .map(_augment,  num_parallel_calls=AUTOTUNE)
        .batch(batch_size)
        .prefetch(AUTOTUNE)
    )

    val_ds = (
        tf.data.Dataset.from_tensor_slices((p_val, l_val))
        .map(load_fn, num_parallel_calls=AUTOTUNE)
        .batch(batch_size)
        .prefetch(AUTOTUNE)
    )

    test_ds = (
        tf.data.Dataset.from_tensor_slices((p_test, l_test))
        .map(load_fn, num_parallel_calls=AUTOTUNE)
        .batch(batch_size)
        .prefetch(AUTOTUNE)
    )

    return train_ds, val_ds, test_ds


# ─────────────────────────────────────────────────────────────────────────────
# Legacy in-memory loader  (kept for reference / small datasets)
# ─────────────────────────────────────────────────────────────────────────────

def load_classification_dataset(data_root: str,
                                 img_size: tuple = IMG_SIZE,
                                 skull_strip_flag: bool = True):
    """
    Load ALL images into RAM at once (legacy).
    Use build_tf_datasets_from_paths() instead to avoid memory errors.
    """
    X, y = [], []
    for split in ["Training", "Testing"]:
        split_dir = os.path.join(data_root, split)
        if not os.path.isdir(split_dir):
            continue
        for label_idx, cls in enumerate(CLASS_NAMES):
            cls_dir = os.path.join(split_dir, cls)
            if not os.path.isdir(cls_dir):
                continue
            for ext in ["*.jpg", "*.jpeg", "*.png"]:
                for p in glob(os.path.join(cls_dir, ext)):
                    try:
                        img = preprocess_image(p, size=img_size,
                                               apply_skull_strip=skull_strip_flag)
                        X.append(img)
                        y.append(label_idx)
                    except Exception as e:
                        print(f"[WARN] Skipping {p}: {e}")

    if len(X) == 0:
        raise RuntimeError(
            "\n\n❌  No images loaded from: " + data_root +
            "\n    ➡  See README.md for download instructions.\n"
        )

    X = np.array(X, dtype=np.float32)
    y = np.array(y, dtype=np.int32)

    X_tv, X_test, y_tv, y_test = train_test_split(
        X, y, test_size=TEST_SPLIT, stratify=y, random_state=RANDOM_SEED)
    val_frac = VAL_SPLIT / (1 - TEST_SPLIT)
    X_train, X_val, y_train, y_val = train_test_split(
        X_tv, y_tv, test_size=val_frac, stratify=y_tv, random_state=RANDOM_SEED)

    print(f"Dataset → train:{len(X_train)}  val:{len(X_val)}  test:{len(X_test)}")
    return X_train, X_val, X_test, y_train, y_val, y_test


def build_tf_datasets(X_train, X_val, X_test, y_train, y_val, y_test,
                      num_classes: int = NUM_CLASSES,
                      batch_size: int = BATCH_SIZE):
    """Legacy: build tf.data from already-loaded numpy arrays."""
    AUTOTUNE = tf.data.AUTOTUNE

    def one_hot(image, label):
        return image, tf.one_hot(label, num_classes)

    train_ds = (tf.data.Dataset.from_tensor_slices((X_train, y_train))
                .shuffle(len(X_train), seed=RANDOM_SEED)
                .map(_augment, num_parallel_calls=AUTOTUNE)
                .map(one_hot,  num_parallel_calls=AUTOTUNE)
                .batch(batch_size).prefetch(AUTOTUNE))

    val_ds = (tf.data.Dataset.from_tensor_slices((X_val, y_val))
              .map(one_hot, num_parallel_calls=AUTOTUNE)
              .batch(batch_size).prefetch(AUTOTUNE))

    test_ds = (tf.data.Dataset.from_tensor_slices((X_test, y_test))
               .map(one_hot, num_parallel_calls=AUTOTUNE)
               .batch(batch_size).prefetch(AUTOTUNE))

    return train_ds, val_ds, test_ds
