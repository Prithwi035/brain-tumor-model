"""
Training pipeline for the Brain Tumor Classifier (EfficientNetB3).

Memory-efficient: images are streamed from disk per batch — the full dataset
is never loaded into RAM at once.

Usage:
    python train_classifier.py --data_root "../data/raw"
    python train_classifier.py --data_root "../data/raw" --epochs 25 --fine_tune
    python train_classifier.py --data_root "../data/raw" --no_skull_strip   (faster)
"""
import os, sys, argparse
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
sys.path.insert(0, os.path.dirname(__file__))

import numpy as np
import tensorflow as tf
from config import *
from preprocessing import (
    collect_image_paths, split_paths,
    build_tf_datasets_from_paths, preprocess_image
)
from classifier_model import (
    build_efficientnet_classifier, compile_classifier, get_callbacks
)
from evaluation import (
    evaluate_classifier, plot_confusion_matrix,
    plot_roc_curves, plot_training_history
)


def parse_args():
    p = argparse.ArgumentParser(description="Train Brain Tumor Classifier")
    p.add_argument("--data_root",      default=RAW_DIR)
    p.add_argument("--epochs",         type=int,   default=EPOCHS)
    p.add_argument("--batch_size",     type=int,   default=BATCH_SIZE)
    p.add_argument("--lr",             type=float, default=LEARNING_RATE)
    p.add_argument("--no_skull_strip", action="store_true",
                   help="Skip skull-stripping preprocessing (faster, less RAM)")
    p.add_argument("--model_path",     default=CLASSIFIER_PATH)
    p.add_argument("--fine_tune",      action="store_true",
                   help="Run phase 2 fine-tuning after phase 1")
    return p.parse_args()


def run_training(args):
    os.makedirs(MODELS_DIR, exist_ok=True)
    os.makedirs(PLOTS_DIR,  exist_ok=True)
    skull_strip_flag = not args.no_skull_strip

    # ── 1. Collect file paths (RAM: only strings, no images) ─────────────────
    print("\n[1/4] Scanning dataset (paths only — no images loaded yet) ...")
    paths, labels = collect_image_paths(args.data_root)
    p_train, p_val, p_test, l_train, l_val, l_test = split_paths(paths, labels)

    # ── 2. Build streaming tf.data datasets ──────────────────────────────────
    print("\n[2/4] Building streaming tf.data pipelines ...")
    if skull_strip_flag:
        print("      Skull-stripping: ON  (images processed per-batch, not all at once)")
    else:
        print("      Skull-stripping: OFF  (--no_skull_strip flag)")

    train_ds, val_ds, test_ds = build_tf_datasets_from_paths(
        p_train, p_val, p_test,
        l_train, l_val, l_test,
        img_size          = IMG_SIZE,
        apply_skull_strip = skull_strip_flag,
        num_classes       = NUM_CLASSES,
        batch_size        = args.batch_size,
    )

    # ── 3. Build model ────────────────────────────────────────────────────────
    print("\n[3/4] Building EfficientNetB3 classifier ...")
    model = build_efficientnet_classifier(freeze_base=True)
    model = compile_classifier(model, lr=args.lr)
    model.summary(line_length=90)

    # ── Phase 1: train head only ──────────────────────────────────────────────
    phase1_epochs = max(5, args.epochs // 2)
    print(f"\n      Phase 1 – Training head only ({phase1_epochs} epochs) ...")
    h1 = model.fit(
        train_ds,
        validation_data = val_ds,
        epochs          = phase1_epochs,
        callbacks       = get_callbacks(args.model_path, patience=7),
        verbose         = 1
    )
    plot_training_history(h1, PLOTS_DIR, prefix="phase1")

    # ── Phase 2: fine-tune last 30 backbone layers (optional) ────────────────
    if args.fine_tune:
        phase2_epochs = args.epochs - phase1_epochs
        print(f"\n      Phase 2 – Fine-tuning last 30 layers ({phase2_epochs} epochs) ...")
        try:
            base = model.get_layer("efficientnetb3")
            for layer in base.layers[-30:]:
                if not isinstance(layer, tf.keras.layers.BatchNormalization):
                    layer.trainable = True
        except ValueError:
            for layer in model.layers[-35:]:
                if not isinstance(layer, tf.keras.layers.BatchNormalization):
                    layer.trainable = True
        model = compile_classifier(model, lr=args.lr / 10)
        ft_path = args.model_path.replace(".keras", "_finetuned.keras")
        h2 = model.fit(
            train_ds,
            validation_data = val_ds,
            epochs          = phase2_epochs,
            callbacks       = get_callbacks(ft_path, patience=7),
            verbose         = 1
        )
        plot_training_history(h2, PLOTS_DIR, prefix="phase2_finetune")
        best_model_path = ft_path
    else:
        best_model_path = args.model_path

    # ── 4. Evaluate ───────────────────────────────────────────────────────────
    print("\n[4/4] Evaluating on test set ...")
    model = tf.keras.models.load_model(best_model_path, compile=False)
    model = compile_classifier(model)

    # Build a small in-memory test set for evaluation plots
    # (test set only — typically 15% ≈ ~1000 images, manageable)
    print("      Loading test images for evaluation plots ...")
    X_test_np, y_test_list = [], []
    load_fn = _make_simple_loader(IMG_SIZE, skull_strip_flag)
    for path, lbl in zip(p_test, l_test):
        img = load_fn(path)
        X_test_np.append(img)
        y_test_list.append(lbl)
    X_test_np = np.array(X_test_np, dtype=np.float32)
    y_test_np = np.array(y_test_list, dtype=np.int32)

    y_pred_prob = model.predict(X_test_np, batch_size=args.batch_size, verbose=0)
    y_pred      = np.argmax(y_pred_prob, axis=1)

    from sklearn.metrics import classification_report
    print("\n" + "=" * 60)
    print("CLASSIFICATION REPORT")
    print("=" * 60)
    print(classification_report(y_test_np, y_pred, target_names=CLASS_NAMES))

    plot_confusion_matrix(
        y_test_np, y_pred,
        save_path=os.path.join(PLOTS_DIR, "confusion_matrix.png")
    )
    plot_roc_curves(
        y_test_np, y_pred_prob,
        save_path=os.path.join(PLOTS_DIR, "roc_curves.png")
    )

    print("\n✅  Training complete!")
    print(f"    Model  → {best_model_path}")
    print(f"    Plots  → {PLOTS_DIR}")


def _make_simple_loader(img_size, apply_skull_strip):
    """Simple synchronous loader for the evaluation step."""
    h, w = img_size
    def load(path):
        try:
            return preprocess_image(path, size=(w, h),
                                    apply_skull_strip=apply_skull_strip)
        except Exception:
            return np.zeros((h, w, 3), dtype=np.float32)
    return load


if __name__ == "__main__":
    args = parse_args()
    run_training(args)
