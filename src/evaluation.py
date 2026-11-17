"""
Evaluation utilities:
  - Classification report, confusion matrix, ROC curves
  - Segmentation metrics: Dice, IoU per sample
  - Visualization helpers
"""
import os, numpy as np, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import cv2
from sklearn.metrics import (
    classification_report, confusion_matrix,
    roc_curve, auc
)
from sklearn.preprocessing import label_binarize
import tensorflow as tf
from config import CLASS_NAMES, NUM_CLASSES, PLOTS_DIR


# ─────────────────────────────────────────────────────────────────────────────
# Classification evaluation
# ─────────────────────────────────────────────────────────────────────────────

def evaluate_classifier(model, test_ds, y_test_onehot):
    """Return predictions and print full classification report."""
    y_pred_prob = model.predict(test_ds, verbose=0)
    y_pred      = np.argmax(y_pred_prob, axis=1)
    y_true      = np.argmax(y_test_onehot, axis=1) if y_test_onehot.ndim > 1 else y_test_onehot

    print("\n" + "=" * 60)
    print("CLASSIFICATION REPORT")
    print("=" * 60)
    print(classification_report(y_true, y_pred, target_names=CLASS_NAMES))
    return y_true, y_pred, y_pred_prob


def plot_confusion_matrix(y_true, y_pred, save_path: str = None):
    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(8, 7))
    im = ax.imshow(cm, interpolation="nearest", cmap="Blues")
    plt.colorbar(im, ax=ax)
    ax.set(xticks=range(NUM_CLASSES), yticks=range(NUM_CLASSES),
           xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES,
           xlabel="Predicted", ylabel="True",
           title="Confusion Matrix – Brain Tumor Classifier")
    plt.xticks(rotation=30, ha="right")
    # Annotate cells
    thresh = cm.max() / 2
    for i in range(NUM_CLASSES):
        for j in range(NUM_CLASSES):
            ax.text(j, i, cm[i, j], ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black", fontsize=12)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Saved confusion matrix → {save_path}")
    plt.close()


def plot_roc_curves(y_true, y_pred_prob, save_path: str = None):
    y_bin = label_binarize(y_true, classes=list(range(NUM_CLASSES)))
    fig, ax = plt.subplots(figsize=(9, 7))
    colors = ["steelblue", "darkorange", "forestgreen", "firebrick"]
    for i, (cls, col) in enumerate(zip(CLASS_NAMES, colors)):
        fpr, tpr, _ = roc_curve(y_bin[:, i], y_pred_prob[:, i])
        roc_auc = auc(fpr, tpr)
        ax.plot(fpr, tpr, lw=2, color=col, label=f"{cls} (AUC={roc_auc:.3f})")
    ax.plot([0, 1], [0, 1], "k--", lw=1)
    ax.set(xlim=[0, 1], ylim=[0, 1.02],
           xlabel="False Positive Rate", ylabel="True Positive Rate",
           title="ROC Curves – Brain Tumor Classifier")
    ax.legend(loc="lower right")
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Saved ROC curves → {save_path}")
    plt.close()


def plot_training_history(history, save_dir: str = PLOTS_DIR, prefix: str = ""):
    metrics = [("accuracy", "Accuracy"), ("loss", "Loss"), ("auc", "AUC")]
    for metric, title in metrics:
        if metric not in history.history:
            continue
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.plot(history.history[metric], label=f"Train {title}")
        val_key = f"val_{metric}"
        if val_key in history.history:
            ax.plot(history.history[val_key], label=f"Val {title}")
        ax.set(xlabel="Epoch", ylabel=title,
               title=f"{prefix} {title} over Epochs")
        ax.legend()
        ax.grid(alpha=0.3)
        plt.tight_layout()
        fname = os.path.join(save_dir, f"{prefix}_{metric}.png".lstrip("_"))
        plt.savefig(fname, dpi=150, bbox_inches="tight")
        plt.close()
    print(f"Training history plots saved to {save_dir}")


# ─────────────────────────────────────────────────────────────────────────────
# Segmentation evaluation
# ─────────────────────────────────────────────────────────────────────────────

def dice_numpy(y_true: np.ndarray, y_pred: np.ndarray,
               threshold: float = 0.5, smooth: float = 1e-6) -> float:
    pred_bin = (y_pred > threshold).astype(np.float32)
    intersection = (y_true * pred_bin).sum()
    return (2 * intersection + smooth) / (y_true.sum() + pred_bin.sum() + smooth)


def iou_numpy(y_true: np.ndarray, y_pred: np.ndarray,
              threshold: float = 0.5, smooth: float = 1e-6) -> float:
    pred_bin = (y_pred > threshold).astype(np.float32)
    intersection = (y_true * pred_bin).sum()
    union = y_true.sum() + pred_bin.sum() - intersection
    return (intersection + smooth) / (union + smooth)


def evaluate_segmentation(model, X_test, masks_test, batch_size: int = 8):
    preds = model.predict(X_test, batch_size=batch_size, verbose=0)
    dice_scores, iou_scores = [], []
    for gt, pr in zip(masks_test, preds):
        dice_scores.append(dice_numpy(gt, pr))
        iou_scores.append(iou_numpy(gt, pr))
    print(f"\nSegmentation Results:")
    print(f"  Mean Dice Score : {np.mean(dice_scores):.4f} ± {np.std(dice_scores):.4f}")
    print(f"  Mean IoU Score  : {np.mean(iou_scores):.4f} ± {np.std(iou_scores):.4f}")
    return preds, dice_scores, iou_scores


# ─────────────────────────────────────────────────────────────────────────────
# Visualization
# ─────────────────────────────────────────────────────────────────────────────

def visualize_segmentation(images, true_masks, pred_masks,
                            n: int = 6, save_path: str = None):
    """Display n samples: original | true mask overlay | predicted mask overlay."""
    n = min(n, len(images))
    fig, axes = plt.subplots(n, 3, figsize=(12, 4 * n))
    titles = ["Original MRI", "Ground Truth Mask", "Predicted Mask"]
    for ax, t in zip(axes[0], titles):
        ax.set_title(t, fontsize=13, fontweight="bold")

    for i in range(n):
        img  = (images[i] * 255).astype(np.uint8) if images[i].max() <= 1 else images[i]
        t_m  = (true_masks[i, ..., 0] > 0.5).astype(np.uint8)
        p_m  = (pred_masks[i, ..., 0] > 0.5).astype(np.uint8)

        def overlay(base_img, mask, color=(255, 50, 50)):
            out = base_img.copy()
            m3  = np.stack([mask * color[0],
                            mask * color[1],
                            mask * color[2]], axis=-1).astype(np.uint8)
            return cv2.addWeighted(out, 0.7, m3, 0.3, 0)

        axes[i, 0].imshow(img)
        axes[i, 0].axis("off")
        axes[i, 1].imshow(overlay(img, t_m, (50, 205, 50)))
        axes[i, 1].axis("off")
        axes[i, 2].imshow(overlay(img, p_m, (220, 50, 50)))
        axes[i, 2].axis("off")

    plt.suptitle("Brain Tumor Segmentation – Ground Truth vs Prediction",
                 fontsize=14, fontweight="bold", y=1.01)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Saved segmentation visualization → {save_path}")
    plt.close()


def visualize_misclassified(X_test, y_true, y_pred, n: int = 8, save_path: str = None):
    wrong = np.where(y_true != y_pred)[0][:n]
    if len(wrong) == 0:
        print("No misclassifications found!")
        return
    fig, axes = plt.subplots(2, min(4, len(wrong)), figsize=(14, 6))
    axes = axes.flatten() if len(wrong) > 1 else [axes]
    for ax, idx in zip(axes, wrong):
        img = (X_test[idx] * 255).astype(np.uint8)
        ax.imshow(img)
        ax.set_title(f"True: {CLASS_NAMES[y_true[idx]]}\nPred: {CLASS_NAMES[y_pred[idx]]}",
                     fontsize=9, color="red")
        ax.axis("off")
    for ax in axes[len(wrong):]:
        ax.axis("off")
    plt.suptitle("Misclassified MRI Samples", fontsize=13, fontweight="bold")
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
