"""
Inference script: run a single MRI image through trained models and display results.

Usage:
    python predict.py --image path/to/mri.jpg
                      [--classifier_path models/saved/classifier_efficientnet.keras]
                      [--unet_path       models/saved/unet_segmentation.keras]
                      [--save_dir        results/predictions]
                      [--no_skull_strip]
"""
import os, sys, argparse
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

import numpy as np
import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import tensorflow as tf
from PIL import Image

from config import CLASS_NAMES, IMG_SIZE, SEG_IMG_SIZE, CLASSIFIER_PATH, UNET_PATH, PREDICTIONS_DIR
from preprocessing import preprocess_image, skull_strip, is_brain_mri
from unet_model import combined_loss, dice_coefficient, iou_score


CLASS_INFO = {
    "glioma":     "🔴",
    "meningioma": "🟠",
    "pituitary":  "🟡",
    "notumor":    "🟢",
}


def generate_gradcam(model, img_rgb: np.ndarray, layer_name: str = "block6f_add") -> np.ndarray:
    """Generate Explainable AI Grad-CAM attribution heatmap."""
    img_clf = cv2.resize(img_rgb, IMG_SIZE).astype(np.float32) / 255.0
    inp = np.expand_dims(img_clf, 0)
    preds_all = model.predict(inp, verbose=0)[0]
    class_idx = int(np.argmax(preds_all))

    target_layer = model.get_layer(layer_name)
    grad_model = tf.keras.models.Model(inputs=model.inputs, outputs=[target_layer.output, model.output])
    with tf.GradientTape() as tape:
        conv_out, preds = grad_model(inp)
        loss = preds[:, class_idx]
    grads = tape.gradient(loss, conv_out)
    pooled = tf.reduce_mean(grads, axis=(0, 1, 2))
    cam = tf.reduce_sum(tf.multiply(pooled, conv_out[0]), axis=-1)
    cam = np.maximum(cam.numpy(), 0)
    cam = cam / (cam.max() + 1e-8)

    # Mask out background
    gray = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY)
    _, brain_mask = cv2.threshold(gray, 20, 255, cv2.THRESH_BINARY)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (35, 35))
    brain_mask = cv2.morphologyEx(brain_mask, cv2.MORPH_CLOSE, kernel)

    h, w = img_rgb.shape[:2]
    cam_resized = cv2.resize(cam, (w, h)) * (brain_mask > 0).astype(np.float32)
    cam_resized = cam_resized / (cam_resized.max() + 1e-8)
    return cam_resized


def parse_args():
    p = argparse.ArgumentParser(description="Brain Tumor Inference")
    p.add_argument("--image",            required=True,              help="Path to MRI image")
    p.add_argument("--classifier_path",  default=CLASSIFIER_PATH)
    p.add_argument("--unet_path",        default=UNET_PATH)
    p.add_argument("--save_dir",         default=PREDICTIONS_DIR)
    p.add_argument("--no_skull_strip",   action="store_true")
    p.add_argument("--threshold",        type=float, default=0.5)
    return p.parse_args()


def run_inference(args):
    os.makedirs(args.save_dir, exist_ok=True)

    custom_objects = {
        "combined_loss":    combined_loss,
        "dice_coefficient": dice_coefficient,
        "iou_score":        iou_score,
        "dice_coef":        dice_coefficient,
        "iou_metric":       iou_score,
    }

    try:
        from keras.layers import Dense
        _orig_dense_init = Dense.__init__
        def _patched_dense_init(self, *args, **kwargs):
            kwargs.pop("quantization_config", None)
            return _orig_dense_init(self, *args, **kwargs)
        Dense.__init__ = _patched_dense_init
    except Exception:
        pass

    # ── Load models ───────────────────────────────────────────────────────────
    clf_model, seg_model = None, None
    if os.path.exists(args.classifier_path):
        print(f"Loading classifier from {args.classifier_path} ...")
        clf_model = tf.keras.models.load_model(args.classifier_path, compile=False, safe_mode=False)
    if os.path.exists(args.unet_path):
        print(f"Loading U-Net from {args.unet_path} ...")
        seg_model = tf.keras.models.load_model(args.unet_path, custom_objects=custom_objects, compile=False, safe_mode=False)

    # ── Preprocess ────────────────────────────────────────────────────────────
    print(f"\nProcessing: {args.image}")
    img = cv2.imread(args.image)
    if img is None:
        raise FileNotFoundError(f"Cannot read: {args.image}")

    # ── Non-MRI Validation Check ──────────────────────────────────────────────
    is_mri, invalid_reason = is_brain_mri(img)
    if not is_mri:
        print(f"\n{'!'*60}")
        print(f"  [ERROR] 🛑 Please upload an MRI image of the brain.")
        print(f"  Reason: {invalid_reason}")
        print(f"{'!'*60}\n")
        sys.exit(1)

    img_rgb_raw = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    if not args.no_skull_strip:
        img_clf_raw = cv2.cvtColor(skull_strip(img.copy()), cv2.COLOR_BGR2RGB)
    else:
        img_clf_raw = img_rgb_raw

    # For classifier
    img_clf = cv2.resize(img_clf_raw, IMG_SIZE).astype(np.float32) / 255.0
    # For segmentation (always use natural high-fidelity MRI without destructive Otsu masking)
    img_seg = cv2.resize(img_rgb_raw, SEG_IMG_SIZE).astype(np.float32) / 255.0

    # ── Classify ──────────────────────────────────────────────────────────────
    pred_class, confidence, probs = None, None, None
    if clf_model:
        probs = clf_model.predict(np.expand_dims(img_clf, 0), verbose=0)[0]
        idx   = int(np.argmax(probs))
        pred_class  = CLASS_NAMES[idx]
        confidence  = float(probs[idx])
        emoji = CLASS_INFO.get(pred_class, "")
        print(f"\n{'─'*40}")
        print(f"  Classification: {emoji} {pred_class.upper()}")
        print(f"  Confidence    : {confidence:.1%}")
        print(f"  All classes   :")
        for cls, p in zip(CLASS_NAMES, probs):
            bar = "█" * int(p * 20)
            print(f"    {cls:12s}: {bar:<20} {p:.1%}")

    # ── Segment ───────────────────────────────────────────────────────────────
    mask = None
    if seg_model:
        mask_raw = seg_model.predict(np.expand_dims(img_seg, 0), verbose=0)[0, ..., 0]
        mask = mask_raw
        binary = (mask_raw > args.threshold).astype(np.uint8)
        pct = binary.mean() * 100
        print(f"\n  Tumor Area (threshold={args.threshold}): {pct:.2f}% of image")

    # ── Explainable AI (Grad-CAM) ─────────────────────────────────────────────
    cam = None
    if clf_model and pred_class != "notumor":
        print("\n  Generating Explainable AI (Grad-CAM) visual attribution ...")
        cam = generate_gradcam(clf_model, img_rgb_raw)

    # ── Visualise & save ──────────────────────────────────────────────────────
    n_panels = 1 + (1 if clf_model else 0) + (1 if cam is not None else 0) + (2 if seg_model else 0)
    fig, axes = plt.subplots(1, n_panels, figsize=(4.5 * n_panels, 4.5))
    if n_panels == 1:
        axes = [axes]

    ax_idx = 0

    # 1. Original
    axes[ax_idx].imshow(img_rgb_raw)
    axes[ax_idx].set_title("Original MRI", fontweight="bold")
    axes[ax_idx].axis("off")
    ax_idx += 1

    # 2. Class probabilities bar chart
    if clf_model and probs is not None:
        colors = ["#e74c3c", "#e67e22", "#f1c40f", "#2ecc71"]
        axes[ax_idx].barh(CLASS_NAMES, probs, color=colors)
        axes[ax_idx].set_xlim(0, 1)
        axes[ax_idx].set_title(
            f"Diagnosis: {pred_class.upper()}\nConfidence: {confidence:.1%}",
            fontweight="bold"
        )
        axes[ax_idx].grid(axis="x", alpha=0.3)
        ax_idx += 1

    # 3. Grad-CAM visual attribution
    if cam is not None:
        h, w = img_rgb_raw.shape[:2]
        cam_res = cv2.resize(cam, (w, h))
        heatmap_c = cv2.applyColorMap((cam_res * 255).astype(np.uint8), cv2.COLORMAP_JET)
        heatmap_c = cv2.cvtColor(heatmap_c, cv2.COLOR_BGR2RGB)
        overlaid_cam = cv2.addWeighted(img_rgb_raw, 0.65, heatmap_c, 0.35, 0)
        axes[ax_idx].imshow(overlaid_cam)
        axes[ax_idx].set_title(f"Grad-CAM (Explainable AI)\nTumor Locus Attribution", fontweight="bold")
        axes[ax_idx].axis("off")
        ax_idx += 1

    # 4 & 5. U-Net Mask overlay + heatmap
    if seg_model and mask is not None:
        # Overlay
        img8 = (img_clf * 255).astype(np.uint8)
        mask_resized = cv2.resize(mask, IMG_SIZE)
        binary_r = (mask_resized > args.threshold).astype(np.uint8)
        overlay = np.zeros_like(img8)
        overlay[binary_r == 1] = [220, 50, 50]
        overlaid = cv2.addWeighted(img8, 0.6, overlay, 0.4, 0)
        contours, _ = cv2.findContours(binary_r, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(overlaid, contours, -1, (255, 255, 0), 2)
        axes[ax_idx].imshow(overlaid)
        axes[ax_idx].set_title(f"U-Net Mask Overlay\n(Area: {pct:.1f}%)", fontweight="bold")
        axes[ax_idx].axis("off")
        ax_idx += 1

        # Heatmap
        im = axes[ax_idx].imshow(mask, cmap="hot", vmin=0, vmax=1)
        axes[ax_idx].set_title(f"U-Net Prob Heatmap\n(Max: {mask.max():.1%})", fontweight="bold")
        axes[ax_idx].axis("off")
        fig.colorbar(im, ax=axes[ax_idx], fraction=0.046)
        ax_idx += 1

    plt.suptitle("Brain Tumor Detection – Inference Results",
                 fontsize=14, fontweight="bold", y=1.01)
    plt.tight_layout()

    out_path = os.path.join(args.save_dir, "inference_result.png")
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"\n  Result saved → {out_path}")

    # Generate Clinical PDF Report
    try:
        from pdf_report import generate_pdf_report
        probs_dict = {c: float(probs[CLASS_NAMES.index(c)]) for c in CLASS_NAMES} if probs is not None else {}
        h, w = img_rgb_raw.shape[:2]
        cam_res = cv2.resize(cam, (w, h)) if cam is not None else np.zeros((h, w), dtype=np.float32)
        heat_c = cv2.applyColorMap((cam_res * 255).astype(np.uint8), cv2.COLORMAP_TURBO)
        heat_c = cv2.cvtColor(heat_c, cv2.COLOR_BGR2RGB)
        cam_overlay_img = cv2.addWeighted(img_rgb_raw, 0.60, heat_c, 0.40, 0)

        mask_res = cv2.resize(mask, (w, h)) if mask is not None else np.zeros((h, w), dtype=np.float32)
        binary_mask = (mask_res > args.threshold).astype(np.uint8)
        unet_overlay_img = img_rgb_raw.copy()
        red_l = np.zeros_like(img_rgb_raw)
        red_l[binary_mask == 1] = [239, 68, 68]
        unet_overlay_img = cv2.addWeighted(unet_overlay_img, 0.60, red_l, 0.40, 0)
        contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(unet_overlay_img, contours, -1, (250, 204, 21), 2)

        pdf_path = os.path.join(args.save_dir, f"clinical_report_{os.path.basename(args.image).split('.')[0]}.pdf")
        generate_pdf_report(
            scan_name=os.path.basename(args.image),
            pred_class=pred_class,
            confidence=confidence,
            probs=probs_dict,
            lesion_pct=pct if mask is not None else 0.0,
            img_rgb=img_rgb_raw,
            cam_overlay=cam_overlay_img,
            unet_overlay=unet_overlay_img,
            output_path=pdf_path
        )
        print(f"  Clinical PDF Report → {pdf_path}")
    except Exception as e:
        print(f"  Note on PDF generation: {e}")


if __name__ == "__main__":
    args = parse_args()
    run_inference(args)
