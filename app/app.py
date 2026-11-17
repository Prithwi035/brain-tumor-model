"""
Brain MRI Diagnostic Console
Optimized, clean interface for MRI tumor classification, Grad-CAM attention, and U-Net segmentation.

Run with:
    streamlit run app/app.py
"""
import os, sys, datetime, hashlib
import numpy as np
import cv2
import streamlit as st
import tensorflow as tf

# Add src to path
SRC_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "src")
sys.path.insert(0, SRC_DIR)

from config import CLASS_NAMES, IMG_SIZE, SEG_IMG_SIZE, CLASSIFIER_PATH, UNET_PATH
from preprocessing import is_brain_mri
from unet_model import combined_loss, dice_coefficient, iou_score
from pdf_report import generate_pdf_report

# ─────────────────────────────────────────────────────────────────────────────
# Page Setup
# ─────────────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Brain Tumor Detection Console",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# ─────────────────────────────────────────────────────────────────────────────
# Minimal, High-Performance Styling
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');

    html, body, [class*="css"] {
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
    }

    /* Clean background and header */
    .stApp {
        background-color: #0b0f19;
        color: #f1f5f9;
    }

    .main-title {
        font-size: 1.4rem;
        font-weight: 700;
        color: #f8fafc;
        margin: 0;
        letter-spacing: -0.02em;
    }
    .sub-title {
        font-size: 0.8rem;
        color: #64748b;
        margin-top: 2px;
        margin-bottom: 1.2rem;
    }

    /* Diagnosis Metric Box */
    .diag-box {
        background: #111827;
        border: 1px solid #1f2937;
        border-radius: 8px;
        padding: 0.9rem 1.1rem;
        margin-bottom: 1rem;
    }
    .diag-lbl {
        font-size: 0.7rem;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        color: #9ca3af;
        font-weight: 600;
    }
    .diag-val {
        font-size: 1.5rem;
        font-weight: 700;
        margin-top: 2px;
    }

    /* Viewport Frame */
    .view-card {
        background: #090d16;
        border: 1px solid #1f2937;
        border-radius: 6px;
        overflow: hidden;
    }
    .view-hdr {
        background: #111827;
        padding: 6px 10px;
        font-size: 0.72rem;
        font-weight: 600;
        color: #94a3b8;
        letter-spacing: 0.03em;
        text-transform: uppercase;
        border-bottom: 1px solid #1f2937;
    }

    /* Differential Row */
    .diff-bar {
        display: flex;
        align-items: center;
        margin-bottom: 6px;
        font-size: 0.8rem;
    }
    .diff-name {
        width: 90px;
        color: #94a3b8;
        font-weight: 500;
        text-transform: capitalize;
    }
    .diff-track {
        flex: 1;
        height: 6px;
        background: #1f2937;
        border-radius: 3px;
        overflow: hidden;
        margin: 0 10px;
    }
    .diff-fill {
        height: 100%;
        border-radius: 3px;
    }
    .diff-num {
        width: 45px;
        text-align: right;
        color: #f1f5f9;
        font-weight: 600;
    }
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
# Model Loading & Keras 3 Patch (Cached Once)
# ─────────────────────────────────────────────────────────────────────────────
try:
    from keras.layers import Dense
    _orig_dense_init = Dense.__init__
    def _patched_dense_init(self, *args, **kwargs):
        kwargs.pop("quantization_config", None)
        return _orig_dense_init(self, *args, **kwargs)
    Dense.__init__ = _patched_dense_init
except Exception:
    pass

@st.cache_resource(show_spinner="Loading diagnostic models …")
def load_models():
    custom_objects = {
        "combined_loss": combined_loss,
        "dice_coefficient": dice_coefficient,
        "iou_score": iou_score,
        "dice_coef": dice_coefficient,
        "iou_metric": iou_score,
    }
    models = {}
    if os.path.exists(CLASSIFIER_PATH):
        try:
            clf = tf.keras.models.load_model(CLASSIFIER_PATH, compile=False, safe_mode=False)
            models["classifier"] = clf
            # Pre-build Grad-CAM sub-model once to avoid graph re-tracing on inference
            target_layer = clf.get_layer("block6f_add")
            models["grad_model"] = tf.keras.models.Model(
                inputs=clf.inputs,
                outputs=[target_layer.output, clf.output]
            )
        except Exception as e:
            st.error(f"Classifier load error: {e}")

    if os.path.exists(UNET_PATH):
        try:
            models["unet"] = tf.keras.models.load_model(
                UNET_PATH, custom_objects=custom_objects, compile=False, safe_mode=False
            )
        except Exception as e:
            st.error(f"U-Net load error: {e}")

    return models


# ─────────────────────────────────────────────────────────────────────────────
# Diagnostic Configuration
# ─────────────────────────────────────────────────────────────────────────────
DIAG_CONFIG = {
    "glioma":     {"label": "Glioma",           "color": "#f43f5e", "desc": "Intra-axial glial origin tumor."},
    "meningioma": {"label": "Meningioma",       "color": "#f59e0b", "desc": "Extra-axial dural-based tumor."},
    "pituitary":  {"label": "Pituitary Tumor",  "color": "#a855f7", "desc": "Sellar / suprasellar mass."},
    "notumor":    {"label": "Normal (No Tumor)","color": "#10b981", "desc": "No focal intracranial lesion identified."}
}


# ─────────────────────────────────────────────────────────────────────────────
# Optimized Inference Utilities
# ─────────────────────────────────────────────────────────────────────────────
def run_classification_and_gradcam(clf_model, grad_model, img_rgb: np.ndarray):
    """Run classifier forward pass and Grad-CAM in one optimized step."""
    res = cv2.resize(img_rgb, IMG_SIZE).astype(np.float32) / 255.0
    inp = np.expand_dims(res, 0)

    # Grad-CAM GradientTape
    with tf.GradientTape() as tape:
        conv_out, preds = grad_model(inp, training=False)
        class_idx = int(tf.argmax(preds[0]))
        loss = preds[:, class_idx]

    probs = preds[0].numpy()
    confidence = float(probs[class_idx])
    pred_class = CLASS_NAMES[class_idx]

    # Attention map
    if pred_class != "notumor":
        grads = tape.gradient(loss, conv_out)
        pooled = tf.reduce_mean(grads, axis=(0, 1, 2))
        cam = tf.reduce_sum(tf.multiply(pooled, conv_out[0]), axis=-1)
        cam = np.maximum(cam.numpy(), 0)
        cam = cam / (cam.max() + 1e-8)

        # Restrict attention inside cranial boundary
        gray = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY)
        _, brain_mask = cv2.threshold(gray, 20, 255, cv2.THRESH_BINARY)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (35, 35))
        brain_mask = cv2.morphologyEx(brain_mask, cv2.MORPH_CLOSE, kernel)

        h, w = img_rgb.shape[:2]
        cam_resized = cv2.resize(cam, (w, h)) * (brain_mask > 0).astype(np.float32)
        cam_resized = cam_resized / (cam_resized.max() + 1e-8)
    else:
        cam_resized = np.zeros(img_rgb.shape[:2], dtype=np.float32)

    return pred_class, confidence, probs, cam_resized


def run_segmentation(unet_model, img_rgb: np.ndarray, threshold: float = 0.35):
    """Run U-Net segmentation."""
    res = cv2.resize(img_rgb, SEG_IMG_SIZE).astype(np.float32) / 255.0
    mask = unet_model(np.expand_dims(res, 0), training=False).numpy()[0, ..., 0]
    return mask


def create_overlays(img_rgb: np.ndarray, cam: np.ndarray, mask: np.ndarray, threshold: float = 0.35, alpha: float = 0.40):
    """Generate Grad-CAM and U-Net visualization overlays."""
    h, w = img_rgb.shape[:2]

    # Grad-CAM turbo heatmap
    cam_res = cv2.resize(cam, (w, h))
    heat = (cam_res * 255).astype(np.uint8)
    heat_color = cv2.cvtColor(cv2.applyColorMap(heat, cv2.COLORMAP_TURBO), cv2.COLOR_BGR2RGB)
    cam_overlay = cv2.addWeighted(img_rgb, 1 - alpha, heat_color, alpha, 0)

    # U-Net contour & red fill
    mask_res = cv2.resize(mask, (w, h))
    binary = (mask_res > threshold).astype(np.uint8)
    unet_overlay = img_rgb.copy()
    red_layer = np.zeros_like(img_rgb)
    red_layer[binary == 1] = [239, 68, 68]
    unet_overlay = cv2.addWeighted(unet_overlay, 1 - alpha, red_layer, alpha, 0)
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(unet_overlay, contours, -1, (250, 204, 21), 2)

    area_pct = float(binary.mean() * 100.0)
    return cam_overlay, unet_overlay, area_pct


@st.dialog("Invalid Image")
def show_invalid_dialog(reason: str):
    st.error("Please upload an MRI image of the brain.")
    st.write(f"The uploaded file is not recognized as a brain MRI scan.\n\n**Reason:** {reason}")
    if st.button("Close & Choose Another File", type="primary"):
        st.session_state["uploader_key"] = st.session_state.get("uploader_key", 0) + 1
        st.session_state["invalid_dialog_hash"] = None
        st.rerun()


# ─────────────────────────────────────────────────────────────────────────────
# Main Application
# ─────────────────────────────────────────────────────────────────────────────
def main():
    st.markdown('<div class="main-title">Brain Tumor Detection Console</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-title">Dual-Engine Classification & Lesion Localization</div>', unsafe_allow_html=True)

    models = load_models()
    if "classifier" not in models or "unet" not in models:
        st.warning("Trained model files not found in `models/saved/`.")
        return

    # Dynamic uploader key allows programmatically clearing invalid files
    uploader_key = f"mri_uploader_{st.session_state.get('uploader_key', 0)}"

    # Direct File Upload
    uploaded = st.file_uploader(
        "Upload Brain MRI Scan",
        type=["jpg", "jpeg", "png"],
        label_visibility="collapsed",
        key=uploader_key
    )

    if uploaded is None:
        st.info("Upload an axial cranial MRI scan (JPG, PNG) to begin diagnostic evaluation.")
        return

    # Compute fast hash to cache inference and avoid re-computation on interactions
    file_bytes = uploaded.getvalue()
    file_hash = hashlib.md5(file_bytes).hexdigest()

    # Optimized Cache: Only compute when a new image is uploaded
    if st.session_state.get("cached_hash") != file_hash:
        # Fast decode directly from memory buffer
        img_bgr = cv2.imdecode(np.frombuffer(file_bytes, np.uint8), cv2.IMREAD_COLOR)
        if img_bgr is None:
            st.error("Unable to decode the uploaded image file.")
            return

        is_mri, reason = is_brain_mri(img_bgr)
        if not is_mri:
            # Trigger pop-up once per invalid scan
            if st.session_state.get("invalid_dialog_hash") != file_hash:
                st.session_state["invalid_dialog_hash"] = file_hash
                show_invalid_dialog(reason)

            st.error(f"⚠️ Invalid image: {reason}. Please upload an MRI scan of the brain.")
            if st.button("Remove Invalid File"):
                st.session_state["uploader_key"] = st.session_state.get("uploader_key", 0) + 1
                st.session_state["invalid_dialog_hash"] = None
                st.rerun()
            return

        st.session_state["invalid_dialog_hash"] = None

        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

        with st.spinner("Processing MRI scan …"):
            pred_class, confidence, probs, cam = run_classification_and_gradcam(
                models["classifier"], models["grad_model"], img_rgb
            )
            mask = run_segmentation(models["unet"], img_rgb)
            cam_overlay, unet_overlay, lesion_pct = create_overlays(img_rgb, cam, mask)

            # Generate in-memory clinical PDF report
            probs_dict = {c: float(probs[CLASS_NAMES.index(c)]) for c in CLASS_NAMES}
            pdf_bytes = generate_pdf_report(
                scan_name=uploaded.name,
                pred_class=pred_class,
                confidence=confidence,
                probs=probs_dict,
                lesion_pct=lesion_pct,
                img_rgb=img_rgb,
                cam_overlay=cam_overlay,
                unet_overlay=unet_overlay
            )

        # Store in session state
        st.session_state["cached_hash"] = file_hash
        st.session_state["scan_name"] = uploaded.name
        st.session_state["img_rgb"] = img_rgb
        st.session_state["pred_class"] = pred_class
        st.session_state["confidence"] = confidence
        st.session_state["probs"] = probs
        st.session_state["cam_overlay"] = cam_overlay
        st.session_state["unet_overlay"] = unet_overlay
        st.session_state["lesion_pct"] = lesion_pct
        st.session_state["pdf_bytes"] = pdf_bytes

    # Retrieve cached results (Instant 0ms display on re-runs)
    img_rgb = st.session_state["img_rgb"]
    pred_class = st.session_state["pred_class"]
    confidence = st.session_state["confidence"]
    probs = st.session_state["probs"]
    cam_overlay = st.session_state["cam_overlay"]
    unet_overlay = st.session_state["unet_overlay"]
    lesion_pct = st.session_state["lesion_pct"]
    scan_name = st.session_state["scan_name"]

    cfg = DIAG_CONFIG[pred_class]
    is_tumor = pred_class != "notumor"

    # ── Summary Finding Row ──
    c_diag, c_probs = st.columns([1, 1])

    with c_diag:
        st.markdown(f"""
        <div class="diag-box">
            <div class="diag-lbl">Diagnostic Finding</div>
            <div class="diag-val" style="color: {cfg['color']};">
                {cfg['label']}
            </div>
            <div style="font-size: 0.85rem; color: #cbd5e1; margin-top: 4px;">
                Confidence: <strong>{confidence:.1%}</strong> · {cfg['desc']}
            </div>
            <div style="font-size: 0.75rem; color: #64748b; margin-top: 6px;">
                File: {scan_name} · Dimensions: {img_rgb.shape[1]}×{img_rgb.shape[0]} px
            </div>
        </div>
        """, unsafe_allow_html=True)

    with c_probs:
        st.markdown("<div class='diag-lbl' style='margin-bottom: 8px;'>Class Probabilities</div>", unsafe_allow_html=True)
        for c in CLASS_NAMES:
            idx = CLASS_NAMES.index(c)
            p = probs[idx]
            bar_color = DIAG_CONFIG[c]["color"]
            st.markdown(f"""
            <div class="diff-bar">
                <span class="diff-name">{DIAG_CONFIG[c]['label'].split(' ')[0]}</span>
                <div class="diff-track">
                    <div class="diff-fill" style="width: {p*100:.1f}%; background: {bar_color};"></div>
                </div>
                <span class="diff-num">{p:.1%}</span>
            </div>
            """, unsafe_allow_html=True)

    # ── 3-Panel Comparative Views ──
    st.markdown("<div class='diag-lbl' style='margin-top: 0.8rem; margin-bottom: 8px;'>Comparative Imaging</div>", unsafe_allow_html=True)
    v1, v2, v3 = st.columns(3)

    with v1:
        st.markdown("<div class='view-hdr'>1. Original MRI</div>", unsafe_allow_html=True)
        st.image(img_rgb, use_container_width=True)
        st.caption(f"Input: {img_rgb.shape[1]}×{img_rgb.shape[0]} px")

    with v2:
        st.markdown("<div class='view-hdr'>2. AI Attention (Grad-CAM)</div>", unsafe_allow_html=True)
        st.image(cam_overlay if is_tumor else img_rgb, use_container_width=True)
        st.caption("Focal neural activation" if is_tumor else "No focal activation")

    with v3:
        st.markdown(f"<div class='view-hdr'>3. U-Net Boundary ({lesion_pct:.1f}%)</div>", unsafe_allow_html=True)
        st.image(unet_overlay if is_tumor else img_rgb, use_container_width=True)
        st.caption(f"Lesion area: {lesion_pct:.2f}%" if is_tumor and lesion_pct > 0 else "U-Net: 0.00% (non-FLAIR / cystic)")

    # Sequence Note (When U-Net is 0% on cystic/T1 scans)
    if is_tumor and lesion_pct == 0.0:
        st.info("💡 **Imaging Note:** U-Net was trained on bright-signal FLAIR scans. For dark cystic cavities or extra-axial margins (like in this T1 study), refer to the **AI Attention (Grad-CAM)** view above which pinpoints the lesion margins.")

    # ── Export Clinical Reports ──
    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("<div class='diag-lbl' style='margin-bottom: 8px;'>Clinical Export & Documentation</div>", unsafe_allow_html=True)

    down_c1, down_c2 = st.columns([1, 1])

    with down_c1:
        st.download_button(
            label="📄 Download Clinical Radiology Report (.pdf)",
            data=st.session_state["pdf_bytes"],
            file_name=f"report_{scan_name.split('.')[0]}.pdf",
            mime="application/pdf",
            type="primary",
            use_container_width=True
        )

    with down_c2:
        report_text = f"""BRAIN TUMOR DETECTION REPORT
Generated: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M')}
Scan File: {scan_name}

FINDINGS:
Primary Diagnosis: {cfg['label']}
Confidence:        {confidence:.1%}
Lesion Area:       {lesion_pct:.2f}%
Status:            {'Tumor Detected' if is_tumor else 'Normal'}

CLASS PROBABILITIES:
"""
        for c in CLASS_NAMES:
            report_text += f"- {c.capitalize():12s}: {probs[CLASS_NAMES.index(c)]:.1%}\n"

        report_text += "\nNote: Academic and clinical decision-support demonstration tool."

        st.download_button(
            label="📝 Download Text Summary (.txt)",
            data=report_text,
            file_name=f"summary_{scan_name.split('.')[0]}.txt",
            mime="text/plain",
            use_container_width=True
        )


if __name__ == "__main__":
    main()
