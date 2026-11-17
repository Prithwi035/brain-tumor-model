"""
Central configuration for the Brain Tumor Detection project.
"""
import os

# ── Paths ────────────────────────────────────────────────────────────────────
BASE_DIR        = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR        = os.path.join(BASE_DIR, "data")
RAW_DIR         = os.path.join(DATA_DIR, "raw")
PROCESSED_DIR   = os.path.join(DATA_DIR, "processed")
MODELS_DIR      = os.path.join(BASE_DIR, "models", "saved")
RESULTS_DIR     = os.path.join(BASE_DIR, "results")
PLOTS_DIR       = os.path.join(RESULTS_DIR, "plots")
PREDICTIONS_DIR = os.path.join(RESULTS_DIR, "predictions")

# ── Dataset (Kaggle: "masoudnickparvar/brain-tumor-mri-dataset") ─────────────
# Classification labels
CLASS_NAMES = ["glioma", "meningioma", "notumor", "pituitary"]
NUM_CLASSES = len(CLASS_NAMES)

# ── Image ────────────────────────────────────────────────────────────────────
IMG_SIZE    = (224, 224)        # (H, W)
CHANNELS    = 3
BATCH_SIZE  = 32

# ── Training ─────────────────────────────────────────────────────────────────
EPOCHS          = 30
LEARNING_RATE   = 1e-4
VAL_SPLIT       = 0.15
TEST_SPLIT      = 0.15
RANDOM_SEED     = 42

# ── U-Net (segmentation) ─────────────────────────────────────────────────────
SEG_IMG_SIZE    = (256, 256)    # Segmentation works at 256×256
SEG_EPOCHS      = 25
SEG_BATCH_SIZE  = 16
SEG_LR          = 1e-4

# ── Model filenames ──────────────────────────────────────────────────────────
CLASSIFIER_PATH = os.path.join(MODELS_DIR, "classifier_efficientnet.keras")
UNET_PATH       = os.path.join(MODELS_DIR, "unet_segmentation.keras")
