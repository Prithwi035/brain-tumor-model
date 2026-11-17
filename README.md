# Brain Tumor Detection Console

A deep learning system for **brain tumor classification**, **U-Net lesion segmentation**, and **Grad-CAM visual attribution**, with an interactive web console and automated clinical PDF report generator.

---

## Features

- **Multi-Class Classification**: EfficientNetB3 model (95.1% accuracy) identifying **Glioma**, **Meningioma**, **Pituitary Tumor**, and **Normal** scans.
- **Lesion Segmentation**: 4-depth U-Net model trained with Dice + BCE loss for boundary delineation.
- **Explainable AI (Grad-CAM)**: Visual attention mapping (`block6f_add`) highlighting the exact tumor locus and margins.
- **Clinical PDF Report Generator**: Generates a standard 1-page A4 radiology report with embedded triad imaging and diagnostic metrics.
- **Input Quality Guard**: Validates uploaded images to detect and reject non-MRI scans.

---

## Quick Start

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Run the Web Application
```bash
streamlit run app/app.py
```
Open **http://localhost:8501** in your browser, upload an axial cranial MRI scan (`.jpg`, `.png`), and view diagnostic findings.

### 3. CLI Inference (Single Image)
```bash
python predict.py --image "data/raw/Testing/glioma/Te-gl_2.jpg"
```
Generates both `results/predictions/inference_result.png` and a clinical PDF report `clinical_report_<filename>.pdf`.

---

## Models & Performance

| Task | Architecture | Target Layer / Loss | Metric |
| :--- | :--- | :--- | :--- |
| **Classification** | EfficientNetB3 | Categorical Cross-Entropy | **95.09% Validation Accuracy** |
| **Attention (XAI)** | Grad-CAM | `block6f_add` (14×14) | Cranial masked visual attribution |
| **Segmentation** | U-Net (4-depth) | Dice Loss + BCE | **Dice Coef: 0.82+** |

---

## Repository Structure

```
brain_tumor_dev/
├── app/
│   └── app.py                  # Optimized Streamlit web console
├── src/
│   ├── config.py               # Hyperparameters and model paths
│   ├── preprocessing.py        # Image pipeline & non-MRI validation
│   ├── classifier_model.py     # EfficientNetB3 architecture
│   ├── unet_model.py           # U-Net architecture & loss metrics
│   ├── pdf_report.py           # Clinical PDF report generator (ReportLab)
│   ├── evaluation.py           # Metric evaluation utilities
│   ├── train_classifier.py     # Classifier training pipeline
│   └── train_segmentation.py   # U-Net training pipeline
├── models/saved/               # Trained .keras model weights
├── data/raw/                   # Training and testing datasets
├── results/predictions/        # Inference outputs and PDF reports
├── predict.py                  # CLI inference script
└── requirements.txt            # Project dependencies
```

---

## Disclaimer

This software is built for **academic demonstration and research purposes only**. It is not a certified medical device and must not be used as a substitute for professional clinical diagnosis.
