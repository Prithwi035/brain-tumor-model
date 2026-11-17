# Trained Model Checkpoints

This directory holds the trained neural network weights for inference:

1. **`classifier_efficientnet.keras`** (~80 MB)
   - Multi-class brain tumor classifier (EfficientNetB3 backbone).
   - Classes: Glioma, Meningioma, Pituitary, Normal.
   
2. **`unet_segmentation.keras`** (~355 MB)
   - 4-depth U-Net for lesion boundary delineation.
   - Note: Excluded from Git tracking due to GitHub's 100 MB file size limit.

### Download Instructions:
Place the trained `.keras` model files into this folder (`models/saved/`) before running `app/app.py` or `predict.py`.
