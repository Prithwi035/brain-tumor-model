"""
Data download helper using kagglehub.
Downloads the Brain Tumor MRI Dataset (masoudnickparvar/brain-tumor-mri-dataset).
"""
import os, shutil
import kagglehub
from config import RAW_DIR


KAGGLE_DATASET = "masoudnickparvar/brain-tumor-mri-dataset"


def download_dataset(destination: str = RAW_DIR, force: bool = False) -> str:
    """
    Download the Kaggle dataset to `destination`.
    Returns the path to the dataset root directory.
    
    Requires Kaggle credentials to be configured:
      - ~/.kaggle/kaggle.json  OR
      - KAGGLE_USERNAME + KAGGLE_KEY environment variables
    """
    dest_marker = os.path.join(destination, ".downloaded")
    if os.path.exists(dest_marker) and not force:
        print(f"Dataset already present at {destination}")
        return destination

    print(f"Downloading dataset '{KAGGLE_DATASET}' via kagglehub ...")
    path = kagglehub.dataset_download(KAGGLE_DATASET)
    print(f"Downloaded to cache: {path}")

    # Copy to project data directory
    os.makedirs(destination, exist_ok=True)
    for item in os.listdir(path):
        src  = os.path.join(path, item)
        dest = os.path.join(destination, item)
        if os.path.isdir(src):
            shutil.copytree(src, dest, dirs_exist_ok=True)
        else:
            shutil.copy2(src, dest)

    # Create marker
    with open(dest_marker, "w") as f:
        f.write(KAGGLE_DATASET)

    print(f"Dataset ready at: {destination}")
    return destination


if __name__ == "__main__":
    download_dataset()
