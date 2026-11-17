"""
Dataset Setup Helper
Guides you through downloading both datasets needed for this project.

Usage:
    python src/setup_dataset.py
"""
import os, sys, shutil, webbrowser
sys.path.insert(0, os.path.dirname(__file__))
from config import RAW_DIR

KAGGLE_JSON_PATH = os.path.join(os.path.expanduser("~"), ".kaggle", "kaggle.json")

DATASETS = {
    "classifier": {
        "id":   "masoudnickparvar/brain-tumor-mri-dataset",
        "url":  "https://www.kaggle.com/datasets/masoudnickparvar/brain-tumor-mri-dataset",
        "desc": "Brain Tumor MRI Dataset (Classification: Glioma/Meningioma/Pituitary/Normal)",
        "check": lambda root: os.path.isdir(os.path.join(root, "Training")),
    },
    "segmentation": {
        "id":   "mateuszbuda/lgg-mri-segmentation",
        "url":  "https://www.kaggle.com/datasets/mateuszbuda/lgg-mri-segmentation",
        "desc": "LGG MRI Segmentation Dataset (paired .tif images + masks)",
        "check": lambda root: len([
            f for f in os.listdir(root)
            if os.path.isdir(os.path.join(root, f)) and f.startswith("TCGA")
        ]) > 0 if os.path.isdir(root) else False,
    },
}

BANNER = """
╔══════════════════════════════════════════════════════════╗
║       Brain Tumor Detection – Dataset Setup Helper       ║
╚══════════════════════════════════════════════════════════╝
"""

def hr(): print("─" * 60)

def check_kaggle_credentials() -> bool:
    if os.path.exists(KAGGLE_JSON_PATH):
        print(f"  ✅ Kaggle credentials found at {KAGGLE_JSON_PATH}")
        return True
    env_ok = os.environ.get("KAGGLE_USERNAME") and os.environ.get("KAGGLE_KEY")
    if env_ok:
        print("  ✅ Kaggle credentials found in environment variables")
        return True
    return False


def setup_kaggle_credentials():
    hr()
    print("\n  ❌ Kaggle credentials NOT found.\n")
    print("  To download datasets automatically you need a Kaggle API key.\n")
    print("  Steps:")
    print("  1. Go to: https://www.kaggle.com/settings  →  API  →  'Create New Token'")
    print("     This downloads a file called kaggle.json")
    print(f"  2. Copy kaggle.json to:  {KAGGLE_JSON_PATH}")
    print("     (create the .kaggle folder if it doesn't exist)")
    print("\n  Press ENTER to open the Kaggle settings page in your browser,")
    print("  or type 'skip' and download manually instead.")
    choice = input("  > ").strip().lower()
    if choice != "skip":
        webbrowser.open("https://www.kaggle.com/settings")
        print(f"\n  After placing kaggle.json at:\n    {KAGGLE_JSON_PATH}")
        print("  Run this script again.\n")
        sys.exit(0)
    return False


def try_kagglehub_download(dataset_id: str, dest: str) -> bool:
    """Download via kagglehub and copy to dest. Returns True on success."""
    try:
        import kagglehub
        print(f"\n  ⬇  Downloading '{dataset_id}' via kagglehub ...")
        path = kagglehub.dataset_download(dataset_id)
        print(f"  Cached at: {path}")
        os.makedirs(dest, exist_ok=True)
        for item in os.listdir(path):
            src  = os.path.join(path, item)
            dst  = os.path.join(dest, item)
            if os.path.isdir(src):
                shutil.copytree(src, dst, dirs_exist_ok=True)
            else:
                shutil.copy2(src, dst)
        print(f"  ✅ Copied to: {dest}")
        return True
    except Exception as e:
        print(f"  ❌ kagglehub download failed: {e}")
        return False


def manual_download_instructions(dataset: dict, dest: str):
    hr()
    print(f"\n  📥 MANUAL DOWNLOAD – {dataset['desc']}\n")
    print(f"  1. Open: {dataset['url']}")
    print(f"  2. Click 'Download' (zip file)")
    print(f"  3. Extract the contents so the structure looks like:\n")
    if "masoudnick" in dataset["id"]:
        print(f"       {dest}\\Training\\glioma\\*.jpg")
        print(f"       {dest}\\Training\\meningioma\\*.jpg")
        print(f"       {dest}\\Training\\notumor\\*.jpg")
        print(f"       {dest}\\Training\\pituitary\\*.jpg")
        print(f"       {dest}\\Testing\\...  (same sub-folders)")
    else:
        print(f"       {dest}\\TCGA_<id>\\<slice>.tif")
        print(f"       {dest}\\TCGA_<id>\\<slice>_mask.tif")
    print(f"\n  After extracting, run the training scripts again.\n")
    input("  Press ENTER to continue...")


def main():
    print(BANNER)
    os.makedirs(RAW_DIR, exist_ok=True)
    has_creds = check_kaggle_credentials()

    if not has_creds:
        has_creds = setup_kaggle_credentials()

    for key, ds in DATASETS.items():
        hr()
        print(f"\n  📦 {ds['desc']}")
        already = ds["check"](RAW_DIR)
        if already:
            print(f"  ✅ Already present in {RAW_DIR} — skipping download.")
            continue

        if has_creds:
            ok = try_kagglehub_download(ds["id"], RAW_DIR)
            if not ok:
                manual_download_instructions(ds, RAW_DIR)
        else:
            manual_download_instructions(ds, RAW_DIR)

    hr()
    print("\n  📊 Verifying dataset structure ...\n")
    for key, ds in DATASETS.items():
        ok = ds["check"](RAW_DIR)
        status = "✅" if ok else "❌  NOT FOUND"
        print(f"    {status}  {ds['desc']}")

    print(f"\n  Dataset root: {RAW_DIR}")
    print("""
  ▶  Next steps:
       cd src
       python train_classifier.py --data_root ../data/raw --epochs 30 --fine_tune
       python train_segmentation.py --data_root ../data/raw --epochs 25
""")


if __name__ == "__main__":
    main()
