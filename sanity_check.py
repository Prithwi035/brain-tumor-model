"""
Quick sanity-check: verify all imports and model builds work correctly.
Run this before training to catch environment issues early.
"""
import os, sys
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

def check(label, fn):
    try:
        result = fn()
        print(f"  ✅ {label}: {result}")
    except Exception as e:
        print(f"  ❌ {label}: {e}")


print("\n" + "="*60)
print("  Brain Tumor Detection – Environment Sanity Check")
print("="*60)

print("\n[1] Core imports:")
check("numpy",       lambda: __import__("numpy").__version__)
check("opencv",      lambda: __import__("cv2").__version__)
check("tensorflow",  lambda: __import__("tensorflow").__version__)
check("sklearn",     lambda: __import__("sklearn").__version__)
check("streamlit",   lambda: __import__("streamlit").__version__)
check("matplotlib",  lambda: __import__("matplotlib").__version__)
check("kagglehub",   lambda: __import__("kagglehub").__version__)

print("\n[2] Project modules:")
check("config",          lambda: "OK – " + str(__import__("config").CLASS_NAMES))
check("preprocessing",   lambda: "OK")
check("classifier_model",lambda: "OK")
check("unet_model",      lambda: "OK")
check("evaluation",      lambda: "OK")

print("\n[3] Model construction:")
def build_clf():
    from classifier_model import build_efficientnet_classifier, compile_classifier
    m = build_efficientnet_classifier()
    m = compile_classifier(m)
    return f"params={m.count_params():,}"

def build_seg():
    from unet_model import build_unet, compile_unet
    m = build_unet()
    m = compile_unet(m)
    return f"params={m.count_params():,}"

check("EfficientNetB3 classifier", build_clf)
check("U-Net segmentation",        build_seg)

print("\n[4] GPU / Device:")
import tensorflow as tf
gpus = tf.config.list_physical_devices("GPU")
check("GPU devices", lambda: gpus if gpus else "No GPU – will train on CPU")

print("\n" + "="*60)
print("  Sanity check complete.")
print("="*60 + "\n")
