"""Official Topaz Gigapixel AI Auto-Parameter Estimator using Neural Networks."""

import io
import math
from pathlib import Path
import numpy as np
from PIL import Image

MODELS_DIR = Path(__file__).resolve().parent / "models"

_ggn_model = None
_gde_model = None


def get_coreml_models():
    global _ggn_model, _gde_model
    if _ggn_model is None or _gde_model is None:
        try:
            import coremltools as ct

            ggn_path = MODELS_DIR / "ggn_ap-v2-fp16-128x128-ml.mlmodelc"
            gde_path = MODELS_DIR / "gde_ap-v1-fp32-64x64.mlmodelc"

            if ggn_path.exists():
                _ggn_model = ct.models.CompiledMLModel(str(ggn_path))
            if gde_path.exists():
                _gde_model = ct.models.CompiledMLModel(str(gde_path))
        except Exception as e:
            print(f"[Autopilot Warning] Could not load CoreML models: {e}")
    return _ggn_model, _gde_model


def estimate_parameters(image_source, model_name: str = "cgi") -> dict:
    """
    Run official Topaz Auto-Parameter Neural Network on an image (Path, bytes, or PIL Image).
    Returns a dict with: {'denoise': float, 'sharpen': float, 'compression': float}
    """
    model_ggn, model_gde = get_coreml_models()

    if isinstance(image_source, (str, Path)):
        img = Image.open(str(image_source)).convert("RGB")
    elif isinstance(image_source, bytes):
        img = Image.open(io.BytesIO(image_source)).convert("RGB")
    elif isinstance(image_source, Image.Image):
        img = image_source.convert("RGB")
    else:
        raise ValueError("Invalid image source for parameter estimation.")

    w, h = img.size
    img_np = np.array(img, dtype=np.float32) / 255.0

    is_std_or_hf = model_name.lower() in ["standard", "high_fidelity"]

    if is_std_or_hf and model_ggn is not None:
        tile_size = 128
        active_model = model_ggn
        out_key = "netOutput"
    elif not is_std_or_hf and model_gde is not None:
        tile_size = 64
        active_model = model_gde
        out_key = "ExpandDims"
    else:
        # Fallback default values if models not available
        return {"denoise": 0.10, "sharpen": 0.45, "compression": 0.0}

    # Sample a 5x5 grid across the entire image
    xs = np.linspace(0, max(0, w - tile_size), num=min(5, max(1, w // tile_size)), dtype=int)
    ys = np.linspace(0, max(0, h - tile_size), num=min(5, max(1, h // tile_size)), dtype=int)

    preds = []
    for y in ys:
        for x in xs:
            patch = img_np[y : y + tile_size, x : x + tile_size, :]
            if patch.shape[0] == tile_size and patch.shape[1] == tile_size:
                patch_batch = patch[np.newaxis, ...]
                out = active_model.predict({"netInput": patch_batch})[out_key].flatten()
                preds.append(out)

    if not preds:
        return {"denoise": 0.10, "sharpen": 0.45, "compression": 0.0}

    preds = np.array(preds)
    median_vals = np.median(preds, axis=0)

    if is_std_or_hf:
        denoise = float(np.clip(median_vals[0], 0.0, 1.0))
        sharpen = float(np.clip(median_vals[1], 0.0, 1.0))
        compression = float(np.clip(median_vals[2], 0.0, 1.0)) if len(median_vals) > 2 else 0.0
    else:
        denoise = float(np.clip(median_vals[0], 0.0, 1.0))
        sharpen = float(np.clip(median_vals[1], 0.0, 1.0))
        compression = 0.0

    return {
        "denoise": round(denoise, 3),
        "sharpen": round(sharpen, 3),
        "compression": round(compression, 3),
    }


