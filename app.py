import io
import math
import os
import shutil
import subprocess
import time
from pathlib import Path
import modal

APP_NAME = "batch-gpu-upscale"
VOLUME_NAME = os.environ.get("UPSCALE_MODEL_VOLUME", "gpu-upscale-models")

TOPAZ_MODELS = {
    "cgi_2x": {
        "name": "CGI / Art (2x)",
        "sr_filename": "ggi-v1-fp16-192x192-2x.onnx",
        "tile_size": 192,
        "overlap": 48,
        "scale": 2,
        "input_net": "netInput:0",
        "input_noise": "t_param:0",
        "input_sharp": "t_param1:0",
        "input_comp": None,
        "output_net": "tl_unet1x2x4x/netOutput2X:0",
    },
    "cgi_4x": {
        "name": "CGI / Art (4x)",
        "sr_filename": "ggi-v1-fp16-192x192-4x.onnx",
        "tile_size": 192,
        "overlap": 48,
        "scale": 4,
        "input_net": "netInput:0",
        "input_noise": "t_param:0",
        "input_sharp": "t_param1:0",
        "input_comp": None,
        "output_net": "tl_unet1x2x4x/netOutput4X:0",
    },
    "high_fidelity_2x": {
        "name": "High Fidelity V2 (2x)",
        "sr_filename": "ghqv2-v1-fp16-128x128-2x-ox.onnx",
        "tile_size": 128,
        "overlap": 32,
        "scale": 2,
        "input_net": "netInput",
        "input_noise": "pb",
        "input_sharp": "pn",
        "input_comp": "pc",
        "output_net": "netOutput",
    },
    "high_fidelity_4x": {
        "name": "High Fidelity V2 (4x)",
        "sr_filename": "ghqv2-v1-fp16-128x128-4x-ox.onnx",
        "tile_size": 128,
        "overlap": 32,
        "scale": 4,
        "input_net": "netInput",
        "input_noise": "pb",
        "input_sharp": "pn",
        "input_comp": "pc",
        "output_net": "netOutput",
    },
    "standard_4x": {
        "name": "Standard V2 (4x)",
        "sr_filename": "ggnv2-v3-fp16-128x128-4x-ox.onnx",
        "tile_size": 128,
        "overlap": 32,
        "scale": 4,
        "input_net": "netInput",
        "input_noise": "pb",
        "input_sharp": "pn",
        "input_comp": "pc",
        "output_net": "netOutput",
    },
}


app = modal.App(APP_NAME)
volume = modal.Volume.from_name(VOLUME_NAME, create_if_missing=True)

topaz_image = (
    modal.Image.from_registry("nvidia/cuda:12.4.1-cudnn-runtime-ubuntu22.04", add_python="3.11")
    .apt_install("unzip")
    .pip_install(
        "onnxruntime-gpu==1.19.2",
        "numpy",
        "pillow",
        "opencv-python-headless",
        "requests",
    )
)


@app.function(image=topaz_image, volumes={"/data": volume}, timeout=600)
def ensure_models():
    """Validate user-provided model files in the persistent volume."""
    missing = [info["sr_filename"] for info in TOPAZ_MODELS.values()
               if not (Path("/data/models") / info["sr_filename"]).is_file()]
    if len(missing) == len(TOPAZ_MODELS):
        raise FileNotFoundError("No model files in /data/models. Supply models you are authorised to use; see README.")
    return {"available": len(TOPAZ_MODELS) - len(missing), "missing": missing}


@app.cls(
    image=topaz_image,
    gpu="L4",
    memory=16384,
    volumes={"/data": volume},
    timeout=600,
)
class TopazEngine:
    @modal.enter()
    def load(self):
        import onnxruntime as ort

        data_dir = Path("/data/models")
        providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]

        self.sessions = {}
        for key, info in TOPAZ_MODELS.items():
            sr_path = str(data_dir / info["sr_filename"])
            if Path(sr_path).exists():
                self.sessions[key] = {
                    "sr": ort.InferenceSession(sr_path, providers=providers),
                    "info": info,
                }
        gpu = subprocess.run(["nvidia-smi", "--query-gpu=name,memory.total", "--format=csv,noheader"], capture_output=True, text=True, check=True)
        print("GPU hardware: " + gpu.stdout.strip())
        for key, entry in self.sessions.items():
            print(f"Model ready: {key}; providers: {entry['sr'].get_providers()}")

    @modal.method()
    def upscale(
        self,
        image_bytes: bytes,
        model_name: str = "cgi",
        scale: int = 4,
        denoise: float = 0.05,
        sharpen: float = 0.45,
        compression: float = 0.0,
        bit_depth: int = 16,
    ) -> bytes:
        import io
        import cv2
        import numpy as np
        from PIL import Image

        model_key = f"{model_name}_{scale}x"
        if model_key not in self.sessions:
            raise ValueError(f"Model {model_key} unavailable; installed: {sorted(self.sessions)}")

        model_data = self.sessions[model_key]
        info = model_data["info"]
        sess_sr = model_data["sr"]
        actual_scale = info["scale"]
        tile_size = info["tile_size"]
        overlap = info["overlap"]

        img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        orig_w, orig_h = img.size
        target_w, target_h = orig_w * actual_scale, orig_h * actual_scale
        print(f"Processing image {orig_w}x{orig_h} -> {actual_scale}x ({target_w}x{target_h}) [{bit_depth}-bit RGB] with Topaz {info['name']} (denoise={denoise:.2f}, sharpen={sharpen:.2f})")

        img_np = np.array(img, dtype=np.float32) / 255.0

        stride = tile_size - overlap
        out_tile_size = tile_size * actual_scale

        # Trapezoidal linear weight for weighted tile blending
        ramp_len = (overlap * actual_scale) // 2
        ramp = np.ones(out_tile_size, dtype=np.float32)
        if ramp_len > 0:
            ramp[:ramp_len] = np.linspace(0.01, 1.0, ramp_len)
            ramp[-ramp_len:] = np.linspace(1.0, 0.01, ramp_len)
        weight_tile = np.outer(ramp, ramp)[..., np.newaxis]

        pad_h = (math.ceil(max(orig_h - tile_size, 0) / stride) * stride + tile_size) - orig_h
        pad_w = (math.ceil(max(orig_w - tile_size, 0) / stride) * stride + tile_size) - orig_w

        padded_img = np.pad(img_np, ((0, pad_h), (0, pad_w), (0, 0)), mode="reflect")
        pad_H, pad_W, _ = padded_img.shape

        out_H, out_W = pad_H * actual_scale, pad_W * actual_scale
        output_canvas = np.zeros((out_H, out_W, 3), dtype=np.float32)
        weight_canvas = np.zeros((out_H, out_W, 1), dtype=np.float32)

        p_denoise = np.array([denoise], dtype=np.float16)
        p_sharpen = np.array([sharpen], dtype=np.float16)
        p_comp = np.array([compression], dtype=np.float16)

        feed_dict_base = {}
        feed_dict_base[info["input_noise"]] = p_denoise
        feed_dict_base[info["input_sharp"]] = p_sharpen
        if info["input_comp"]:
            feed_dict_base[info["input_comp"]] = p_comp

        t0 = time.time()
        tile_count = 0
        for y in range(0, pad_H - tile_size + 1, stride):
            for x in range(0, pad_W - tile_size + 1, stride):
                tile = padded_img[y : y + tile_size, x : x + tile_size]
                tile_in = tile.astype(np.float16)[np.newaxis, ...]

                feed_dict = feed_dict_base.copy()
                feed_dict[info["input_net"]] = tile_in

                sr_out = sess_sr.run([info["output_net"]], feed_dict)[0]
                tile_corrected = np.clip(sr_out[0].astype(np.float32), 0.0, 1.0)

                out_y, out_x = y * actual_scale, x * actual_scale
                output_canvas[out_y : out_y + out_tile_size, out_x : out_x + out_tile_size] += tile_corrected * weight_tile
                weight_canvas[out_y : out_y + out_tile_size, out_x : out_x + out_tile_size] += weight_tile
                tile_count += 1

        final_img = output_canvas / np.maximum(weight_canvas, 1e-6)
        final_img = final_img[: orig_h * actual_scale, : orig_w * actual_scale]
        final_img = np.clip(final_img, 0.0, 1.0)

        elapsed = time.time() - t0
        print(f"Topaz {info['name']} ({actual_scale}x, {bit_depth}-bit) processed {tile_count} tiles on Nvidia L4 in {elapsed:.2f}s!")

        if bit_depth == 16:
            final_uint16 = (final_img * 65535.0).astype(np.uint16)
            bgr_16 = cv2.cvtColor(final_uint16, cv2.COLOR_RGB2BGR)
            success, enc = cv2.imencode(".png", bgr_16)
            if not success:
                raise RuntimeError("PNG encoding failed")
            return enc.tobytes()
        else:
            final_uint8 = np.round(final_img * 255.0).astype(np.uint8)
            bgr_8 = cv2.cvtColor(final_uint8, cv2.COLOR_RGB2BGR)
            success, enc = cv2.imencode(".png", bgr_8)
            if not success:
                raise RuntimeError("PNG encoding failed")
            return enc.tobytes()
