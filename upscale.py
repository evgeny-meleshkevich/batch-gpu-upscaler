#!/usr/bin/env python3
"""Batch Exact 2x / 4x Upscaler using official Topaz Gigapixel (CGI / High Fidelity / Standard) on Nvidia L4 (Modal) with 16-bit / 8-bit RGB PNG and Auto-AI parameter detection."""

import argparse
import os
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path
import modal

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
OUTPUTS_DIR = SCRIPT_DIR / "outputs"

IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".webp", ".bmp", ".tiff"}


def get_image_files(target_path: Path) -> list[Path]:
    if target_path.is_file():
        if target_path.suffix.lower() in IMAGE_EXTENSIONS:
            return [target_path]
        return []
    if target_path.is_dir():
        files = []
        for ext in IMAGE_EXTENSIONS:
            files.extend(target_path.glob(f"*{ext}"))
            files.extend(target_path.glob(f"*{ext.upper()}"))
        return sorted([f for f in files if "topaz" not in str(f.parent).lower() and not f.name.startswith(".")])
    return []


def main():
    parser = argparse.ArgumentParser(description="Official Topaz Gigapixel Upscaler on Modal (Nvidia L4)")
    parser.add_argument("path", nargs="?", help="Path to image file or folder containing images")
    parser.add_argument("--model", "-m", default="cgi", choices=["cgi", "high_fidelity", "standard"], help="Topaz model (default: cgi)")
    parser.add_argument("--scale", "-s", type=int, default=4, choices=[2, 4], help="Scale factor (2x or 4x, default: 4)")
    parser.add_argument("--bit-depth", "-b", type=int, default=16, choices=[8, 16], help="PNG Color Bit Depth (default: 16-bit Master)")
    parser.add_argument("--auto", "-a", action="store_true", default=False, help="Use optional local parameter models if available")
    parser.add_argument("--denoise", "-d", type=float, default=None, help="Manual Remove Noise parameter (0.0 - 1.0)")
    parser.add_argument("--sharpen", "-sh", type=float, default=None, help="Manual Sharpen parameter (0.0 - 1.0)")
    parser.add_argument("--compression", "-c", type=float, default=None, help="Manual Remove Compression parameter (0.0 - 1.0)")
    parser.add_argument("--inspect", action="store_true", help="List inputs and configuration without cloud calls")
    args = parser.parse_args()

    if not args.path:
        print("Usage: upscale.py <path/to/image.png or path/to/folder/> [options]")
        sys.exit(1)

    target_path = Path(args.path).resolve()
    if not target_path.exists():
        print(f"Error: path not found: {target_path}")
        sys.exit(1)

    image_files = get_image_files(target_path)
    if not image_files:
        print(f"No image files found in: {target_path}")
        sys.exit(0)

    model_label = {
        "cgi": f"CGI / Art ({args.scale}x)",
        "high_fidelity": f"High Fidelity V2 ({args.scale}x)",
        "standard": f"Standard V2 ({args.scale}x)",
    }.get(args.model, args.model)

    use_auto = args.auto and (args.denoise is None and args.sharpen is None)

    print(f"Found {len(image_files)} image(s) to process.")
    print(f"Model: Topaz {model_label}")
    print(f"Scale: {args.scale}x Exact Pixel Magnification")
    print(f"Color Depth: {args.bit_depth}-bit RGB ({65536 if args.bit_depth == 16 else 256} levels/channel)")
    if use_auto:
        print(f"Parameters: [Topaz Auto-AI Neural Estimator Active per image]")
    else:
        print(f"Parameters (Manual): Denoise={args.denoise or 0.05}, Sharpen={args.sharpen or 0.45}, Compression={args.compression or 0.0}")
    print(f"GPU configuration: Nvidia L4 (24GB VRAM) on Modal")

    if args.model == "standard" and args.scale != 4:
        parser.error("The configured Standard model supports only 4x")
    if args.inspect:
        print("[INSPECT ONLY] No images uploaded; no GPU computation requested.")
        for idx, path in enumerate(image_files, 1):
            print(f"  [{idx}/{len(image_files)}] {path.name}")
        return

    sys.path.insert(0, str(SCRIPT_DIR))
    from app import app, TopazEngine, ensure_models
    from autopilot import estimate_parameters

    out_folder_name = f"{args.scale}X_TOPAZ_{args.model.upper()}_{args.bit_depth}BIT"
    if target_path.is_dir():
        out_dir = target_path / out_folder_name
    else:
        out_dir = target_path.parent / out_folder_name
    out_dir.mkdir(parents=True, exist_ok=True)

    try:
        with app.run():
            ensure_models.remote()
            engine = TopazEngine()

            for idx, img_path in enumerate(image_files, 1):
                img_bytes = img_path.read_bytes()

                if use_auto:
                    auto_p = estimate_parameters(img_bytes, args.model)
                    denoise_val = auto_p["denoise"]
                    sharpen_val = auto_p["sharpen"]
                    comp_val = auto_p["compression"]
                    print(f"\n[{idx}/{len(image_files)}] Topaz Upscaling: {img_path.name} ({args.scale}x, {args.bit_depth}-bit)")
                    print(f"  -> Topaz Auto-AI: Denoise={denoise_val*100:.1f}%, Sharpen={sharpen_val*100:.1f}%, Compress={comp_val*100:.1f}%")
                else:
                    denoise_val = args.denoise if args.denoise is not None else 0.05
                    sharpen_val = args.sharpen if args.sharpen is not None else 0.45
                    comp_val = args.compression if args.compression is not None else 0.0
                    print(f"\n[{idx}/{len(image_files)}] Topaz Upscaling: {img_path.name} ({args.scale}x, {args.bit_depth}-bit)")
                    print(f"  -> Manual: Denoise={denoise_val*100:.1f}%, Sharpen={sharpen_val*100:.1f}%, Compress={comp_val*100:.1f}%")

                start_t = time.time()
                out_bytes = engine.upscale.remote(
                    img_bytes,
                    model_name=args.model,
                    scale=args.scale,
                    denoise=denoise_val,
                    sharpen=sharpen_val,
                    compression=comp_val,
                    bit_depth=args.bit_depth,
                )
                elapsed = time.time() - start_t

                stem = img_path.stem
                out_file = out_dir / f"{stem}_topaz_{args.model}_{args.scale}x_{args.bit_depth}bit.png"
                out_file.write_bytes(out_bytes)
                print(f"  -> Saved: {out_file.name} ({elapsed:.2f}s, {len(out_bytes)/1024/1024:.2f} MB)")
    finally:
        print(f"\n[Modal Guardian] Local run context exited. Check your provider dashboard for resource state and billing.")
        print(f"All results saved in: {out_dir}")

    # Open folder in macOS Finder automatically
    try:
        subprocess.run(["open", str(out_dir)], check=False)
    except Exception:
        pass


if __name__ == "__main__":
    main()
