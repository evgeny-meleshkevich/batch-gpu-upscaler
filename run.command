#!/bin/bash
set -e
cd "$(dirname "$0")"
TARGET=$(osascript -e 'POSIX path of (choose folder with prompt "Choose an image folder / Выберите папку изображений")')
PYTHON=python3
if [ -x .venv/bin/python ]; then PYTHON=.venv/bin/python; fi
"$PYTHON" upscale.py "$TARGET" --model cgi --scale 4 --bit-depth 16
