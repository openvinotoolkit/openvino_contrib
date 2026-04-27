#!/usr/bin/env python3
# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
"""
Export YOLOv12 (Ultralytics) to ONNX and OpenVINO IR.

Steps performed:
1. Create (if missing) a virtual environment at /Users/anesterov/venvs/ultralytics-yolo12
2. Install/upgrade ultralytics inside that venv
3. Download YOLOv12n weights and export to ONNX (ops set 12, imgsz=640)
4. Convert ONNX to OpenVINO IR using mo (CLI)

Paths:
  ONNX: /Users/anesterov/models/yolo12/yolov12n.onnx
  IR  : /Users/anesterov/models/yolo12/ir/yolov12n.xml
"""

import os
import subprocess
import sys
import venv
from pathlib import Path

VENV_DIR = Path("/Users/anesterov/venvs/ultralytics-yolo12")
MODEL_DIR = Path("/Users/anesterov/models/yolo12")
ONNX_PATH = MODEL_DIR / "yolov12n.onnx"
IR_DIR = MODEL_DIR / "ir"
IR_XML = IR_DIR / "yolov12n.xml"


def ensure_venv():
    if VENV_DIR.exists():
        print(f"[yolo12_export] Using existing venv: {VENV_DIR}")
        return
    print(f"[yolo12_export] Creating venv at {VENV_DIR}")
    VENV_DIR.parent.mkdir(parents=True, exist_ok=True)
    venv.EnvBuilder(with_pip=True, clear=False, symlinks=True).create(VENV_DIR)


def venv_python():
    return VENV_DIR / "bin" / "python"


def install_ultralytics():
    print("[yolo12_export] Installing/Upgrading ultralytics in venv...")
    subprocess.check_call([str(venv_python()), "-m", "pip", "install", "-U", "ultralytics"])


def export_onnx():
    print("[yolo12_export] Exporting YOLOv12n to ONNX...")
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    # Run the export inside the venv to ensure correct deps
    code = f"""
from ultralytics import YOLO
import os
os.chdir("{MODEL_DIR}")
model = YOLO("yolov12n.pt")
model.export(format="onnx", opset=12, imgsz=640)
"""
    subprocess.check_call([str(venv_python()), "-c", code])
    produced = MODEL_DIR / "yolov12n.onnx"
    if produced != ONNX_PATH:
        if produced.exists():
            produced.rename(ONNX_PATH)
    if not ONNX_PATH.exists():
        raise FileNotFoundError(f"ONNX export failed, file not found: {ONNX_PATH}")
    print(f"[yolo12_export] ONNX ready: {ONNX_PATH}")


def export_ir():
    print("[yolo12_export] Converting ONNX to OpenVINO IR via mo...")
    IR_DIR.mkdir(parents=True, exist_ok=True)
    cmd = [
        "mo",
        "--input_model", str(ONNX_PATH),
        "--output_dir", str(IR_DIR),
        "--model_name", "yolov12n",
        "--input_shape", "[1,3,640,640]",
        "--data_type", "FP32",
    ]
    subprocess.check_call(cmd)
    if not IR_XML.exists():
        raise FileNotFoundError(f"IR conversion failed, file not found: {IR_XML}")
    print(f"[yolo12_export] IR ready: {IR_XML}")


def main():
    ensure_venv()
    install_ultralytics()
    export_onnx()
    try:
        export_ir()
    except FileNotFoundError as e:
        print(f"[yolo12_export] Warning: IR not produced ({e}). "
              f"Ensure 'mo' is in PATH and rerun if needed.", file=sys.stderr)
        sys.exit(1)
    print("[yolo12_export] Done.")


if __name__ == "__main__":
    main()

