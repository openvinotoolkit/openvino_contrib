#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
import venv
import zipfile
from pathlib import Path
from typing import Any

DEFAULT_MODEL_SPECS = [
    "standard=yolo26n.pt",
    "compact=yolov10n.pt",
]


def run(command: list[str], cwd: Path | None = None) -> None:
    subprocess.run(command, cwd=cwd, check=True)


def venv_python(venv_dir: Path) -> Path:
    if os.name == "nt":
        return venv_dir / "Scripts" / "python.exe"
    return venv_dir / "bin" / "python"


def ensure_venv(venv_dir: Path) -> Path:
    venv_dir = venv_dir.resolve()
    python = venv_python(venv_dir)
    if not python.exists():
        venv.EnvBuilder(with_pip=True).create(venv_dir)
        run([str(python), "-m", "pip", "install", "--upgrade", "pip"])
    return python


def ensure_python_packages(python: Path) -> None:
    check = subprocess.run(
        [str(python), "-c", "import openvino, ultralytics"],
        check=False,
    )
    if check.returncode != 0:
        run([str(python), "-m", "pip", "install", "-U", "openvino", "ultralytics"])


def export_model(
    python: Path,
    model_name: str,
    work_dir: Path,
    image_size: int,
) -> Path:
    export_code = f"""
from pathlib import Path
from ultralytics import YOLO

model = YOLO({model_name!r})
attempts = [
    dict(nms=True),
    dict(end2end=True),
]
last_error = None
for extra in attempts:
    try:
        model.export(format="openvino", imgsz={image_size}, batch=1, dynamic=False, **extra)
        break
    except Exception as exc:
        last_error = exc
else:
    raise last_error

expected = Path({model_name!r}).stem + "_openvino_model"
print(Path.cwd() / expected)
"""
    result = subprocess.run(
        [str(python), "-c", export_code],
        cwd=work_dir,
        check=True,
        capture_output=True,
        text=True,
    )
    model_dir = Path(result.stdout.strip().splitlines()[-1])
    if not model_dir.is_dir():
        raise FileNotFoundError(f"Ultralytics did not create OpenVINO model directory: {model_dir}")
    return model_dir


def parse_model_spec(model_spec: str) -> tuple[str, str]:
    alias, separator, model_name = model_spec.partition("=")
    if not separator:
        model_name = alias
        alias = Path(model_name).stem
    alias = alias.strip()
    model_name = model_name.strip()
    if not alias or not model_name:
        raise ValueError(f"Invalid model spec: {model_spec!r}. Expected alias=model.pt or model.pt.")
    return alias, model_name


def write_coco_names(
    python: Path,
    model_name: str,
    work_dir: Path,
    output_file: Path,
) -> None:
    names_code = f"""
from pathlib import Path
from ultralytics import YOLO

names = YOLO({model_name!r}).names
Path({str(output_file)!r}).write_text(
    "\\n".join(names[i] for i in range(len(names))) + "\\n",
    encoding="utf-8",
)
"""
    run([str(python), "-c", names_code], cwd=work_dir)


def verify_openvino_model(
    python: Path,
    model_xml: Path,
    image_size: int,
    model_alias: str,
) -> None:
    verify_code = f"""
from openvino import Core

core = Core()
model = core.read_model({str(model_xml)!r})
compiled = core.compile_model(model, "CPU")
outputs = list(compiled.outputs)
if not outputs:
    raise RuntimeError("Model has no outputs")
shape = [int(dim) for dim in outputs[0].shape]
if len(shape) < 2 or shape[-1] != 6:
    raise RuntimeError(f"Expected end-to-end YOLO output with last dimension 6, got {{shape}}")
inputs = list(compiled.inputs)
if not inputs:
    raise RuntimeError("Model has no inputs")
input_shape = [int(dim) for dim in inputs[0].shape]
if input_shape[-1] != {image_size} or input_shape[-2] != {image_size}:
    raise RuntimeError(f"Expected {image_size}x{image_size} input, got {{input_shape}}")
print("Verified OpenVINO YOLO model {model_alias}", input_shape, shape)
"""
    run([str(python), "-c", verify_code])


def copy_model_files(
    model_dir: Path,
    output_dir: Path,
    model_alias: str,
    model_stem: str,
) -> Path:
    output_model_dir = output_dir / model_alias
    output_model_dir.mkdir(parents=True, exist_ok=True)
    required = [f"{model_stem}.xml", f"{model_stem}.bin", "metadata.yaml"]
    missing = [name for name in required if not (model_dir / name).is_file()]
    if missing:
        raise FileNotFoundError(f"OpenVINO model export is missing required files: {missing}")

    for name in required:
        shutil.copy2(model_dir / name, output_model_dir / name)

    return output_model_dir


def sha256(file: Path) -> str:
    digest = hashlib.sha256()
    with file.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_manifest(
    output_dir: Path,
    models: list[dict[str, Any]],
    image_size: int,
) -> None:
    files = sorted(path for path in output_dir.rglob("*") if path.is_file())
    manifest: dict[str, Any] = {
        "format": "openvino-yolo-image-tagger-bundle",
        "image_size": image_size,
        "models": models,
        "files": {
            path.relative_to(output_dir).as_posix(): {
                "sha256": sha256(path),
                "size": path.stat().st_size,
            }
            for path in files
        },
    }
    (output_dir / "openvino_vision_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def write_zip(
    output_dir: Path,
    zip_output: Path,
) -> None:
    zip_output.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zip_output, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for file in sorted(output_dir.rglob("*")):
            if file.is_file():
                archive.write(file, file.relative_to(output_dir).as_posix())


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model",
        action="append",
        dest="models",
        help="Model spec as alias=model.pt or model.pt. Defaults to standard=yolo26n.pt and compact=yolov10n.pt.",
    )
    parser.add_argument("--image-size", type=int, default=640)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--work-dir", required=True, type=Path)
    parser.add_argument("--zip-output", type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output_dir = args.output_dir.resolve()
    args.work_dir = args.work_dir.resolve()
    if args.zip_output:
        args.zip_output = args.zip_output.resolve()
    args.work_dir.mkdir(parents=True, exist_ok=True)
    if args.output_dir.exists():
        shutil.rmtree(args.output_dir)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    python = ensure_venv(args.work_dir / ".venv")
    ensure_python_packages(python)

    model_specs = args.models or DEFAULT_MODEL_SPECS
    parsed_models = [parse_model_spec(model_spec) for model_spec in model_specs]
    write_coco_names(python, parsed_models[0][1], args.work_dir, args.output_dir / "coco.names")

    manifest_models: list[dict[str, Any]] = []
    for model_alias, model_name in parsed_models:
        model_dir = export_model(python, model_name, args.work_dir, args.image_size)
        model_stem = Path(model_name).stem
        output_model_dir = copy_model_files(model_dir, args.output_dir, model_alias, model_stem)
        verify_openvino_model(
            python,
            output_model_dir / f"{model_stem}.xml",
            args.image_size,
            model_alias,
        )
        manifest_models.append(
            {
                "id": model_alias,
                "source_model": model_name,
                "directory": model_alias,
                "xml": f"{model_stem}.xml",
            },
        )

    write_manifest(args.output_dir, manifest_models, args.image_size)
    if args.zip_output:
        write_zip(args.output_dir, args.zip_output)
        print(f"Wrote {args.zip_output}")


if __name__ == "__main__":
    main()
