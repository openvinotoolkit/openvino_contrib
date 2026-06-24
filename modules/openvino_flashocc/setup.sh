#!/usr/bin/env bash
# Copyright (C) 2018-2026 Intel Corporation
# =============================================================================
# setup.sh — FlashOCC OpenVINO 2026.3 environment setup
#
# What this script does:
#   1.  Checks for Python 3.12 and cmake / build dependencies
#   2.  Clones deepaks2/openvino  (branch: gpu-flashocc-fixes) — contains
#       5 GPU kernel patches required for correct FlashOCC inference:
#         • permute_tile_8x8_4x4 for [0,2,1,3] order
#         • activation_opt kernel for Softplus + F16 + dynamic batch
#         • Softplus F16 numerical stability fix
#         • bfyx_to_bfyx_f16 conv kernel for dynamic/multi-batch models
#         • select_preferred_formats: prefer clDNN for ≤4-channel convolutions
#           (eliminates image_encoder conv1 layout reorder → -2.34ms / frame)
#   3.  Builds the OpenVINO GPU plugin + Python wheel  (Release, ~20 min)
#   4.  Creates venv_ov2026_ws with Python 3.12
#   5.  Installs the built OV wheel + requirements.txt into the venv
#   6.  Builds the bev_pool OpenVINO C++ extension
#   7.  Writes setup.env  (sourced by run_flashocc_ov_ws.sh)
#   8.  Optionally runs the 80-sample E2E benchmark  (--run-test)
#
# Usage:
#   bash setup.sh \
#       [--flashocc-ref <commit|tag|branch>] \
#       [--flashocc-patch-dir /path/to/patches] \
#       [--no-sync-thirdparty-projects] \
#       [--num-samples 80] \
#       [--jobs 16] \
#       [--model-dir /path/to/split_f16out] \
#       [--model-variant m0|m1] \
#       [--prepare-models]      # force regeneration even if models exist
#       [--skip-ov-build]     # reuse existing ov_build/ clone + build
#       [--skip-bevpool-build] \
#       [--run-test]
#
# After setup, run inference with:
#   bash run_flashocc_ov_ws.sh [--num-samples N] [--ov-device GPU|CPU]
# =============================================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# ── Defaults ─────────────────────────────────────────────────────────────────
OV_REPO="https://github.com/deepaks2/openvino.git"
OV_BRANCH="gpu-flashocc-fixes"
OV_CLONE_DIR="${SCRIPT_DIR}/ov_build/openvino"
OV_BUILD_DIR="${OV_CLONE_DIR}/build"
DEFAULT_LOCAL_OV_DIR="$(realpath -m "${SCRIPT_DIR}/../../openvino")"
OV_LOCAL_REPO_DIR="${OV_LOCAL_REPO_DIR:-}"
if [[ -z "${OV_LOCAL_REPO_DIR}" && -d "${DEFAULT_LOCAL_OV_DIR}/.git" ]]; then
  OV_LOCAL_REPO_DIR="${DEFAULT_LOCAL_OV_DIR}"
fi
OV_LOCAL_BUILD_DIR="${OV_LOCAL_BUILD_DIR:-${OV_LOCAL_REPO_DIR:+${OV_LOCAL_REPO_DIR}/build}}"
VENV_DIR="${SCRIPT_DIR}/venv_ov2026_ws"
BEV_BUILD_DIR="${SCRIPT_DIR}/openvino_extensions/bev_pool/build_ws"
CMAKE_PY_VENV_DIR="${SCRIPT_DIR}/venv_cmake_py310"
JOBS=$(nproc)

MODEL_DIR=""
MODEL_VARIANT="m0"
PREPARE_MODELS=0
NUM_SAMPLES=80
SKIP_OV_BUILD=0
SKIP_BEVPOOL_BUILD=0
RUN_TEST=0
CONVERT_VENV_DIR="${SCRIPT_DIR}/venv_convert"
MMDET3D_REPO="https://github.com/open-mmlab/mmdetection3d.git"
MMDET3D_DIR="${SCRIPT_DIR}/mmdetection3d"
MMDET3D_BRANCH="v1.0.0rc4"

FLASHOCC_REPO="https://github.com/Yzichen/FlashOCC.git"
FLASHOCC_REF="master"
FLASHOCC_CACHE_DIR="${SCRIPT_DIR}/.cache/flashocc-upstream"
FLASHOCC_PATCH_DIR="${SCRIPT_DIR}/patches/flashocc"
FLASHOCC_PROJECTS_DIR="${SCRIPT_DIR}/projects"
SYNC_THIRDPARTY_PROJECTS=1

# ── Colour helpers ────────────────────────────────────────────────────────────
RED='\033[0;31m'; GREEN='\033[0;32m'; YELLOW='\033[1;33m'; CYAN='\033[0;36m'; NC='\033[0m'
info()  { echo -e "${CYAN}[setup] $*${NC}"; }
ok()    { echo -e "${GREEN}[setup] ✓ $*${NC}"; }
warn()  { echo -e "${YELLOW}[setup] ⚠ $*${NC}"; }
die()   { echo -e "${RED}[setup] ✗ $*${NC}" >&2; exit 1; }

use_local_openvino_checkout() {
  [[ -n "${OV_LOCAL_REPO_DIR}" && -d "${OV_LOCAL_REPO_DIR}/.git" ]] || return 1
  OV_CLONE_DIR="${OV_LOCAL_REPO_DIR}"
  OV_BUILD_DIR="${OV_LOCAL_BUILD_DIR:-${OV_CLONE_DIR}/build}"
  warn "Falling back to local OpenVINO checkout: ${OV_CLONE_DIR}"
  return 0
}

run_quiet_step() {
  local step_name="$1"
  local log_file="$2"
  shift 2

  mkdir -p "$(dirname "${log_file}")"
  info "  ${step_name} (log: ${log_file})"
  if ! "$@" >"${log_file}" 2>&1; then
    warn "${step_name} failed. Last 80 log lines:"
    tail -80 "${log_file}" >&2 || true
    return 1
  fi
}

sync_thirdparty_flashocc_projects() {
  if [[ ${SYNC_THIRDPARTY_PROJECTS} -eq 0 ]]; then
    info "Step A0: Skipping third-party FlashOCC sync (--no-sync-thirdparty-projects)"
    return 0
  fi

  command -v git &>/dev/null || die "git not found (required for third-party sync)"

  info "Step A0: Syncing third-party FlashOCC projects from ${FLASHOCC_REPO}@${FLASHOCC_REF} …"
  mkdir -p "$(dirname "${FLASHOCC_CACHE_DIR}")"

  if [[ -d "${FLASHOCC_CACHE_DIR}/.git" ]]; then
    run_quiet_step "Fetching FlashOCC upstream" "${SCRIPT_DIR}/ov_build/logs/flashocc_fetch.log" \
      git -C "${FLASHOCC_CACHE_DIR}" fetch --prune --tags origin
  else
    rm -rf "${FLASHOCC_CACHE_DIR}"
    run_quiet_step "Cloning FlashOCC upstream" "${SCRIPT_DIR}/ov_build/logs/flashocc_clone.log" \
      git clone --filter=blob:none "${FLASHOCC_REPO}" "${FLASHOCC_CACHE_DIR}"
  fi

  run_quiet_step "Resolving FlashOCC ref" "${SCRIPT_DIR}/ov_build/logs/flashocc_checkout.log" \
    git -C "${FLASHOCC_CACHE_DIR}" checkout --force "${FLASHOCC_REF}"
  run_quiet_step "Updating FlashOCC to ref" "${SCRIPT_DIR}/ov_build/logs/flashocc_reset.log" \
    git -C "${FLASHOCC_CACHE_DIR}" reset --hard "${FLASHOCC_REF}"

  local resolved_ref
  resolved_ref="$(git -C "${FLASHOCC_CACHE_DIR}" rev-parse HEAD)"
  [[ -d "${FLASHOCC_CACHE_DIR}/projects" ]] || die "Upstream FlashOCC ref ${resolved_ref} does not contain projects/"

  local staging
  staging="$(mktemp -d)"
  trap 'rm -rf "${staging}"' RETURN

  mkdir -p "${staging}/projects"
  cp -a "${FLASHOCC_CACHE_DIR}/projects/." "${staging}/projects/"

  git -C "${staging}" init -q
  git -C "${staging}" add projects
  git -C "${staging}" commit -q -m "base upstream projects" || true

  if [[ -d "${FLASHOCC_PATCH_DIR}" ]]; then
    mapfile -t patch_files < <(find "${FLASHOCC_PATCH_DIR}" -maxdepth 1 -type f -name '*.patch' | sort)
    if [[ ${#patch_files[@]} -gt 0 ]]; then
      info "  Applying ${#patch_files[@]} local patch(es) from ${FLASHOCC_PATCH_DIR}"
      for patch_file in "${patch_files[@]}"; do
        if git -C "${staging}" apply --check "${patch_file}"; then
          git -C "${staging}" apply "${patch_file}"
          continue
        fi

        # Fallback for patches created relative to projects/ root.
        if git -C "${staging}" apply --check --directory=projects "${patch_file}"; then
          git -C "${staging}" apply --directory=projects "${patch_file}"
          continue
        fi

        die "Failed to apply patch ${patch_file} to upstream FlashOCC projects"
      done
    else
      warn "Patch directory exists but contains no *.patch files: ${FLASHOCC_PATCH_DIR}"
    fi
  else
    warn "No patch directory found at ${FLASHOCC_PATCH_DIR}; using pure upstream projects/"
  fi

  rm -rf "${FLASHOCC_PROJECTS_DIR}"
  mkdir -p "${FLASHOCC_PROJECTS_DIR}"
  cp -a "${staging}/projects/." "${FLASHOCC_PROJECTS_DIR}/"

  cat > "${FLASHOCC_PROJECTS_DIR}/.upstream-source" <<EOF
repo=${FLASHOCC_REPO}
ref=${FLASHOCC_REF}
resolved_commit=${resolved_ref}
patch_dir=${FLASHOCC_PATCH_DIR}
generated_at_utc=$(date -u +%Y-%m-%dT%H:%M:%SZ)
EOF

  ok "Third-party projects synced from ${resolved_ref}"
}

usage() {
  sed -n '/^# Usage:/,/^# ===/p' "$0" | head -20
  exit 0
}

have_required_ir_files() {
  local d="$1"
  [[ -f "${d}/image_encoder.xml" && -f "${d}/image_encoder.bin" && \
     -f "${d}/bev_trunk.xml" && -f "${d}/bev_trunk.bin" && \
     -f "${d}/bev_trunk_argmax_only.xml" && -f "${d}/bev_trunk_argmax_only.bin" ]]
}

prepare_models_if_needed() {
  local target_dir="$1"

  if [[ ${PREPARE_MODELS} -eq 0 ]] && have_required_ir_files "$target_dir"; then
    ok "Model IRs found in ${target_dir}"
    return 0
  fi

  info "Step A: Preparing model IRs (first-time setup path) …"

  if [[ ! -d "${MMDET3D_DIR}/configs/_base_" ]]; then
    info "  Cloning mmdetection3d config dependency (${MMDET3D_BRANCH}) …"
    git clone --depth 1 --branch "${MMDET3D_BRANCH}" "${MMDET3D_REPO}" "${MMDET3D_DIR}"
  fi

  local PYTHON310
  PYTHON310=$(command -v python3.10 2>/dev/null || true)
  [[ -n "$PYTHON310" ]] || die "python3.10 not found. Install with: sudo apt install python3.10 python3.10-venv"

  # Try Python 3.9 for conversion venv (numba 0.53.0 needs Python < 3.10)
  # Fall back to Python 3.10 with newer numba if 3.9 unavailable
  PYTHON39=$(command -v python3.9 2>/dev/null || true)
  CONV_PYTHON="${PYTHON39:-$PYTHON310}"
  CONV_NUMBA_VER="0.53.0"
  CONV_NETWORKX_SPEC="networkx>=2.2,<2.3"
  
  if [[ "$CONV_PYTHON" == "$PYTHON310" ]]; then
    # Python 3.10: use the OpenVINO conversion env pin documented for this repo.
    CONV_NUMBA_VER="0.59.1"
    CONV_NETWORKX_SPEC="networkx>=2.8,<3"
    info "  Using Python 3.10 for conversion (numba ${CONV_NUMBA_VER}, ${CONV_NETWORKX_SPEC})"
  else
    info "  Using Python 3.9 for conversion (numba 0.53.0)"
  fi

  local CONV_PY="${CONVERT_VENV_DIR}/bin/python"
  local CONV_PIP="${CONVERT_VENV_DIR}/bin/pip"

  if [[ ! -d "${CONVERT_VENV_DIR}" ]]; then
    info "  Creating conversion venv at ${CONVERT_VENV_DIR}"
    "$CONV_PYTHON" -m venv "${CONVERT_VENV_DIR}"
  fi

  info "  Installing conversion dependencies (CPU-only PyTorch stack) …"
  # Keep setuptools in a range where pkg_resources is present for legacy setup.py builds.
  "$CONV_PIP" install --upgrade pip "setuptools<81" wheel -q
  "$CONV_PIP" install gdown -q
  "$CONV_PIP" install "torch==1.13.1+cpu" "torchvision==0.14.1+cpu" --index-url https://download.pytorch.org/whl/cpu -q
  # mmdet imports mmcv.ops (mmcv._ext), so conversion needs mmcv-full.
  # Disable build isolation to reuse venv torch/setuptools if a source build is needed.
  "$CONV_PIP" install --no-build-isolation "mmcv-full==1.6.0" -q
  "$CONV_PIP" install "mmdet==2.28.2" "mmsegmentation==0.30.0" -q
    # Install mmdet3d with corrected dependency versions
    # Using --no-deps to skip problematic pinned deps, then install all required core deps explicitly
    "$CONV_PIP" install --no-build-isolation --no-deps "mmdet3d==1.0.0rc4" -q
    "$CONV_PIP" install \
      "numba==${CONV_NUMBA_VER}" \
      "${CONV_NETWORKX_SPEC}" \
      "numpy>=1.23.5" \
      "plyfile" \
      "scikit-image" \
      "tensorboard" \
      "trimesh>=2.35.39,<2.35.40" \
      -q
  # Patch mmdet3d to skip optional dataset SDK imports (not needed for conversion)
  CONVERT_SITE_ROOT="${CONVERT_VENV_DIR}" "${CONVERT_VENV_DIR}/bin/python" - <<'PY_PATCH'
import glob
import os
import pathlib
import sys

site_packages = glob.glob(os.path.join(os.environ['CONVERT_SITE_ROOT'], 'lib', 'python*', 'site-packages'))
if not site_packages:
    print('⚠ site-packages not found, skipping mmdet3d patch')
    sys.exit(0)

root = pathlib.Path(site_packages[0]) / 'mmdet3d'
patches = {
    root / 'core' / 'evaluation' / '__init__.py': [
        ('from .lyft_eval import lyft_eval\n', '# from .lyft_eval import lyft_eval  # Optional: requires lyft_dataset_sdk\n'),
        ('from .nuscenes_eval import nuscenes_eval\n', '# from .nuscenes_eval import nuscenes_eval  # Optional: requires nuscenes-devkit\n'),
        ("    'kitti_eval_coco_style', 'kitti_eval', 'indoor_eval', 'lyft_eval',\n", "    'kitti_eval_coco_style', 'kitti_eval', 'indoor_eval',\n"),
    ],
    root / 'datasets' / '__init__.py': [
        ('from .lyft_dataset import LyftDataset\n', '# from .lyft_dataset import LyftDataset  # Optional: requires lyft_dataset_sdk\n'),
        ('from .nuscenes_dataset import NuScenesDataset\n', '# from .nuscenes_dataset import NuScenesDataset  # Optional: requires nuscenes-devkit\n'),
        ('from .nuscenes_mono_dataset import NuScenesMonoDataset\n', '# from .nuscenes_mono_dataset import NuScenesMonoDataset  # Optional: requires nuscenes-devkit\n'),
        ("    'build_dataset', 'NuScenesDataset', 'NuScenesMonoDataset', 'LyftDataset',\n", "    'build_dataset',\n"),
    ],
}

for path, replacements in patches.items():
    if not path.exists():
        continue
    content = path.read_text()
    updated = content
    for old, new in replacements:
        updated = updated.replace(old, new)
    path.write_text(updated)

print('✓ Patched mmdet3d to skip optional dataset SDK imports')
PY_PATCH
  "$CONV_PIP" install "opencv-python<4.10" "openvino>=2024.0" onnx -q

  mkdir -p "${SCRIPT_DIR}/checkpoints"
  local CKPT_PATH=""
  case "${MODEL_VARIANT}" in
    m0)
      CKPT_PATH="${SCRIPT_DIR}/checkpoints/flashocc-r50.pth"
      if [[ ! -f "${CKPT_PATH}" ]]; then
        info "  Downloading M0 checkpoint …"
        "$CONV_PY" -m gdown 14my3jdqiIv6VIrkozQ6-ruEcBOPVlWGJ -O "${CKPT_PATH}"
      fi
      ;;
    m1)
      CKPT_PATH="${SCRIPT_DIR}/checkpoints/flashocc-r50-m1.pth"
      if [[ ! -f "${CKPT_PATH}" ]]; then
        info "  Downloading M1 checkpoint …"
        "$CONV_PY" -m gdown 1k9BzXB2nRyvXhqf7GQx3XNSej6Oq6I-B -O "${CKPT_PATH}"
      fi
      ;;
    *)
      die "Unsupported --model-variant: ${MODEL_VARIANT} (expected m0 or m1)"
      ;;
  esac

  info "  Exporting split OpenVINO models to ${target_dir} …"
  (
    cd "${SCRIPT_DIR}"
    export PYTHONPATH="${SCRIPT_DIR}:${SCRIPT_DIR}/projects:${PYTHONPATH:-}"
    export FLASHOCC_CONVERSION_SAFE_IMPORTS=1
    "$CONV_PY" "${SCRIPT_DIR}/convert_to_openvino.py" \
      --model-variant "${MODEL_VARIANT}" \
      --export-mode split \
      --split-out-dir "${target_dir}"
  )

  have_required_ir_files "$target_dir" || die "Model conversion finished but required IR files are missing in ${target_dir}"
  ok "Model IRs ready: ${target_dir}"
}

# ── Argument parsing ──────────────────────────────────────────────────────────
while [[ $# -gt 0 ]]; do
  case "$1" in
    --flashocc-ref)      FLASHOCC_REF="$2"; shift 2 ;;
    --flashocc-patch-dir) FLASHOCC_PATCH_DIR="$2"; shift 2 ;;
    --no-sync-thirdparty-projects) SYNC_THIRDPARTY_PROJECTS=0; shift ;;
    --model-dir)         MODEL_DIR="$2";    shift 2 ;;
    --model-variant)     MODEL_VARIANT="$2"; shift 2 ;;
    --prepare-models)    PREPARE_MODELS=1;   shift ;;
    --num-samples)       NUM_SAMPLES="$2";  shift 2 ;;
    --jobs)              JOBS="$2";         shift 2 ;;
    --skip-ov-build)     SKIP_OV_BUILD=1;   shift ;;
    --skip-bevpool-build) SKIP_BEVPOOL_BUILD=1; shift ;;
    --run-test)          RUN_TEST=1;        shift ;;
    -h|--help)           usage ;;
    *) die "Unknown option: $1  (run with --help for usage)" ;;
  esac
done

if [[ -z "$MODEL_DIR" ]]; then
  MODEL_DIR="${SCRIPT_DIR}/split_f16out"
fi

case "$MODEL_VARIANT" in
  m0|m1) ;;
  *) die "Invalid --model-variant: ${MODEL_VARIANT} (expected m0 or m1)" ;;
esac

# ── Step 0: Pre-flight checks ─────────────────────────────────────────────────
info "Step 0: Checking prerequisites …"

PYTHON312=$(command -v python3.12 2>/dev/null || true)
[[ -n "$PYTHON312" ]] || die "python3.12 not found. Install with: sudo apt install python3.12 python3.12-venv python3.12-dev"

# Check for Python 3.10 (required for wheel building via cmake)
PYTHON310=$(command -v python3.10 2>/dev/null || true)
[[ -n "$PYTHON310" ]] || die "python3.10 not found for OpenVINO build. Install with: sudo apt install python3.10"

CMAKE=$(command -v cmake 2>/dev/null || true)
[[ -n "$CMAKE" ]] || die "cmake not found. Install with: sudo apt install cmake"

command -v git &>/dev/null   || die "git not found"
if command -v ninja &>/dev/null; then
  BUILD_TOOL="Ninja"
else
  warn "ninja not found, build will use make (slower)"
  BUILD_TOOL="Unix Makefiles"
fi
command -v cc  &>/dev/null   || die "C compiler not found. Install build-essential."

ok "Prerequisites OK  (Python $(python3.12 --version), cmake $($CMAKE --version | head -1 | awk '{print $3}'))"

# ── Step A0: Sync third-party FlashOCC projects ──────────────────────────────
sync_thirdparty_flashocc_projects

# ── Step A: Use existing models or auto-generate them ───────────────────────
if [[ -d "$MODEL_DIR" ]] && have_required_ir_files "$MODEL_DIR" && [[ $PREPARE_MODELS -eq 0 ]]; then
  ok "Using existing model dir: ${MODEL_DIR}"
else
  if [[ ! -d "$MODEL_DIR" ]]; then
    mkdir -p "$MODEL_DIR"
  fi
  prepare_models_if_needed "$MODEL_DIR"
fi

# ── Step 1: Clone OpenVINO ────────────────────────────────────────────────────
if [[ $SKIP_OV_BUILD -eq 0 ]]; then
  info "Step 1: Cloning ${OV_REPO}  (branch: ${OV_BRANCH}) …"
  if [[ -d "$OV_CLONE_DIR/.git" ]]; then
    warn "ov_build/openvino already exists — fetching latest commits"
    git -C "$OV_CLONE_DIR" fetch origin
    git -C "$OV_CLONE_DIR" checkout "$OV_BRANCH"
    git -C "$OV_CLONE_DIR" reset --hard "origin/${OV_BRANCH}"
  else
    mkdir -p "${SCRIPT_DIR}/ov_build"
    if ! git clone --depth 50 --branch "$OV_BRANCH" "$OV_REPO" "$OV_CLONE_DIR"; then
      use_local_openvino_checkout || die "Failed to clone ${OV_REPO} and no local OpenVINO checkout is available"
    fi
  fi

  # Init submodules (oneDNN + others needed for GPU build)
  info "  Updating submodules …"
  git -C "$OV_CLONE_DIR" submodule update --init --recursive \
    thirdparty/oneDNN \
    thirdparty/ocl/clhpp_headers \
    thirdparty/ocl/icd_loader \
    thirdparty/ittapi \
    thirdparty/telemetry \
    thirdparty/pugixml \
    2>/dev/null || git -C "$OV_CLONE_DIR" submodule update --init --recursive
  ok "Clone done: $(git -C "$OV_CLONE_DIR" log --oneline -1)"

  # ── Step 2: Build OpenVINO ─────────────────────────────────────────────────
  # Use Python 3.10 for wheel build (has distutils); will install wheel into Python 3.12 venv
  PYTHON310=$(command -v python3.10 2>/dev/null || true)
  [[ -n "$PYTHON310" ]] || die "python3.10 not found for OpenVINO build"

  # Use a dedicated Python 3.10 venv for cmake Python checks so we never depend
  # on externally-managed system site-packages.
  CMAKE_PYTHON310="${CMAKE_PY_VENV_DIR}/bin/python"
  if [[ ! -x "$CMAKE_PYTHON310" ]]; then
    info "  Creating cmake helper venv at ${CMAKE_PY_VENV_DIR}"
    "$PYTHON310" -m venv "${CMAKE_PY_VENV_DIR}"
  fi
  info "  Ensuring cmake Python deps in helper venv (packaging, setuptools>=80, wheel>=0.45) …"
  "${CMAKE_PYTHON310}" -m pip install --upgrade pip -q
  "${CMAKE_PYTHON310}" -m pip install -q packaging "setuptools>=80" "wheel>=0.45"
  "${CMAKE_PYTHON310}" -c 'import packaging, setuptools.command.bdist_wheel' >/dev/null 2>&1 || \
    die "cmake helper venv is missing required wheel-build modules"

  info "Step 2: Configuring OpenVINO cmake  (jobs: ${JOBS}) …"
  mkdir -p "$OV_BUILD_DIR"
  run_quiet_step "Configuring OpenVINO" "${SCRIPT_DIR}/ov_build/logs/cmake_configure.log" \
    cmake -S "$OV_CLONE_DIR" -B "$OV_BUILD_DIR" \
    -G "${BUILD_TOOL}" \
    -DCMAKE_BUILD_TYPE=Release \
    -DENABLE_INTEL_GPU=ON \
    -DGPU_RT_TYPE=OCL \
    -DENABLE_ONEDNN_FOR_GPU=ON \
    -DENABLE_PYTHON=ON \
    -DENABLE_PYTHON_API=ON \
    -DENABLE_WHEEL=ON \
    -DENABLE_PYTHON_PACKAGING=OFF \
    -DPython3_EXECUTABLE="${CMAKE_PYTHON310}" \
    -DENABLE_INTEL_CPU=OFF \
    -DENABLE_INTEL_NPU=OFF \
    -DENABLE_OV_TF_FRONTEND=OFF \
    -DENABLE_OV_PADDLE_FRONTEND=OFF \
    -DENABLE_OV_JAX_FRONTEND=OFF \
    -DENABLE_SAMPLES=OFF \
    -DENABLE_TESTS=OFF \
    -DENABLE_DOCS=OFF

  info "  Building OpenVINO GPU plugin + Python wheel … (this can take a long time)"
  run_quiet_step "Building OpenVINO GPU plugin + wheel" "${SCRIPT_DIR}/ov_build/logs/cmake_build.log" \
    cmake --build "$OV_BUILD_DIR" \
    --target openvino_intel_gpu_plugin ie_wheel \
    -j "${JOBS}"

  OV_WHEEL=$(ls "${OV_BUILD_DIR}"/wheels/openvino-*.whl 2>/dev/null | head -1)
  [[ -n "$OV_WHEEL" ]] || die "OV wheel not found in ${OV_BUILD_DIR}/wheels/ — build may have failed"
  ok "OpenVINO built: $(basename "$OV_WHEEL")"
else
  info "Step 1-2: Skipping OV clone/build (--skip-ov-build)"
  OV_WHEEL=$(ls "${OV_BUILD_DIR}"/wheels/openvino-*.whl 2>/dev/null | head -1)
  [[ -n "$OV_WHEEL" ]] || die "No wheel in ${OV_BUILD_DIR}/wheels/ — cannot skip build without existing wheel"
  ok "Reusing existing wheel: $(basename "$OV_WHEEL")"
fi

OV_RELEASE_DIR="${OV_CLONE_DIR}/bin/intel64/Release"
[[ -f "${OV_RELEASE_DIR}/libopenvino_intel_gpu_plugin.so" ]] || \
  die "GPU plugin .so not found at ${OV_RELEASE_DIR}"

# ── Step 3: Create Python venv matching wheel ABI ───────────────────────────
# Build currently produces cp310 wheel, so install into Python 3.10 venv.
VENV_PYTHON="python3.10"
if [[ "${OV_WHEEL}" == *"cp312"* ]]; then
  VENV_PYTHON="python3.12"
fi

info "Step 3: Setting up ${VENV_PYTHON} virtual environment …"
if [[ -d "${VENV_DIR}" ]]; then
  warn "Venv already exists at ${VENV_DIR} — reinstalling packages"
else
  "${VENV_PYTHON}" -m venv "${VENV_DIR}"
fi

VENV_PY="${VENV_DIR}/bin/python3"
VENV_PIP="${VENV_DIR}/bin/pip"

"$VENV_PIP" install --upgrade pip setuptools wheel -q

# ── Step 4: Install OV wheel ──────────────────────────────────────────────────
info "Step 4: Installing OpenVINO wheel into venv …"
"$VENV_PIP" install "${OV_WHEEL}" --force-reinstall -q
ok "OV installed: $("$VENV_PY" -c 'import openvino; print(openvino.__version__)')"

# ── Step 5: Install requirements.txt ─────────────────────────────────────────
info "Step 5: Installing requirements.txt …"
"$VENV_PIP" install -r "${SCRIPT_DIR}/requirements.txt" -q
ok "Dependencies installed"

# ── Step 6: Build bev_pool extension ─────────────────────────────────────────
if [[ $SKIP_BEVPOOL_BUILD -eq 0 ]]; then
  info "Step 6: Building bev_pool OpenVINO extension …"
  BEV_SRC="${SCRIPT_DIR}/openvino_extensions/bev_pool"
  OV_CMAKE_DIR="${OV_BUILD_DIR}"   # cmake config is in the build dir

  mkdir -p "$BEV_BUILD_DIR"
  run_quiet_step "Configuring bev_pool extension" "${SCRIPT_DIR}/ov_build/logs/bevpool_configure.log" \
    cmake -S "$BEV_SRC" -B "$BEV_BUILD_DIR" \
    -DCMAKE_BUILD_TYPE=Release \
    -DOpenVINO_DIR="${OV_CMAKE_DIR}" \
    -DPython3_EXECUTABLE="${VENV_PY}"

  run_quiet_step "Building bev_pool extension" "${SCRIPT_DIR}/ov_build/logs/bevpool_build.log" \
    cmake --build "$BEV_BUILD_DIR" -j "${JOBS}"
  BEV_SO="${BEV_BUILD_DIR}/libopenvino_bevpool_extension.so"
  [[ -f "$BEV_SO" ]] || die "bev_pool extension .so not found after build"
  ok "bev_pool extension built: ${BEV_SO}"
else
  info "Step 6: Skipping bev_pool build (--skip-bevpool-build)"
  BEV_SO="${BEV_BUILD_DIR}/libopenvino_bevpool_extension.so"
fi

# ── Step 7: Write setup.env ───────────────────────────────────────────────────
info "Step 7: Writing setup.env …"
cat > "${SCRIPT_DIR}/setup.env" <<EOF
# Auto-generated by setup.sh — do not edit manually
OV_RELEASE_DIR="${OV_RELEASE_DIR}"
OV_BEV_SO="${BEV_SO}"
FLASHOCC_MODEL_DIR="${MODEL_DIR}"
EOF
ok "setup.env written"

# ── Summary ───────────────────────────────────────────────────────────────────
echo ""
echo -e "${GREEN}══════════════════════════════════════════════════════════${NC}"
echo -e "${GREEN}  Setup complete!${NC}"
echo -e "${GREEN}══════════════════════════════════════════════════════════${NC}"
echo ""
echo "  OV version   : $("$VENV_PY" -c 'import openvino; print(openvino.__version__)')"
echo "  OV release   : ${OV_RELEASE_DIR}"
echo "  GPU plugin   : ${OV_RELEASE_DIR}/libopenvino_intel_gpu_plugin.so"
echo "  bev_pool ext : ${BEV_SO}"
echo "  venv         : ${VENV_DIR}"
echo "  model dir    : ${MODEL_DIR}"
echo ""
echo "  To run E2E benchmark:"
echo "    bash run_flashocc_ov_ws.sh [--num-samples ${NUM_SAMPLES}]"
echo ""

# ── Step 8: Optional E2E test ─────────────────────────────────────────────────
if [[ $RUN_TEST -eq 1 ]]; then
  info "Step 8: Running E2E benchmark (${NUM_SAMPLES} samples) …"
  bash "${SCRIPT_DIR}/run_flashocc_ov_ws.sh" \
    --num-samples "$NUM_SAMPLES"
fi
