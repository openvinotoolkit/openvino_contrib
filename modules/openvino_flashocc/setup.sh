#!/usr/bin/env bash
# Copyright (C) 2018-2026 Intel Corporation
# =============================================================================
# setup.sh — FlashOCC OpenVINO 2026.3 environment setup
#
# What this script does:
#   1.  Checks for Python 3.10+ and required dependencies
#   2.  Prepares model IRs (first-time setup only)
#   3.  Creates venv_flashocc_ws with Python 3.10 or 3.11
#   4.  Installs OpenVINO from pip + requirements.txt into the venv
#   5.  Builds the bev_pool OpenVINO C++ extension
#   6.  Writes setup.env  (sourced by run_flashocc_ov_ws.sh)
#   7.  Optionally runs the benchmark  (--run-test)
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
#       [--run-test]
#
# After setup, run inference with:
#   bash run_flashocc_ov_ws.sh [--num-samples N] [--ov-device GPU|CPU]
# =============================================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# ── Defaults ─────────────────────────────────────────────────────────────────
VENV_DIR="${SCRIPT_DIR}/venv_flashocc_ws"
JOBS=$(nproc)

MODEL_DIR=""
MODEL_VARIANT="m0"
PREPARE_MODELS=0
NUM_SAMPLES=80
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

# Check for Python 3.10 or 3.11 (for runtime venv)
PYTHON310=$(command -v python3.10 2>/dev/null || true)
PYTHON311=$(command -v python3.11 2>/dev/null || true)
VENV_PYTHON="${PYTHON310:-${PYTHON311}}"
[[ -n "$VENV_PYTHON" ]] || die "python3.10 or python3.11 not found. Install with: sudo apt install python3.10 python3.10-venv"

command -v git &>/dev/null   || die "git not found"

ok "Prerequisites OK  ($($VENV_PYTHON --version))"

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

# Allow conversion/export-only workflow to complete cleanly without requiring
# runtime venv and dependencies.
if [[ $RUN_TEST -eq 0 ]]; then
  ok "Prepare-models-only flow complete (runtime venv setup skipped)"
  exit 0
fi

# ── Step 1: Create Python venv ───────────────────────────────────────────────
info "Step 1: Setting up virtual environment …"
if [[ -d "${VENV_DIR}" ]]; then
  warn "Venv already exists at ${VENV_DIR} — reinstalling packages"
else
  "${VENV_PYTHON}" -m venv "${VENV_DIR}"
fi

VENV_PY="${VENV_DIR}/bin/python3"
VENV_PIP="${VENV_DIR}/bin/pip"

"$VENV_PIP" install --upgrade pip setuptools wheel -q

# ── Step 2: Install OpenVINO from pip ────────────────────────────────────────
info "Step 2: Installing OpenVINO from pip …"
"$VENV_PIP" install "openvino>=2024.0" -q
ok "OV installed: $("$VENV_PY" -c 'import openvino; print(openvino.__version__)')"

# ── Step 3: Install requirements.txt ─────────────────────────────────────────
info "Step 3: Installing requirements.txt …"
"$VENV_PIP" install -r "${SCRIPT_DIR}/requirements.txt" -q
ok "Dependencies installed"

# ── Step 4: Build bev_pool OpenVINO extension ─────────────────────────────────
info "Step 4: Building bev_pool OpenVINO extension …"
BEV_SRC="${SCRIPT_DIR}/openvino_extensions/bev_pool"
BEV_BUILD_DIR="${SCRIPT_DIR}/openvino_extensions/bev_pool/build_ws"

# Find OpenVINO cmake config from pip installation
OV_CMAKE_DIR="$("$VENV_PY" -c "import openvino; import os; print(os.path.dirname(openvino.__file__))" 2>/dev/null || echo "")"
[[ -n "$OV_CMAKE_DIR" ]] || die "Failed to locate OpenVINO installation directory"
OV_CMAKE_DIR="${OV_CMAKE_DIR}/../openvino_cmake"

mkdir -p "$BEV_BUILD_DIR"

if command -v cmake &>/dev/null; then
  CMAKE=$(command -v cmake)
else
  die "cmake not found. Install with: sudo apt install cmake"
fi

if command -v ninja &>/dev/null; then
  BUILD_TOOL="Ninja"
else
  BUILD_TOOL="Unix Makefiles"
fi

run_quiet_step "Configuring bev_pool extension" "${SCRIPT_DIR}/ov_build/logs/bevpool_configure.log" \
  "$CMAKE" -S "$BEV_SRC" -B "$BEV_BUILD_DIR" \
  -G "${BUILD_TOOL}" \
  -DCMAKE_BUILD_TYPE=Release \
  -DOpenVINO_DIR="${OV_CMAKE_DIR}" \
  -DPython3_EXECUTABLE="${VENV_PY}"

run_quiet_step "Building bev_pool extension" "${SCRIPT_DIR}/ov_build/logs/bevpool_build.log" \
  "$CMAKE" --build "$BEV_BUILD_DIR" -j "${JOBS}"

BEV_SO="${BEV_BUILD_DIR}/libopenvino_bevpool_extension.so"
[[ -f "$BEV_SO" ]] || die "bev_pool extension .so not found after build at ${BEV_SO}"
ok "bev_pool extension built: ${BEV_SO}"

# ── Step 5: Write setup.env ──────────────────────────────────────────────────
info "Step 5: Writing setup.env …"
cat > "${SCRIPT_DIR}/setup.env" <<EOF
# Auto-generated by setup.sh — do not edit manually
FLASHOCC_VENV="${VENV_DIR}"
FLASHOCC_MODEL_DIR="${MODEL_DIR}"
FLASHOCC_BEV_SO="${BEV_SO}"
EOF
ok "setup.env written"

# ── Summary ───────────────────────────────────────────────────────────────────
echo ""
echo -e "${GREEN}══════════════════════════════════════════════════════════${NC}"
echo -e "${GREEN}  Setup complete!${NC}"
echo -e "${GREEN}══════════════════════════════════════════════════════════${NC}"
echo ""
echo "  OV version   : $("$VENV_PY" -c 'import openvino; print(openvino.__version__)')"
echo "  venv         : ${VENV_DIR}"
echo "  model dir    : ${MODEL_DIR}"
echo "  bev_pool ext : ${BEV_SO}"
echo ""
echo "  To run E2E benchmark:"
echo "    bash run_flashocc_ov_ws.sh [--num-samples ${NUM_SAMPLES}]"
echo ""

# ── Step 6: Optional E2E test ────────────────────────────────────────────────
if [[ $RUN_TEST -eq 1 ]]; then
  info "Step 6: Running E2E benchmark (${NUM_SAMPLES} samples) …"
  bash "${SCRIPT_DIR}/run_flashocc_ov_ws.sh" \
    --num-samples "$NUM_SAMPLES"
fi
