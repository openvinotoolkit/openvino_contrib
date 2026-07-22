#!/usr/bin/env bash
# Copyright (C) 2018-2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
# =============================================================================
# setup.sh — FlashOCC OpenVINO 2026.3 environment setup
#
# What this script does:
#   1.  Checks for Conda and required dependencies
#   2.  Prepares model IRs (first-time setup only)
#   3.  Creates Conda environments with Python 3.10
#   4.  Installs OpenVINO from pip + requirements.txt into the runtime Conda env
#   5.  Builds the bev_pool OpenVINO C++ extension
#   6.  Writes setup.env  (sourced by run_flashocc_ov_ws.sh)
#   7.  Optionally runs the benchmark  (--run-test)
#
# Usage:
#   bash setup.sh \
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
RUNTIME_CONDA_ENV_DIR="${SCRIPT_DIR}/.conda/flashocc_ws"
CONDA_PYTHON_VERSION="3.10"
JOBS=$(nproc)

MODEL_DIR=""
MODEL_VARIANT="m0"
PREPARE_MODELS=0
NUM_SAMPLES=80
RUN_TEST=0

# ── Colour helpers ────────────────────────────────────────────────────────────
RED='\033[0;31m'; GREEN='\033[0;32m'; YELLOW='\033[1;33m'; CYAN='\033[0;36m'; NC='\033[0m'
info()  { echo -e "${CYAN}[setup] $*${NC}"; }
ok()    { echo -e "${GREEN}[setup] ✓ $*${NC}"; }
warn()  { echo -e "${YELLOW}[setup] ⚠ $*${NC}"; }
die()   { echo -e "${RED}[setup] ✗ $*${NC}" >&2; exit 1; }

find_conda() {
  if [[ -n "${CONDA_EXE:-}" ]]; then
    return 0
  fi

  CONDA_EXE="$(command -v conda 2>/dev/null || true)"
  [[ -n "${CONDA_EXE}" ]] || die "conda not found. Install Miniconda/Mambaforge and ensure 'conda' is on PATH."
}

ensure_conda_env() {
  local env_dir="$1"
  local python_version="$2"
  local label="$3"

  find_conda
  if [[ -x "${env_dir}/bin/python" && -x "${env_dir}/bin/pip" ]]; then
    info "  Using existing ${label} Conda environment at ${env_dir}"
  else
    info "  Creating ${label} Conda environment at ${env_dir} (Python ${python_version})"
    mkdir -p "$(dirname "${env_dir}")"
    "${CONDA_EXE}" create -y -q -p "${env_dir}" "python=${python_version}" pip
  fi

  [[ -x "${env_dir}/bin/python" ]] || die "${label} Conda environment is missing python at ${env_dir}/bin/python"
  [[ -x "${env_dir}/bin/pip" ]] || die "${label} Conda environment is missing pip at ${env_dir}/bin/pip"
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

  info "Step A: Preparing model IRs (standalone, no mmdet required) …"

  # ── Conversion venv (Python 3 + CPU torch + openvino + onnx; no mmdet) ──────
  local CONV_VENV="${SCRIPT_DIR}/.venv_convert"
  if [[ ! -x "${CONV_VENV}/bin/python" ]]; then
    info "  Creating standalone conversion venv at ${CONV_VENV} …"
    python3 -m venv "${CONV_VENV}"
  fi

  local CONV_PY="${CONV_VENV}/bin/python"
  local CONV_PIP="${CONV_VENV}/bin/pip"

  info "  Installing standalone conversion dependencies (CPU-only PyTorch, no mmdet) …"
  "$CONV_PIP" install --upgrade pip wheel -q
  "$CONV_PIP" install gdown -q
  # CPU-only torch is sufficient for ONNX export; pin to wheels currently
  # published on the PyTorch CPU index so fresh machines resolve consistently.
  "$CONV_PIP" install "torch==2.6.0+cpu" "torchvision==0.21.0+cpu" --index-url https://download.pytorch.org/whl/cpu -q
  "$CONV_PIP" install "openvino>=2024.0" "onnx>=1.14" "numpy>=1.23.5" "opencv-python-headless>=4.8" -q

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
    export PYTHONPATH="${SCRIPT_DIR}:${PYTHONPATH:-}"
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

find_conda
command -v git &>/dev/null   || die "git not found"

ok "Prerequisites OK  ($("${CONDA_EXE}" --version))"

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
# runtime Conda environment and dependencies.
if [[ $RUN_TEST -eq 0 ]]; then
  ok "Prepare-models-only flow complete (runtime Conda environment setup skipped)"
  exit 0
fi

# ── Step 1: Create runtime Conda environment ─────────────────────────────────
info "Step 1: Setting up runtime Conda environment …"
ensure_conda_env "${RUNTIME_CONDA_ENV_DIR}" "${CONDA_PYTHON_VERSION}" "runtime"

RUNTIME_PY="${RUNTIME_CONDA_ENV_DIR}/bin/python"
RUNTIME_PIP="${RUNTIME_CONDA_ENV_DIR}/bin/pip"

"$RUNTIME_PIP" install --upgrade pip setuptools wheel -q

# ── Step 2: Install OpenVINO from pip ────────────────────────────────────────
info "Step 2: Installing OpenVINO from pip …"
"$RUNTIME_PIP" install "openvino>=2024.0" -q
ok "OV installed: $("$RUNTIME_PY" -c 'import openvino; print(openvino.__version__)')"

# ── Step 3: Install requirements.txt ─────────────────────────────────────────
info "Step 3: Installing requirements.txt …"
"$RUNTIME_PIP" install -r "${SCRIPT_DIR}/requirements.txt" -q
ok "Dependencies installed"

# ── Step 4: Build bev_pool OpenVINO extension ─────────────────────────────────
info "Step 4: Building bev_pool OpenVINO extension …"
BEV_SRC="${SCRIPT_DIR}/openvino_extensions/bev_pool"
BEV_BUILD_DIR="${SCRIPT_DIR}/openvino_extensions/bev_pool/build_ws"

# Find OpenVINO cmake config from pip installation
OV_CMAKE_DIR="$("$RUNTIME_PY" -c "import openvino; import os; print(os.path.dirname(openvino.__file__))" 2>/dev/null || echo "")"
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
  -DPython3_EXECUTABLE="${RUNTIME_PY}"

run_quiet_step "Building bev_pool extension" "${SCRIPT_DIR}/ov_build/logs/bevpool_build.log" \
  "$CMAKE" --build "$BEV_BUILD_DIR" -j "${JOBS}"

BEV_SO="${BEV_BUILD_DIR}/libopenvino_bevpool_extension.so"
[[ -f "$BEV_SO" ]] || die "bev_pool extension .so not found after build at ${BEV_SO}"
ok "bev_pool extension built: ${BEV_SO}"

# ── Step 5: Write setup.env ──────────────────────────────────────────────────
info "Step 5: Writing setup.env …"
cat > "${SCRIPT_DIR}/setup.env" <<EOF
# Auto-generated by setup.sh — do not edit manually
FLASHOCC_CONDA_ENV="${RUNTIME_CONDA_ENV_DIR}"
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
echo "  OV version   : $("$RUNTIME_PY" -c 'import openvino; print(openvino.__version__)')"
echo "  conda env    : ${RUNTIME_CONDA_ENV_DIR}"
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
