#!/usr/bin/env bash
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

  local CONV_PY="${CONVERT_VENV_DIR}/bin/python"
  local CONV_PIP="${CONVERT_VENV_DIR}/bin/pip"

  if [[ ! -d "${CONVERT_VENV_DIR}" ]]; then
    info "  Creating conversion venv at ${CONVERT_VENV_DIR}"
    "$PYTHON310" -m venv "${CONVERT_VENV_DIR}"
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
  "$CONV_PIP" install --no-build-isolation --no-deps "mmdet3d==1.0.0rc4" -q
  "$CONV_PIP" install "opencv-python<4.10" tensorboard "numba>=0.56,<0.60" "networkx>=2.8,<3" "numpy==1.23.5" plyfile scikit-image "trimesh>=2.35.39,<2.35.40" -q
  "$CONV_PIP" install "openvino>=2024.0" onnx -q

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
    git clone --depth 50 --branch "$OV_BRANCH" "$OV_REPO" "$OV_CLONE_DIR"
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
"$VENV_PIP" install -r "${SCRIPT_DIR}/requirements_ov.txt" -q
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
