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
#       --model-dir  /path/to/split_f16out \
#       --data-pkl   /path/to/nuscenes_infos_val.pkl \
#       --data-root  /path/to/nuscenes \
#       [--num-samples 80] \
#       [--jobs 16] \
#       [--skip-ov-build]     # reuse existing ov_build/ clone + build
#       [--skip-bevpool-build] \
#       [--run-test]
#
# After setup, run inference with:
#   bash run_flashocc_ov_ws.sh [--num-samples N] [--data-pkl ...] [...]
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
JOBS=$(nproc)

MODEL_DIR=""
DATA_PKL="${SCRIPT_DIR}/data/nuscenes/nuscenes_infos_val.pkl"
DATA_ROOT="${SCRIPT_DIR}/data/nuscenes"
NUM_SAMPLES=80
SKIP_OV_BUILD=0
SKIP_BEVPOOL_BUILD=0
RUN_TEST=0

# ── Colour helpers ────────────────────────────────────────────────────────────
RED='\033[0;31m'; GREEN='\033[0;32m'; YELLOW='\033[1;33m'; CYAN='\033[0;36m'; NC='\033[0m'
info()  { echo -e "${CYAN}[setup] $*${NC}"; }
ok()    { echo -e "${GREEN}[setup] ✓ $*${NC}"; }
warn()  { echo -e "${YELLOW}[setup] ⚠ $*${NC}"; }
die()   { echo -e "${RED}[setup] ✗ $*${NC}" >&2; exit 1; }

usage() {
  sed -n '/^# Usage:/,/^# ===/p' "$0" | head -20
  exit 0
}

# ── Argument parsing ──────────────────────────────────────────────────────────
while [[ $# -gt 0 ]]; do
  case "$1" in
    --model-dir)         MODEL_DIR="$2";    shift 2 ;;
    --data-pkl)          DATA_PKL="$2";     shift 2 ;;
    --data-root)         DATA_ROOT="$2";    shift 2 ;;
    --num-samples)       NUM_SAMPLES="$2";  shift 2 ;;
    --jobs)              JOBS="$2";         shift 2 ;;
    --skip-ov-build)     SKIP_OV_BUILD=1;   shift ;;
    --skip-bevpool-build) SKIP_BEVPOOL_BUILD=1; shift ;;
    --run-test)          RUN_TEST=1;        shift ;;
    -h|--help)           usage ;;
    *) die "Unknown option: $1  (run with --help for usage)" ;;
  esac
done

[[ -n "$MODEL_DIR" ]] || die "--model-dir is required.
  Point it to the split_f16out directory containing:
    image_encoder.xml/.bin
    bev_trunk.xml/.bin  (f16-output version)
    bev_trunk_argmax_only.xml/.bin  (fused bev_trunk+ArgMax)"

[[ -d "$MODEL_DIR" ]] || die "Model dir not found: $MODEL_DIR"

# ── Step 0: Pre-flight checks ─────────────────────────────────────────────────
info "Step 0: Checking prerequisites …"

PYTHON312=$(command -v python3.12 2>/dev/null || true)
[[ -n "$PYTHON312" ]] || die "python3.12 not found. Install with: sudo apt install python3.12 python3.12-venv python3.12-dev"

CMAKE=$(command -v cmake 2>/dev/null || true)
[[ -n "$CMAKE" ]] || die "cmake not found. Install with: sudo apt install cmake"

command -v git &>/dev/null   || die "git not found"
command -v ninja &>/dev/null || { warn "ninja not found, build will use make (slower)"; BUILD_TOOL="Unix Makefiles"; } && BUILD_TOOL="Ninja"
command -v cc  &>/dev/null   || die "C compiler not found. Install build-essential."

ok "Prerequisites OK  (Python $(python3.12 --version), cmake $($CMAKE --version | head -1 | awk '{print $3}'))"

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
  info "Step 2: Configuring OpenVINO cmake  (jobs: ${JOBS}) …"
  mkdir -p "$OV_BUILD_DIR"
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
    -DENABLE_INTEL_CPU=OFF \
    -DENABLE_INTEL_NPU=OFF \
    -DENABLE_OV_TF_FRONTEND=OFF \
    -DENABLE_OV_PADDLE_FRONTEND=OFF \
    -DENABLE_OV_JAX_FRONTEND=OFF \
    -DENABLE_SAMPLES=OFF \
    -DENABLE_TESTS=OFF \
    -DENABLE_DOCS=OFF \
    2>&1 | tail -15

  info "  Building OpenVINO GPU plugin + Python wheel …"
  cmake --build "$OV_BUILD_DIR" \
    --target openvino_intel_gpu_plugin ie_wheel \
    -j "${JOBS}" \
    2>&1 | tail -20

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

# ── Step 3: Create Python 3.12 venv ──────────────────────────────────────────
info "Step 3: Setting up Python 3.12 virtual environment …"
if [[ -d "${VENV_DIR}" ]]; then
  warn "Venv already exists at ${VENV_DIR} — reinstalling packages"
else
  python3.12 -m venv "${VENV_DIR}"
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
  cmake -S "$BEV_SRC" -B "$BEV_BUILD_DIR" \
    -DCMAKE_BUILD_TYPE=Release \
    -DOpenVINO_DIR="${OV_CMAKE_DIR}" \
    -DPython3_EXECUTABLE="${VENV_PY}" \
    2>&1 | tail -5

  cmake --build "$BEV_BUILD_DIR" -j "${JOBS}" 2>&1 | tail -5
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
FLASHOCC_DATA_PKL="${DATA_PKL}"
FLASHOCC_DATA_ROOT="${DATA_ROOT}"
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
    --num-samples "$NUM_SAMPLES" \
    --data-pkl    "$DATA_PKL" \
    --data-root   "$DATA_ROOT"
fi
