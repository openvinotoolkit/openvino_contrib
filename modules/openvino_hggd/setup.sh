#!/bin/bash
# Copyright (C) 2018-2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
# HGGD Intel iGPU — Environment Setup
# Creates a conda environment with all dependencies for OpenVINO GPU inference.
#
# Usage: bash setup.sh [ENV_NAME]
#   ENV_NAME defaults to "hggd_intel"

set -e

ENV_NAME="${1:-hggd_intel}"
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

echo "=== HGGD Intel Setup ==="
echo "Environment: $ENV_NAME"
echo "Script dir:  $SCRIPT_DIR"

# Check conda
if ! command -v conda &>/dev/null; then
    echo "ERROR: conda not found. Install miniforge first."
    exit 1
fi

# Create environment
if conda env list | grep -q "^${ENV_NAME} "; then
    echo "Environment '$ENV_NAME' already exists. Activate with: conda activate $ENV_NAME"
else
    echo "Creating conda environment: $ENV_NAME (Python 3.10)"
    conda create -n "$ENV_NAME" python=3.10 -y
fi

echo "Installing packages into $ENV_NAME..."
conda run -n "$ENV_NAME" pip install --upgrade pip

# Core Intel XPU runtime libraries used by hggd_xpu.
conda run -n "$ENV_NAME" pip install \
    intel-cmplr-lib-rt==2025.0.2 \
    intel-cmplr-lib-ur==2025.0.2 \
    intel-cmplr-lic-rt==2025.0.2 \
    intel-pti==0.10.0 \
    intel-sycl-rt==2025.0.2 \
    tcmlib==1.2.0 \
    umf==0.9.1

# Core dependencies
conda run -n "$ENV_NAME" pip install \
    torch==2.6.0+xpu \
    torchvision==0.21.0+xpu \
    pytorch-triton-xpu==3.2.0 \
    --index-url https://download.pytorch.org/whl/xpu

conda run -n "$ENV_NAME" pip install \
    openvino==2026.1.0 \
    numpy==2.0.2 \
    scipy==1.15.3 \
    open3d==0.19.0 \
    opencv-python==4.13.0.92 \
    scikit-image==0.25.2 \
    matplotlib==3.10.8 \
    pandas==2.3.3 \
    tensorboardX==2.6.4 \
    torchsummary==1.5.1 \
    tqdm==4.67.3 \
    transforms3d==0.4.2 \
    trimesh==4.11.4 \
    autolab_core==1.1.1 \
    cvxopt==1.3.3 \
    grasp-nms==1.0.2 \
    numba==0.64.0 \
    pillow==12.1.1

# graspnetAPI 1.2.11 has a stale numpy pin (1.20.3); hggd_xpu uses it with the newer stack.
conda run -n "$ENV_NAME" pip install --no-deps graspnetAPI==1.2.11

# Make the environment self-contained: activating it should prefer its own Intel runtime libs.
ENV_PREFIX="$(conda run -n "$ENV_NAME" python -c 'import sys; print(sys.prefix)')"
ACTIVATE_DIR="$ENV_PREFIX/etc/conda/activate.d"
DEACTIVATE_DIR="$ENV_PREFIX/etc/conda/deactivate.d"
mkdir -p "$ACTIVATE_DIR" "$DEACTIVATE_DIR"

cat > "$ACTIVATE_DIR/hggd_intel_runtime.sh" <<'EOF'
#!/bin/bash
export _HGGD_INTEL_OLD_LD_LIBRARY_PATH="${LD_LIBRARY_PATH:-}"
export LD_LIBRARY_PATH="$CONDA_PREFIX/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
EOF

cat > "$DEACTIVATE_DIR/hggd_intel_runtime.sh" <<'EOF'
#!/bin/bash
if [[ -n "${_HGGD_INTEL_OLD_LD_LIBRARY_PATH+x}" ]]; then
    if [[ -n "$_HGGD_INTEL_OLD_LD_LIBRARY_PATH" ]]; then
        export LD_LIBRARY_PATH="$_HGGD_INTEL_OLD_LD_LIBRARY_PATH"
    else
        unset LD_LIBRARY_PATH
    fi
    unset _HGGD_INTEL_OLD_LD_LIBRARY_PATH
fi
EOF

chmod +x "$ACTIVATE_DIR/hggd_intel_runtime.sh" "$DEACTIVATE_DIR/hggd_intel_runtime.sh"

# Build the OV point cloud extension
echo ""
echo "=== Building OpenVINO Point Cloud Extension ==="
EXT_DIR="$SCRIPT_DIR/ov_gpu_extensions"
if [[ -f "$EXT_DIR/build/libopenvino_pointcloud_extension.so" ]]; then
    echo "Extension already built: $EXT_DIR/build/libopenvino_pointcloud_extension.so"
else
    mkdir -p "$EXT_DIR/build"
    pushd "$EXT_DIR/build"
    conda run -n "$ENV_NAME" cmake ..
    conda run -n "$ENV_NAME" make -j"$(nproc)"
    popd
    echo "Built: $EXT_DIR/build/libopenvino_pointcloud_extension.so"
fi

echo ""
echo "=== Setup Complete ==="
echo "Activate:  conda activate $ENV_NAME"
echo "Run:       bash run.sh"
