#!/bin/bash
# Copyright (C) 2018-2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#
# UNDO script for the OpenVINO SAM-6D port.
#
# Run this AFTER manually running setup_from_original.sh to restore this
# directory back to its original "patches-only" state:
#   - Deletes every restored original file and every patched file.
#   - Removes the setup completion marker.
#   - Prunes empty directories.
#
# It keeps ONLY the OpenVINO port files and patches (the KEEP list below).

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CODE_DIR="$SCRIPT_DIR/SAM-6D"
MARKER="$CODE_DIR/.ov_setup_done"

[ -d "$CODE_DIR" ] || { echo "[FAIL] Code directory not found: $CODE_DIR"; exit 1; }

# --- KEEP list: OpenVINO port files + patches (relative to SAM-6D/) ----------
KEEP_LIST="$(cat <<'EOF'
Data/Example/gt_pose.json
Data/Example/mask_visib/000003_000001.png
Data/Example/models_info.json
Instance_Segmentation_Model/eval_ism_ov_bop.py
Instance_Segmentation_Model/eval_utils.py
Instance_Segmentation_Model/export_ism.py
Instance_Segmentation_Model/.gitignore
Instance_Segmentation_Model/infer_ism_ov.py
Instance_Segmentation_Model/ov_sam_layer_norm.py
Instance_Segmentation_Model/run_ism.sh
ov_environment_u24.yaml
patches/ism.patch
patches/pem.patch
Pose_Estimation_Model/eval_utils.py
Pose_Estimation_Model/common_infer_utils.py
Pose_Estimation_Model/run_pos.sh
Pose_Estimation_Model/.gitattributes
Pose_Estimation_Model/model/ov_pointnet2_op/ball_query.cl
Pose_Estimation_Model/model/ov_pointnet2_op/ball_query.cpp
Pose_Estimation_Model/model/ov_pointnet2_op/ball_query.hpp
Pose_Estimation_Model/model/ov_pointnet2_op/CMakeLists.txt
Pose_Estimation_Model/model/ov_pointnet2_op/custom_det.cl
Pose_Estimation_Model/model/ov_pointnet2_op/custom_det.cpp
Pose_Estimation_Model/model/ov_pointnet2_op/custom_det.hpp
Pose_Estimation_Model/model/ov_pointnet2_op/custom_svd_u.cl
Pose_Estimation_Model/model/ov_pointnet2_op/custom_svd_u.cpp
Pose_Estimation_Model/model/ov_pointnet2_op/custom_svd_u.hpp
Pose_Estimation_Model/model/ov_pointnet2_op/custom_svd_v.cl
Pose_Estimation_Model/model/ov_pointnet2_op/custom_svd_v.cpp
Pose_Estimation_Model/model/ov_pointnet2_op/custom_svd_v.hpp
Pose_Estimation_Model/model/ov_pointnet2_op/furthest_point_sampling.cl
Pose_Estimation_Model/model/ov_pointnet2_op/furthest_point_sampling.cpp
Pose_Estimation_Model/model/ov_pointnet2_op/furthest_point_sampling.hpp
Pose_Estimation_Model/model/ov_pointnet2_op/gather_operation.cl
Pose_Estimation_Model/model/ov_pointnet2_op/gather_operation.cpp
Pose_Estimation_Model/model/ov_pointnet2_op/gather_operation.hpp
Pose_Estimation_Model/model/ov_pointnet2_op/grouping_operation.cl
Pose_Estimation_Model/model/ov_pointnet2_op/grouping_operation.cpp
Pose_Estimation_Model/model/ov_pointnet2_op/grouping_operation.hpp
Pose_Estimation_Model/model/ov_pointnet2_op/ov_extension.cpp
Pose_Estimation_Model/model/ov_pointnet2_op/pem_gpu_ops.xml
Pose_Estimation_Model/model/ov_pointnet2_op/validate_gpu_ops_xml.py
Pose_Estimation_Model/pem_model_convert_ov_ir.py
Pose_Estimation_Model/run_inference_custom_openvino.py
Pose_Estimation_Model/run_inference_custom_pytorch.py
Pose_Estimation_Model/test_bop_subset_eval_ov.py
README.md
setup_env.sh
EOF
)"

# --- Delete everything not in the KEEP list ----------------------------------
echo "[UNDO] Restoring patches-only state under SAM-6D/ ..."
deleted=0
while IFS= read -r rel; do
    [ -n "$rel" ] || continue
    case $'\n'"$KEEP_LIST"$'\n' in
        *$'\n'"$rel"$'\n'*) : ;;                              # in KEEP list -> keep
        *) rm -f "$CODE_DIR/$rel"; deleted=$((deleted+1)) ;;  # not in KEEP list -> delete
    esac
done < <(cd "$CODE_DIR" && find . -type f | sed 's|^\./||')

# Remove the setup marker and prune empty directories.
rm -f "$MARKER"
find "$CODE_DIR" -depth -type d -empty -delete 2>/dev/null || true

# Reset module-root README.md to base state (attribution only)
cat > "$SCRIPT_DIR/README.md" <<'EOF'
<!--
Copyright (C) 2018-2026 Intel Corporation
SPDX-License-Identifier: Apache-2.0
-->

# OV-SAM-6D

This directory is an OpenVINO port of [SAM-6D](https://github.com/JiehongLin/SAM-6D) based on commit `1c2543b`.
The original SAM-6D source files are **not** committed here; they are fetched and patched by `setup_from_original.sh`.

For OpenVINO-specific setup and usage details, see [OV_README.md](OV_README.md).

---
EOF

# Remove restored original pics/
rm -rf "$SCRIPT_DIR/pics"

remaining="$(find "$CODE_DIR" -type f | wc -l)"
echo "[UNDO] Done. Deleted $deleted file(s); $remaining file(s) remain (patches-only state)."
