/*
 * Copyright (C) 2018-2026 Intel Corporation
 * SPDX-License-Identifier: Apache-2.0
 */


// SparseConv3d Operation for OpenVINO — SAM 3D Objects
//
// Full sparse 3D convolution engine as a single OV custom op, following
// the BEVFusion SparseEncoder pattern. Internally uses OpenCL for
// GPU-accelerated sparse convolution on Intel GPUs.
//
// Supports:
//   - SubMConv3d (submanifold convolution, preserves sparsity)
//   - SparseConv3d (stride > 1, output coord set changes)
//   - SparseInverseConv3d (transpose / upsample)
//   - SparseLinear (per-voxel linear transform)
//   - SparseDownsample (coordinate-based average pooling)
//   - SparseUpsample (nearest-neighbor via cached indices)
//   - Residual blocks (conv + norm + act + conv + skip)
//
// Architecture (configurable via layer_defs attribute):
//   Processes a stack of sparse conv layers defined by packed metadata.
//   Each layer: (type, in_ch, out_ch, kernel_elems) with fused BN.
//
// Inputs:
//   0: features    [MAX_N, C_in]       FP32 — sparse voxel features
//   1: coords      [MAX_N, 4]          I32  — [batch_idx, x, y, z]
//   2: num_voxels  [1]                 I32  — actual voxel count
//   3: params      [TOTAL_PARAMS]      FP32 — packed weights + BN params
//
// Output:
//   0: out_features [MAX_N, C_out]     FP32 — transformed sparse features

#pragma once

#include <openvino/op/op.hpp>

namespace SAM3DExtension {

// Maximum supported voxel count (padded input size)
constexpr int64_t SPCONV_MAX_N = 65536;

// Hash table size for neighbor map construction (must be power of 2)
constexpr int64_t HASH_TABLE_SIZE = 131072;  // 2^17

// Maximum kernel elements (3×3×3 = 27)
constexpr int64_t MAX_KERNEL_ELEMS = 27;

// Layer types (must match Python-side LAYER_DEFS)
constexpr int LAYER_SUBM_CONV     = 0;  // SubMConv3d (stride=1, preserves coords)
constexpr int LAYER_RES_CONV      = 1;  // SubMConv3d inside a ResBlock
constexpr int LAYER_STRIDED_DOWN  = 2;  // SparseConv3d (stride=2, changes coords)
constexpr int LAYER_INV_CONV_UP   = 3;  // SparseInverseConv3d (transpose upsample)
constexpr int LAYER_LINEAR        = 4;  // Per-voxel linear (weight + bias)
constexpr int LAYER_SPARSE_TO_DENSE = 5; // Scatter to dense 3D grid
constexpr int LAYER_LAYERNORM_SILU = 6; // Affine LayerNorm + SiLU (fused resblock)
constexpr int LAYER_RESIDUAL_ADD   = 7; // Residual add + optional skip linear
constexpr int LAYER_NORM2_ACT      = 8; // Non-affine LN + scale/shift + SiLU


class SparseConv3dOp : public ov::op::Op {
public:
    OPENVINO_OP("SparseConv3dEngine", "sam3d");

    SparseConv3dOp() = default;

    SparseConv3dOp(const ov::Output<ov::Node>& features,
                   const ov::Output<ov::Node>& coords,
                   const ov::Output<ov::Node>& num_voxels,
                   const ov::Output<ov::Node>& params);

    // Variable-input constructor (for fused resblock with 6 inputs)
    explicit SparseConv3dOp(const ov::OutputVector& args);

    void validate_and_infer_types() override;
    std::shared_ptr<ov::Node> clone_with_new_inputs(
        const ov::OutputVector& new_args) const override;
    bool visit_attributes(ov::AttributeVisitor& visitor) override;

    bool evaluate(ov::TensorVector& outputs,
                  const ov::TensorVector& inputs) const override;
    bool has_evaluate() const override;

private:
    // Attributes (serialized to/from IR XML)
    int64_t m_max_voxels = SPCONV_MAX_N;
    int64_t m_in_channels = 8;
    int64_t m_out_channels = 8;
    int64_t m_num_layers = 0;
    int64_t m_spatial_resolution = 64;

    // Layer definitions encoded as flat int array:
    // [type_0, cin_0, cout_0, nk_0, type_1, cin_1, cout_1, nk_1, ...]
    std::vector<int64_t> m_layer_defs;
};

// ============================================================
// SparseMeshUpsample — FP32 mesh decoder upsample path.
// Runs both SparseSubdivideBlock3d blocks + out_layer SparseLinear.
// Inputs:
//   0: features   [MAX_N, 768]  FP32 — mesh_hidden_feats from mesh_attn
//   1: coords     [MAX_N, 4]    I32  — [batch, z, y, x], res 64
//   2: num_voxels [1]           I32  — actual voxel count
//   3: params     [P]           FP32 — packed block + out_layer weights
// Outputs:
//   0: out_features [MAX_OUT, 101] FP32 — per-voxel mesh attrs (res 256)
//   1: out_coords   [MAX_OUT, 4]   I32  — [batch, z, y, x], res 256
//   2: out_num      [1]            I32  — actual output voxel count
// ============================================================
class SparseMeshUpsampleOp : public ov::op::Op {
public:
    OPENVINO_OP("SparseMeshUpsample", "sam3d");

    SparseMeshUpsampleOp() = default;

    SparseMeshUpsampleOp(const ov::Output<ov::Node>& features,
                         const ov::Output<ov::Node>& coords,
                         const ov::Output<ov::Node>& num_voxels,
                         const ov::Output<ov::Node>& params);

    explicit SparseMeshUpsampleOp(const ov::OutputVector& args);

    void validate_and_infer_types() override;
    std::shared_ptr<ov::Node> clone_with_new_inputs(
        const ov::OutputVector& new_args) const override;
    bool visit_attributes(ov::AttributeVisitor& visitor) override;

    bool evaluate(ov::TensorVector& outputs,
                  const ov::TensorVector& inputs) const override;
    bool has_evaluate() const override;

private:
    int64_t m_max_out = SPCONV_MAX_N * 64;  // 8x per block, 2 blocks
    // Channel dims: {Cin0,Cmid0,Cout0, Cin1,Cmid1,Cout1, out_in, out_out}
    std::vector<int64_t> m_dims = {768, 192, 192, 192, 96, 96, 96, 101};
};

}  // namespace SAM3DExtension
