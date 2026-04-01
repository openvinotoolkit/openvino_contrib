// SparseEncoder Operation for OpenVINO
//
// Full sparse encoder pipeline as a single OV custom op.
// Internally uses OpenCL for GPU-accelerated sparse convolution.
//
// Architecture (fixed, BEVFusion):
//   conv_input(5→16) → 4 encoder layers (8 residual blocks + 3 downsamples)
//   → conv_out(128→128, k=1×1×3, s=1×1×2) → dense → BEV [1,256,180,180]
//
// Inputs:
//   0: features    [MAX_N, 5]          FP32 — voxel mean features
//   1: coords      [MAX_N, 4]          I32  — [batch, x, y, z]
//   2: num_voxels  [1]                 I32  — actual voxel count
//   3: params      [TOTAL_PARAMS]      FP32 — packed weights + BN params
//
// Output:
//   0: bev_features [1, 256, 180, 180] FP32

#pragma once

#include <openvino/op/op.hpp>

namespace BEVFusionExtension {

// Fixed architecture constants
constexpr int64_t SPENC_MAX_N         = 60000;
constexpr int64_t SPENC_NUM_FEATURES  = 5;
constexpr int64_t SPENC_BEV_C         = 256;
constexpr int64_t SPENC_BEV_H         = 180;
constexpr int64_t SPENC_BEV_W         = 180;

// Layer structure (hardcoded for BEVFusion)
// Total conv layers: conv_input + 8 res convs (4 blocks × 2) + 3 downsamples + conv_out = 21
// Total param sets (weight+scale+bias): 21
//
// Parameter packing order:
//   conv_input:  weight[27*5*16] + scale[16] + bias[16]
//   res0_0_c1:   weight[27*16*16] + scale[16] + bias[16]
//   res0_0_c2:   weight[27*16*16] + scale[16] + bias[16]
//   res0_1_c1:   weight[27*16*16] + scale[16] + bias[16]
//   res0_1_c2:   weight[27*16*16] + scale[16] + bias[16]
//   ds0:         weight[27*16*32] + scale[32] + bias[32]
//   res1_0_c1:   weight[27*32*32] + scale[32] + bias[32]
//   ... (continues for all 21 layers)
//   conv_out:    weight[3*128*128] + scale[128] + bias[128]

// Total parameter count (precomputed):
//   Weights: 2160 + 4*6912 + 13824 + 4*27648 + 55296 + 4*110592 + 221184 + 4*442368 + 49152
//          = 2,691,696
//   Scales:  16*5 + 32*5 + 64*5 + 128*5 + 128 = 1,328
//   Biases:  same = 1,328
//   TOTAL = 2,694,352
constexpr int64_t SPENC_TOTAL_PARAMS  = 2694352;


class SparseEncoderOp : public ov::op::Op {
public:
    OPENVINO_OP("SparseEncoder", "bevfusion");

    SparseEncoderOp() = default;

    SparseEncoderOp(const ov::Output<ov::Node>& features,
                    const ov::Output<ov::Node>& coords,
                    const ov::Output<ov::Node>& num_voxels,
                    const ov::Output<ov::Node>& params);

    void validate_and_infer_types() override;
    std::shared_ptr<ov::Node> clone_with_new_inputs(
        const ov::OutputVector& new_args) const override;
    bool visit_attributes(ov::AttributeVisitor& visitor) override;

    bool evaluate(ov::TensorVector& outputs,
                  const ov::TensorVector& inputs) const override;
    bool has_evaluate() const override;
};

}  // namespace BEVFusionExtension
