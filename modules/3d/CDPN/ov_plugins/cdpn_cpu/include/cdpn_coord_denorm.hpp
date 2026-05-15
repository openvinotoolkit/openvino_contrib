// CdpnCoordDenorm - coordinate map denormalisation + confidence min-max norm.
//
//   channels 0-2: multiply by obj_extents (denormalise)
//   channel 3:    min-max normalise to [0,1]
//
// Single-output design (4-channel output)
// Downstream uses VariadicSplit to separate.
//
// Inputs:
//   0: coord_maps  [1, 4, 64, 64] f32 - raw NN output
//   1: obj_extents [1, 1, 1, 3]   f32 - |min_x|, |min_y|, |min_z|
//
// Outputs:
//   0: combined    [1, 4, 64, 64] f32
//      channels 0-2: denormalised xyz coords
//      channel 3:    min-max normalised confidence

#pragma once

#include <openvino/op/op.hpp>

namespace CdpnExtension {

class CdpnCoordDenorm : public ov::op::Op {
public:
    OPENVINO_OP("CdpnCoordDenorm");

    CdpnCoordDenorm() = default;

    CdpnCoordDenorm(const ov::Output<ov::Node>& coord_maps,
                    const ov::Output<ov::Node>& obj_extents);

    void validate_and_infer_types() override;
    std::shared_ptr<ov::Node> clone_with_new_inputs(
        const ov::OutputVector& new_args) const override;
    bool visit_attributes(ov::AttributeVisitor& visitor) override;
    bool evaluate(ov::TensorVector& outputs,
                  const ov::TensorVector& inputs) const override;
    bool has_evaluate() const override;
};

}  // namespace CdpnExtension
