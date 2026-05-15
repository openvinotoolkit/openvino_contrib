#include "cdpn_coord_denorm.hpp"

#include <algorithm>
#include <cmath>

namespace CdpnExtension {

CdpnCoordDenorm::CdpnCoordDenorm(const ov::Output<ov::Node>& coord_maps,
                                  const ov::Output<ov::Node>& obj_extents)
    : Op({coord_maps, obj_extents}) {
    constructor_validate_and_infer_types();
}

void CdpnCoordDenorm::validate_and_infer_types() {
    // Output same shape as coord_maps: [1, 4, H, W]
    set_output_type(0, ov::element::f32, get_input_partial_shape(0));
}

std::shared_ptr<ov::Node> CdpnCoordDenorm::clone_with_new_inputs(
    const ov::OutputVector& new_args) const {
    OPENVINO_ASSERT(new_args.size() == 2, "CdpnCoordDenorm expects 2 inputs");
    return std::make_shared<CdpnCoordDenorm>(new_args[0], new_args[1]);
}

bool CdpnCoordDenorm::visit_attributes(ov::AttributeVisitor& /*visitor*/) {
    return true;
}

bool CdpnCoordDenorm::evaluate(ov::TensorVector& outputs,
                               const ov::TensorVector& inputs) const {
    // coord_maps: [1, 4, R, R] where R = spatial resolution (64)
    const auto shape = inputs[0].get_shape();
    const int R = static_cast<int>(shape[2]);
    const int R2 = R * R;

    const auto* coord = inputs[0].data<float>();
    const auto* ext = inputs[1].data<float>();  // [1,1,1,3] -> flat [3]

    outputs[0].set_shape(shape);
    auto* out = outputs[0].data<float>();

    const float abs_x = ext[0];
    const float abs_y = ext[1];
    const float abs_z = ext[2];

    // Channels 0-2: multiply by extents (denormalise)
    for (int i = 0; i < R2; ++i) out[0 * R2 + i] = coord[0 * R2 + i] * abs_x;
    for (int i = 0; i < R2; ++i) out[1 * R2 + i] = coord[1 * R2 + i] * abs_y;
    for (int i = 0; i < R2; ++i) out[2 * R2 + i] = coord[2 * R2 + i] * abs_z;

    // Channel 3: min-max normalise confidence
    const float* raw_conf = coord + 3 * R2;
    float* conf_out = out + 3 * R2;

    float cmin = raw_conf[0], cmax = raw_conf[0];
    for (int i = 1; i < R2; ++i) {
        cmin = std::min(cmin, raw_conf[i]);
        cmax = std::max(cmax, raw_conf[i]);
    }
    float range = cmax - cmin;
    if (range < 1e-8f) range = 1.0f;
    const float inv_range = 1.0f / range;

    for (int i = 0; i < R2; ++i) {
        conf_out[i] = (raw_conf[i] - cmin) * inv_range;
    }

    return true;
}

bool CdpnCoordDenorm::has_evaluate() const {
    return true;
}

}  // namespace CdpnExtension
