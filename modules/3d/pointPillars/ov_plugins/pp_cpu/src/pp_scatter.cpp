// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include "pp_scatter.hpp"

#include <algorithm>
#include <cstring>
#include <vector>

namespace PpExtension {
namespace {

enum InputPort : size_t {
    kPillarFeaturesInput = 0,
    kPillarCoordinatesInput,
    kPillarCountInput,
    kInputCount,
};

constexpr size_t kOutputPort = 0;
constexpr size_t kOutputCount = 1;
constexpr size_t kScalarValueIndex = 0;

// Number of feature channels carried by the scatter input.
int64_t feature_channel_count(const ov::PartialShape& shape) {
    if (shape.rank().is_static() && shape.rank().get_length() >= 2) {
        const auto& last = shape[shape.rank().get_length() - 1];
        if (last.is_static()) return last.get_length();
    }
    return PP_PFN_CHANNELS;
}

// Coordinates and counts may use int32 or the f32 metadata representation.
int32_t read_int32(const ov::Tensor& tensor, int64_t index) {
    return tensor.get_element_type() == ov::element::i32
               ? tensor.data<const int32_t>()[index]
               : static_cast<int32_t>(tensor.data<const float>()[index]);
}

}  // namespace

PPScatterBEV::PPScatterBEV(const ov::Output<ov::Node>& pillar_features,
                           const ov::Output<ov::Node>& pillar_coordinates,
                           const ov::Output<ov::Node>& pillar_count)
    : Op({pillar_features, pillar_coordinates, pillar_count}) {
    constructor_validate_and_infer_types();
}

void PPScatterBEV::validate_and_infer_types() {
    const int64_t feature_channels = feature_channel_count(get_input_partial_shape(kPillarFeaturesInput));
    set_output_type(kOutputPort, get_input_element_type(kPillarFeaturesInput),
                    ov::PartialShape{1, feature_channels, PP_GRID_Y, PP_GRID_X});
}

std::shared_ptr<ov::Node> PPScatterBEV::clone_with_new_inputs(const ov::OutputVector& new_args) const {
    OPENVINO_ASSERT(new_args.size() == 3, "PPScatterBEV expects 3 inputs");
    return std::make_shared<PPScatterBEV>(new_args[0], new_args[1], new_args[2]);
}

bool PPScatterBEV::visit_attributes(ov::AttributeVisitor&) {
    return true;
}

bool PPScatterBEV::has_evaluate() const {
    return true;
}

bool PPScatterBEV::evaluate(ov::TensorVector& outputs, const ov::TensorVector& inputs) const {
    OPENVINO_ASSERT(inputs.size() == kInputCount && outputs.size() == kOutputCount,
                    "PPScatterBEV expects 3 inputs and 1 output");

    const auto& pillar_features_shape = inputs[kPillarFeaturesInput].get_shape();
    const int64_t feature_channels = pillar_features_shape.empty()
                                 ? PP_PFN_CHANNELS
                                 : static_cast<int64_t>(pillar_features_shape[pillar_features_shape.size() - 1]);
    outputs[kOutputPort].set_shape({1, static_cast<size_t>(feature_channels), static_cast<size_t>(PP_GRID_Y),
                                    static_cast<size_t>(PP_GRID_X)});

    const int64_t max_pillar_count =
        pillar_features_shape.empty() ? 0 : static_cast<int64_t>(pillar_features_shape[0]);
    int32_t pillar_count = read_int32(inputs[kPillarCountInput], kScalarValueIndex);
    pillar_count = std::clamp<int32_t>(
        pillar_count, 0, static_cast<int32_t>(std::min<int64_t>(max_pillar_count, PP_MAX_VOXELS)));

    // Build the cell-to-pillar lookup before writing the output.
    std::vector<int32_t> cell_to_pillar(static_cast<size_t>(PP_BEV_CELL_COUNT), -1);
    for (int32_t pillar = 0; pillar < pillar_count; ++pillar) {
        const int32_t y = read_int32(inputs[kPillarCoordinatesInput],
                         static_cast<int64_t>(pillar) * PP_COORD_VALUES + PP_COORD_Y);
        const int32_t x = read_int32(inputs[kPillarCoordinatesInput],
                         static_cast<int64_t>(pillar) * PP_COORD_VALUES + PP_COORD_X);
        if (y < 0 || y >= PP_GRID_Y || x < 0 || x >= PP_GRID_X) continue;
        cell_to_pillar[static_cast<size_t>(y) * PP_GRID_X + x] = pillar;
    }

    // Copy feature elements without changing their representation.
    const size_t element_size = inputs[kPillarFeaturesInput].get_element_type().size();
    const auto* pillar_features = static_cast<const uint8_t*>(inputs[kPillarFeaturesInput].data());
    auto* bev_output = static_cast<uint8_t*>(outputs[kOutputPort].data());

    for (int64_t channel = 0; channel < feature_channels; ++channel) {
        uint8_t* plane = bev_output + static_cast<size_t>(channel * PP_BEV_CELL_COUNT) * element_size;
        for (int64_t cell = 0; cell < PP_BEV_CELL_COUNT; ++cell) {
            const int32_t pillar = cell_to_pillar[static_cast<size_t>(cell)];
            uint8_t* destination = plane + static_cast<size_t>(cell) * element_size;
            if (pillar < 0) {
                std::memset(destination, 0, element_size);
            } else {
                std::memcpy(destination,
                            pillar_features + (static_cast<size_t>(pillar) * feature_channels + channel) * element_size,
                            element_size);
            }
        }
    }
    return true;
}

}  // namespace PpExtension
