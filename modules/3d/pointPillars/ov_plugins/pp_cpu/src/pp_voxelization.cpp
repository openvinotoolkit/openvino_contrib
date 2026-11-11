// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include "pp_voxelization.hpp"

#include <algorithm>
#include <cmath>
#include <cstring>

namespace PpExtension {
namespace {

// Convert a float to IEEE 754 half-precision bits.
uint16_t float_to_half_bits(float value) {
    uint32_t bits = 0;
    std::memcpy(&bits, &value, sizeof(bits));
    const uint32_t sign = (bits >> 16) & 0x8000u;
    int32_t exp = static_cast<int32_t>((bits >> 23) & 0xffu) - 127 + 15;
    uint32_t mant = bits & 0x7fffffu;

    if (exp <= 0) {
        if (exp < -10) return static_cast<uint16_t>(sign);
        mant |= 0x800000u;
        const int shift = 14 - exp;
        uint32_t half_mant = mant >> shift;
        const uint32_t remainder = mant & ((1u << shift) - 1u);
        const uint32_t halfway = 1u << (shift - 1);
        if (remainder > halfway || (remainder == halfway && (half_mant & 1u))) half_mant++;
        return static_cast<uint16_t>(sign | half_mant);
    }

    if (exp >= 31) {
        if (mant == 0) return static_cast<uint16_t>(sign | 0x7c00u);
        return static_cast<uint16_t>(sign | 0x7c00u | (mant >> 13) | 1u);
    }

    uint32_t half_mant = mant >> 13;
    const uint32_t remainder = mant & 0x1fffu;
    if (remainder > 0x1000u || (remainder == 0x1000u && (half_mant & 1u))) {
        half_mant++;
        if (half_mant == 0x400u) {
            half_mant = 0;
            exp++;
            if (exp >= 31) return static_cast<uint16_t>(sign | 0x7c00u);
        }
    }
    return static_cast<uint16_t>(sign | (static_cast<uint32_t>(exp) << 10) | half_mant);
}

// Compute grid coordinates for a point. Return false when it is outside the scene.
bool point_grid(float x, float y, float z, int& ix, int& iy, int& iz) {
    if (x < PP_RANGE_MIN_X || x >= PP_RANGE_MAX_X || y < PP_RANGE_MIN_Y || y >= PP_RANGE_MAX_Y ||
        z < PP_RANGE_MIN_Z || z >= PP_RANGE_MAX_Z)
        return false;

    ix = static_cast<int>(std::floor((x - PP_RANGE_MIN_X) / PP_VOXEL_SIZE_X));
    iy = static_cast<int>(std::floor((y - PP_RANGE_MIN_Y) / PP_VOXEL_SIZE_Y));
    iz = static_cast<int>(std::floor((z - PP_RANGE_MIN_Z) / PP_VOXEL_SIZE_Z));
    return ix >= 0 && ix < PP_GRID_X && iy >= 0 && iy < PP_GRID_Y && iz >= 0 && iz < PP_GRID_Z;
}

// Flatten the horizontal grid coordinates into a cell index.
int cell_index(int ix, int iy) {
    return iy * static_cast<int>(PP_GRID_X) + ix;
}

void write_feature(uint16_t* features, int pillar, int point, int channel, float value) {
    const int64_t index = (static_cast<int64_t>(pillar) * PP_MAX_POINTS_PER_VOXEL + point) *
                              PP_NUM_OUTPUT_FEATURES +
                          channel;
    features[index] = float_to_half_bits(value);
}

// Compute the center coordinate for a grid cell.
float voxel_center(float voxel_size, int grid, float min_range) {
    volatile float product = static_cast<float>(grid) * voxel_size;
    volatile float half_step = voxel_size / 2.0f;
    volatile float center = product + half_step;
    center = center + min_range;
    return center;
}

}  // namespace

// ---------------------------------------------------------------------------
// Stage 1: scatter points and compact pillars.
// ---------------------------------------------------------------------------

PPVoxelizationScatter::PPVoxelizationScatter(const ov::Output<ov::Node>& points,
                                             const ov::Output<ov::Node>& point_count)
    : Op({points, point_count}) {
    constructor_validate_and_infer_types();
}

void PPVoxelizationScatter::validate_and_infer_types() {
    set_output_type(0, ov::element::f32, ov::PartialShape{1, 1, 1, PP_META_SIZE});
}

std::shared_ptr<ov::Node> PPVoxelizationScatter::clone_with_new_inputs(const ov::OutputVector& new_args) const {
    OPENVINO_ASSERT(new_args.size() == 2, "PPVoxelizationScatter expects 2 inputs");
    return std::make_shared<PPVoxelizationScatter>(new_args[0], new_args[1]);
}

bool PPVoxelizationScatter::visit_attributes(ov::AttributeVisitor&) {
    return true;
}

bool PPVoxelizationScatter::has_evaluate() const {
    return true;
}

bool PPVoxelizationScatter::evaluate(ov::TensorVector& outputs, const ov::TensorVector& inputs) const {
    OPENVINO_ASSERT(inputs.size() == 2 && outputs.size() == 1, "PPVoxelizationScatter expects 2 inputs and 1 output");
    outputs[0].set_shape({1, 1, 1, PP_META_SIZE});
    auto* metadata = reinterpret_cast<int32_t*>(outputs[0].data<float>());
    const float* points = inputs[0].data<const float>();
    const auto input_shape = inputs[0].get_shape();
    int point_count = inputs[1].data<const int32_t>()[0];
    point_count = std::max(0, std::min<int>(point_count, static_cast<int>(input_shape[0])));
    point_count = std::min<int>(point_count, static_cast<int>(PP_MAX_POINTS));

    // Clear the lookup, bucket points by cell, and assign one pillar per occupied cell.
    std::fill(metadata + PP_META_G2P, metadata + PP_META_G2P + PP_GRID_SIZE, 0);
    int pillar_count = 0;
    for (int point_id = 0; point_id < point_count; ++point_id) {
        const float* point = points + point_id * PP_NUM_POINT_FEATURES;
        int ix = 0, iy = 0, iz = 0;
        if (!point_grid(point[0], point[1], point[2], ix, iy, iz)) continue;
        const int grid = cell_index(ix, iy);

        int pillar = metadata[PP_META_G2P + grid] - 1;
        if (pillar < 0) {
            pillar = pillar_count++;
            metadata[PP_META_G2P + grid] = pillar + 1;
            if (pillar < PP_MAX_VOXELS) {
              const int64_t coord = PP_META_COORD + pillar * PP_COORD_VALUES;
              metadata[PP_META_COUNT + pillar] = 0;
              metadata[coord + PP_COORD_BATCH] = 0;
              metadata[coord + PP_COORD_Z] = iz;
              metadata[coord + PP_COORD_Y] = iy;
              metadata[coord + PP_COORD_X] = ix;
            }
        }
        if (pillar >= PP_MAX_VOXELS) continue;

        const int slot = metadata[PP_META_COUNT + pillar]++;
        if (slot < PP_MAX_POINTS_PER_VOXEL)
            metadata[PP_META_RAW + static_cast<int64_t>(pillar) * PP_META_RAW_STRIDE + slot] = point_id;
    }
    metadata[PP_META_COUNTER] = pillar_count;
    return true;
}

// ---------------------------------------------------------------------------
// Stage 2: generate pillar features.
// ---------------------------------------------------------------------------

PPVoxelizationFinalize::PPVoxelizationFinalize(const ov::Output<ov::Node>& metadata,
                                               const ov::Output<ov::Node>& points)
    : Op({metadata, points}) {
    constructor_validate_and_infer_types();
}

void PPVoxelizationFinalize::validate_and_infer_types() {
    set_output_type(0, ov::element::f16,
                    ov::PartialShape{PP_MAX_VOXELS, PP_MAX_POINTS_PER_VOXEL, PP_NUM_OUTPUT_FEATURES});
}

std::shared_ptr<ov::Node> PPVoxelizationFinalize::clone_with_new_inputs(const ov::OutputVector& new_args) const {
    OPENVINO_ASSERT(new_args.size() == 2, "PPVoxelizationFinalize expects 2 inputs");
    return std::make_shared<PPVoxelizationFinalize>(new_args[0], new_args[1]);
}

bool PPVoxelizationFinalize::visit_attributes(ov::AttributeVisitor&) {
    return true;
}

bool PPVoxelizationFinalize::has_evaluate() const {
    return true;
}

bool PPVoxelizationFinalize::evaluate(ov::TensorVector& outputs, const ov::TensorVector& inputs) const {
    OPENVINO_ASSERT(inputs.size() == 2 && outputs.size() == 1, "PPVoxelizationFinalize expects 2 inputs and 1 output");
    outputs[0].set_shape({PP_MAX_VOXELS, PP_MAX_POINTS_PER_VOXEL, PP_NUM_OUTPUT_FEATURES});
    const auto* metadata = reinterpret_cast<const int32_t*>(inputs[0].data<const float>());
    const float* points = inputs[1].data<const float>();
    auto* features = reinterpret_cast<uint16_t*>(outputs[0].data());

    const int num_pillars = std::min<int32_t>(metadata[PP_META_COUNTER], PP_MAX_VOXELS);
    std::fill(features,
              features + static_cast<int64_t>(num_pillars) * PP_MAX_POINTS_PER_VOXEL * PP_NUM_OUTPUT_FEATURES, 0);

    for (int pillar = 0; pillar < num_pillars; ++pillar) {
        int pillar_point_count = metadata[PP_META_COUNT + pillar];
        if (pillar_point_count <= 0) continue;
        pillar_point_count = std::min<int>(pillar_point_count, static_cast<int>(PP_MAX_POINTS_PER_VOXEL));

        const int64_t raw_point_index_offset = PP_META_RAW + static_cast<int64_t>(pillar) * PP_META_RAW_STRIDE;
        float mean_x = 0.0f;
        float mean_y = 0.0f;
        float mean_z = 0.0f;
        for (int point_index = 0; point_index < pillar_point_count; ++point_index) {
            const int64_t point_offset =
                static_cast<int64_t>(metadata[raw_point_index_offset + point_index]) * PP_NUM_POINT_FEATURES;
            mean_x += points[point_offset + 0];
            mean_y += points[point_offset + 1];
            mean_z += points[point_offset + 2];
        }
        mean_x /= static_cast<float>(pillar_point_count);
        mean_y /= static_cast<float>(pillar_point_count);
        mean_z /= static_cast<float>(pillar_point_count);

        const int grid_z = metadata[PP_META_COORD + pillar * PP_COORD_VALUES + PP_COORD_Z];
        const int grid_y = metadata[PP_META_COORD + pillar * PP_COORD_VALUES + PP_COORD_Y];
        const int grid_x = metadata[PP_META_COORD + pillar * PP_COORD_VALUES + PP_COORD_X];
        const float center_x = voxel_center(PP_VOXEL_SIZE_X, grid_x, PP_RANGE_MIN_X);
        const float center_y = voxel_center(PP_VOXEL_SIZE_Y, grid_y, PP_RANGE_MIN_Y);
        const float center_z = voxel_center(PP_VOXEL_SIZE_Z, grid_z, PP_RANGE_MIN_Z);

        for (int point_index = 0; point_index < pillar_point_count; ++point_index) {
            const int64_t point_offset =
                static_cast<int64_t>(metadata[raw_point_index_offset + point_index]) * PP_NUM_POINT_FEATURES;
            const float x = points[point_offset + 0];
            const float y = points[point_offset + 1];
            const float z = points[point_offset + 2];
            const float intensity = points[point_offset + 3];
            write_feature(features, pillar, point_index, 0, x);
            write_feature(features, pillar, point_index, 1, y);
            write_feature(features, pillar, point_index, 2, z);
            write_feature(features, pillar, point_index, 3, intensity);
            write_feature(features, pillar, point_index, 4, x - mean_x);
            write_feature(features, pillar, point_index, 5, y - mean_y);
            write_feature(features, pillar, point_index, 6, z - mean_z);
            write_feature(features, pillar, point_index, 7, x - center_x);
            write_feature(features, pillar, point_index, 8, y - center_y);
            write_feature(features, pillar, point_index, 9, z - center_z);
        }
    }
    return true;
}

// ---------------------------------------------------------------------------
// Stage 3: emit pillar coordinates.
// ---------------------------------------------------------------------------

PPVoxelizationCoords::PPVoxelizationCoords(const ov::Output<ov::Node>& metadata)
    : Op({metadata}) {
    constructor_validate_and_infer_types();
}

void PPVoxelizationCoords::validate_and_infer_types() {
    set_output_type(0, ov::element::f32, ov::PartialShape{PP_MAX_VOXELS, PP_COORD_VALUES});
}

std::shared_ptr<ov::Node> PPVoxelizationCoords::clone_with_new_inputs(const ov::OutputVector& new_args) const {
    OPENVINO_ASSERT(new_args.size() == 1, "PPVoxelizationCoords expects 1 input");
    return std::make_shared<PPVoxelizationCoords>(new_args[0]);
}

bool PPVoxelizationCoords::visit_attributes(ov::AttributeVisitor&) {
    return true;
}

bool PPVoxelizationCoords::has_evaluate() const {
    return true;
}

bool PPVoxelizationCoords::evaluate(ov::TensorVector& outputs, const ov::TensorVector& inputs) const {
    OPENVINO_ASSERT(inputs.size() == 1 && outputs.size() == 1, "PPVoxelizationCoords expects 1 input and 1 output");
    outputs[0].set_shape({PP_MAX_VOXELS, PP_COORD_VALUES});
    const auto* metadata = reinterpret_cast<const int32_t*>(inputs[0].data<const float>());
    auto* coords = outputs[0].data<float>();

    const int num_pillars = std::min<int32_t>(metadata[PP_META_COUNTER], PP_MAX_VOXELS);
    std::fill(coords, coords + PP_MAX_VOXELS * PP_COORD_VALUES, 0.0f);
    for (int pillar = 0; pillar < num_pillars; ++pillar) {
        for (int coordinate_index = 0; coordinate_index < PP_COORD_VALUES; ++coordinate_index)
            coords[pillar * PP_COORD_VALUES + coordinate_index] =
                static_cast<float>(metadata[PP_META_COORD + pillar * PP_COORD_VALUES + coordinate_index]);
    }
    return true;
}

// ---------------------------------------------------------------------------
// Stage 4: emit the pillar count.
// ---------------------------------------------------------------------------

PPVoxelizationParams::PPVoxelizationParams(const ov::Output<ov::Node>& metadata)
    : Op({metadata}) {
    constructor_validate_and_infer_types();
}

void PPVoxelizationParams::validate_and_infer_types() {
    set_output_type(0, ov::element::f32, ov::PartialShape{1});
}

std::shared_ptr<ov::Node> PPVoxelizationParams::clone_with_new_inputs(const ov::OutputVector& new_args) const {
    OPENVINO_ASSERT(new_args.size() == 1, "PPVoxelizationParams expects 1 input");
    return std::make_shared<PPVoxelizationParams>(new_args[0]);
}

bool PPVoxelizationParams::visit_attributes(ov::AttributeVisitor&) {
    return true;
}

bool PPVoxelizationParams::has_evaluate() const {
    return true;
}

bool PPVoxelizationParams::evaluate(ov::TensorVector& outputs, const ov::TensorVector& inputs) const {
    OPENVINO_ASSERT(inputs.size() == 1 && outputs.size() == 1, "PPVoxelizationParams expects 1 input and 1 output");
    outputs[0].set_shape({1});
    const auto* metadata = reinterpret_cast<const int32_t*>(inputs[0].data<const float>());
    outputs[0].data<float>()[0] = static_cast<float>(std::min<int32_t>(metadata[PP_META_COUNTER], PP_MAX_VOXELS));
    return true;
}

// ---------------------------------------------------------------------------
// Stage 5: emit the cell-to-pillar lookup.
// ---------------------------------------------------------------------------

PPVoxelizationCellPillar::PPVoxelizationCellPillar(const ov::Output<ov::Node>& metadata)
    : Op({metadata}) {
    constructor_validate_and_infer_types();
}

void PPVoxelizationCellPillar::validate_and_infer_types() {
    set_output_type(0, ov::element::i32, ov::PartialShape{PP_GRID_SIZE});
}

std::shared_ptr<ov::Node> PPVoxelizationCellPillar::clone_with_new_inputs(const ov::OutputVector& new_args) const {
    OPENVINO_ASSERT(new_args.size() == 1, "PPVoxelizationCellPillar expects 1 input");
    return std::make_shared<PPVoxelizationCellPillar>(new_args[0]);
}

bool PPVoxelizationCellPillar::visit_attributes(ov::AttributeVisitor&) {
    return true;
}

bool PPVoxelizationCellPillar::has_evaluate() const {
    return true;
}

bool PPVoxelizationCellPillar::evaluate(ov::TensorVector& outputs, const ov::TensorVector& inputs) const {
    OPENVINO_ASSERT(inputs.size() == 1 && outputs.size() == 1,
                    "PPVoxelizationCellPillar expects 1 input and 1 output");
    outputs[0].set_shape({PP_GRID_SIZE});
    const auto* metadata = reinterpret_cast<const int32_t*>(inputs[0].data<const float>());
    auto* table = outputs[0].data<int32_t>();

    // Empty cells and pillars beyond the configured limit are emitted as zero.
    const int32_t num_pillars = std::min<int32_t>(metadata[PP_META_COUNTER], PP_MAX_VOXELS);
    for (int64_t cell = 0; cell < PP_GRID_SIZE; ++cell) {
        const int32_t slot = metadata[PP_META_G2P + cell];
        table[cell] = (slot > 0 && slot <= num_pillars) ? slot : 0;
    }
    return true;
}

}  // namespace PpExtension
