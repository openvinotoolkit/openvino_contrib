// Voxelization CPU Reference Implementation for OpenVINO

#include "voxelize_op.hpp"
#include <cmath>
#include <algorithm>
#include <unordered_map>
#include <cstring>

using namespace BEVFusionExtension;

// Grid parameters (BEVFusion defaults)
static constexpr float VOXEL_X = 0.075f;
static constexpr float VOXEL_Y = 0.075f;
static constexpr float VOXEL_Z = 0.2f;
static constexpr float RANGE_X_MIN = -54.0f;
static constexpr float RANGE_Y_MIN = -54.0f;
static constexpr float RANGE_Z_MIN = -5.0f;
static constexpr float RANGE_X_MAX = 54.0f;
static constexpr float RANGE_Y_MAX = 54.0f;
static constexpr float RANGE_Z_MAX = 3.0f;


// ═══════════════════════════════════════════════════════════════════════════
// VoxelizeScatter
// ═══════════════════════════════════════════════════════════════════════════

VoxelizeScatter::VoxelizeScatter(
    const ov::Output<ov::Node>& points,
    const ov::Output<ov::Node>& num_points)
    : Op({points, num_points})
{
    constructor_validate_and_infer_types();
}

void VoxelizeScatter::validate_and_infer_types() {
    set_output_type(0, ov::element::i32,
                    ov::PartialShape{VOXEL_WORKSPACE_SIZE});
}

std::shared_ptr<ov::Node> VoxelizeScatter::clone_with_new_inputs(
    const ov::OutputVector& new_args) const
{
    OPENVINO_ASSERT(new_args.size() == 2, "VoxelizeScatter requires 2 inputs");
    return std::make_shared<VoxelizeScatter>(new_args[0], new_args[1]);
}

bool VoxelizeScatter::visit_attributes(ov::AttributeVisitor&) {
    return true;
}

bool VoxelizeScatter::evaluate(ov::TensorVector& outputs,
                               const ov::TensorVector& inputs) const
{
    const float* pts = inputs[0].data<float>();
    const int32_t* np_ptr = inputs[1].data<int32_t>();
    int num_pts = np_ptr[0];

    outputs[0].set_shape({static_cast<size_t>(VOXEL_WORKSPACE_SIZE)});
    int32_t* ws = outputs[0].data<int32_t>();
    std::memset(ws, 0, VOXEL_WORKSPACE_SIZE * sizeof(int32_t));

    // CPU hash-table voxelization (inline layout matching GPU kernel)
    // Hash table entries are at ws[HT_OFFSET + slot * ENTRY_STRIDE]:
    //   [key_stored, accum0..4, count, batch, x, y, z]
    std::unordered_map<int64_t, int> key_to_slot;
    int voxel_num = 0;

    for (int i = 0; i < num_pts && i < VOXEL_MAX_POINTS; ++i) {
        float px = pts[i * VOXEL_NUM_FEATURES + 0];
        float py = pts[i * VOXEL_NUM_FEATURES + 1];
        float pz = pts[i * VOXEL_NUM_FEATURES + 2];

        int ix = static_cast<int>(std::floor((px - RANGE_X_MIN) / VOXEL_X));
        int iy = static_cast<int>(std::floor((py - RANGE_Y_MIN) / VOXEL_Y));
        int iz = static_cast<int>(std::floor((pz - RANGE_Z_MIN) / VOXEL_Z));

        if (ix < 0 || ix >= VOXEL_GRID_X ||
            iy < 0 || iy >= VOXEL_GRID_Y ||
            iz < 0 || iz >= VOXEL_GRID_Z)
            continue;

        int64_t key = static_cast<int64_t>(ix) * VOXEL_GRID_Y * VOXEL_GRID_Z
                    + static_cast<int64_t>(iy) * VOXEL_GRID_Z + iz;

        int slot;
        auto it = key_to_slot.find(key);
        if (it == key_to_slot.end()) {
            if (voxel_num >= VOXEL_MAX_VOXELS) continue;

            // Find hash slot via open-addressing (mimic GPU)
            slot = static_cast<int>(key & (VOXEL_HASH_SIZE - 1));
            while (ws[VOXEL_HT_OFFSET + slot * VOXEL_ENTRY_STRIDE] != 0) {
                slot = (slot + 1) & (VOXEL_HASH_SIZE - 1);
            }
            key_to_slot[key] = slot;
            voxel_num++;

            int entry_base = VOXEL_HT_OFFSET + slot * VOXEL_ENTRY_STRIDE;
            ws[entry_base + 0]  = static_cast<int32_t>(key + 1);  // key_stored
            ws[entry_base + 7]  = 0;   // batch
            ws[entry_base + 8]  = ix;  // coord_x
            ws[entry_base + 9]  = iy;  // coord_y
            ws[entry_base + 10] = iz;  // coord_z
        } else {
            slot = it->second;
        }

        int entry_base = VOXEL_HT_OFFSET + slot * VOXEL_ENTRY_STRIDE;
        int pt_count = ws[entry_base + 6];

        if (pt_count < VOXEL_MAX_PTS_PER) {
            for (int f = 0; f < VOXEL_NUM_FEATURES; ++f) {
                int ival = static_cast<int>(pts[i * VOXEL_NUM_FEATURES + f]
                                            * VOXEL_FP_SCALE);
                ws[entry_base + 1 + f] += ival;
            }
            ws[entry_base + 6] = pt_count + 1;
        }
    }

    ws[0] = voxel_num;
    return true;
}

bool VoxelizeScatter::has_evaluate() const { return true; }


// ═══════════════════════════════════════════════════════════════════════════
// VoxelizeMean
// ═══════════════════════════════════════════════════════════════════════════

VoxelizeMean::VoxelizeMean(const ov::Output<ov::Node>& workspace)
    : Op({workspace})
{
    constructor_validate_and_infer_types();
}

void VoxelizeMean::validate_and_infer_types() {
    set_output_type(0, ov::element::f32,
                    ov::PartialShape{VOXEL_MAX_VOXELS + 1, 9});
}

std::shared_ptr<ov::Node> VoxelizeMean::clone_with_new_inputs(
    const ov::OutputVector& new_args) const
{
    OPENVINO_ASSERT(new_args.size() == 1, "VoxelizeMean requires 1 input");
    return std::make_shared<VoxelizeMean>(new_args[0]);
}

bool VoxelizeMean::visit_attributes(ov::AttributeVisitor&) {
    return true;
}

bool VoxelizeMean::evaluate(ov::TensorVector& outputs,
                            const ov::TensorVector& inputs) const
{
    const int32_t* ws = inputs[0].data<int32_t>();
    int num_voxels = ws[0];

    outputs[0].set_shape({static_cast<size_t>(VOXEL_MAX_VOXELS + 1), 9});
    float* result = outputs[0].data<float>();
    std::memset(result, 0, (VOXEL_MAX_VOXELS + 1) * 9 * sizeof(float));

    // Row 0: metadata
    result[0] = static_cast<float>(num_voxels);

    // Scan hash table entries to extract voxels (matching GPU kernel logic)
    int compact_row = 0;
    for (int slot = 0; slot < VOXEL_HASH_SIZE && compact_row < num_voxels; ++slot) {
        int entry_base = VOXEL_HT_OFFSET + slot * VOXEL_ENTRY_STRIDE;
        int key_stored = ws[entry_base + 0];

        if (key_stored <= 0) continue;   // empty or poisoned

        int raw_count = ws[entry_base + 6];
        if (raw_count <= 0) continue;

        // Match reference: divide by total count (not capped at MAX_PTS_PER).
        // Features only accumulate for first MAX_PTS_PER points,
        // but count includes all points assigned to the voxel.
        float inv = 1.0f / (static_cast<float>(VOXEL_FP_SCALE) * raw_count);
        int out_base = (compact_row + 1) * 9;

        for (int f = 0; f < VOXEL_NUM_FEATURES; ++f) {
            result[out_base + f] = static_cast<float>(ws[entry_base + 1 + f]) * inv;
        }

        result[out_base + 5] = static_cast<float>(ws[entry_base + 7]);   // batch
        result[out_base + 6] = static_cast<float>(ws[entry_base + 8]);   // coord_x
        result[out_base + 7] = static_cast<float>(ws[entry_base + 9]);   // coord_y
        result[out_base + 8] = static_cast<float>(ws[entry_base + 10]);  // coord_z

        compact_row++;
    }

    return true;
}

bool VoxelizeMean::has_evaluate() const { return true; }
