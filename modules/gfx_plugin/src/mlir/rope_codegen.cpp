// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "mlir/codegen_common.hpp"

#include <sstream>

namespace ov {
namespace gfx_plugin {

std::string generate_msl_for_rope(const RopeCodegenDesc& d, mlir::ModuleOp) {
    const std::string src_t = msl_type_from_element(d.input_type);
    const std::string trig_t = msl_type_from_element(d.cos_type);
    const std::string out_t = msl_type_from_element(d.output_type);
    const uint32_t head_size = d.head_size;
    const uint32_t rotary_dims = d.rotary_dims ? d.rotary_dims : head_size;
    const uint32_t half_rotary = rotary_dims / 2;
    const uint32_t cos_sin_dims = d.cos_sin_dims ? d.cos_sin_dims : rotary_dims;
    const uint32_t cos_sin_offset = (cos_sin_dims == half_rotary) ? 0 : half_rotary;
    const uint32_t heads = d.heads ? d.heads : 1;
    const uint32_t batch = d.batch ? d.batch : 1;
    const uint32_t cos_rank = d.cos_rank;
    const auto cd = d.cos_dims;
    const uint32_t dyn = d.cos_dynamic_mask;

    std::ostringstream ss;
    ss << "#include <metal_stdlib>\nusing namespace metal;\n\n";
    ss << "kernel void rope_kernel(device const " << src_t << "* src [[buffer(0)]],\n";
    ss << "                        device const " << trig_t << "* cos_tbl [[buffer(1)]],\n";
    ss << "                        device const " << trig_t << "* sin_tbl [[buffer(2)]],\n";
    if (d.has_position) {
        ss << "                        device const int* pos_tbl [[buffer(3)]],\n";
        ss << "                        device " << out_t << "* dst [[buffer(4)]],\n";
    } else {
        ss << "                        device " << out_t << "* dst [[buffer(3)]],\n";
    }
    ss << "                        uint gid [[thread_position_in_grid]],\n";
    ss << "                        uint total [[threads_per_grid]]) {\n";
    ss << "  if (gid >= total) return;\n";
    ss << "  constexpr uint head_size = " << head_size << "u;\n";
    ss << "  constexpr uint rotary_dims = " << rotary_dims << "u;\n";
    ss << "  constexpr uint half_rotary = " << half_rotary << "u;\n";
    ss << "  constexpr uint cos_sin_offset = " << cos_sin_offset << "u;\n";
    ss << "  constexpr uint heads = " << heads << "u;\n";
    ss << "  constexpr uint batch = " << batch << "u;\n";
    ss << "  uint d = gid % head_size;\n";
    ss << "  uint tmp = gid / head_size;\n";
    ss << "  uint seq = max(total / max(batch * heads * head_size, 1u), 1u);\n";
    ss << "  uint p = tmp % seq;\n";
    ss << "  tmp /= seq;\n";
    ss << "  uint h = tmp % heads;\n";
    ss << "  uint b = tmp / heads;\n";
    ss << "  uint cos_dim0 = " << cd[0] << "u;\n";
    ss << "  uint cos_dim1 = " << cd[1] << "u;\n";
    ss << "  uint cos_dim2 = " << cd[2] << "u;\n";
    ss << "  uint cos_dim3 = " << cd[3] << "u;\n";
    if (dyn & (1u << 0)) {
        ss << "  cos_dim0 = batch;\n";
    }
    if (dyn & (1u << 1)) {
        ss << "  cos_dim1 = heads;\n";
    }
    if (dyn & (1u << 2)) {
        ss << "  cos_dim2 = seq;\n";
    }
    if (dyn & (1u << 3)) {
        ss << "  cos_dim3 = head_size;\n";
    }
    if (d.has_position) {
        ss << "  uint cos_p = uint(max(pos_tbl[b * seq + p], 0));\n";
    } else {
        ss << "  uint cos_p = p;\n";
    }
    ss << "  if (d >= rotary_dims) { dst[gid] = (" << out_t << ")src[gid]; return; }\n";
    if (d.is_interleaved) {
        ss << "  uint pair_d = (d ^ 1u);\n";
        ss << "  uint trig_d = d >> 1u;\n";
        ss << "  float x0 = float(src[gid]);\n";
        ss << "  float x1 = float(src[(gid - d) + pair_d]);\n";
        ss << "  float sign = (d & 1u) ? 1.0f : -1.0f;\n";
    } else {
        ss << "  uint first_half = d < half_rotary;\n";
        ss << "  uint pair_d = first_half ? (d + half_rotary) : (d - half_rotary);\n";
        ss << "  uint trig_d = first_half ? d : (d - half_rotary + cos_sin_offset);\n";
        ss << "  float x0 = float(src[gid]);\n";
        ss << "  float x1 = float(src[(gid - d) + pair_d]);\n";
        ss << "  float sign = first_half ? -1.0f : 1.0f;\n";
    }
    ss << "  uint trig_index = 0u;\n";
    if (cos_rank == 2) {
        ss << "  trig_index = cos_p * cos_dim3 + trig_d;\n";
    } else if (cos_rank == 3) {
        ss << "  uint c1 = (cos_dim1 == 1u) ? 0u : h;\n";
        ss << "  uint c2 = (cos_dim2 == 1u) ? 0u : cos_p;\n";
        ss << "  trig_index = (c1 * cos_dim2 + c2) * cos_dim3 + trig_d;\n";
    } else {
        ss << "  uint c0 = (cos_dim0 == 1u) ? 0u : b;\n";
        ss << "  uint c1 = (cos_dim1 == 1u) ? 0u : h;\n";
        ss << "  uint c2 = (cos_dim2 == 1u) ? 0u : cos_p;\n";
        ss << "  trig_index = ((c0 * cos_dim1 + c1) * cos_dim2 + c2) * cos_dim3 + trig_d;\n";
    }
    ss << "  float c = float(cos_tbl[trig_index]);\n";
    ss << "  float s = float(sin_tbl[trig_index]);\n";
    ss << "  float y = c * x0 + sign * s * x1;\n";
    ss << "  dst[gid] = (" << out_t << ")y;\n";
    ss << "}\n";
    return ss.str();
}

}  // namespace gfx_plugin
}  // namespace ov
