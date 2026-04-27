// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "mlir/codegen_common.hpp"

#include <sstream>

#include "openvino/core/except.hpp"

namespace ov {
namespace gfx_plugin {

std::string generate_msl_for_rms(const RmsCodegenDesc& d, mlir::ModuleOp module) {
    OPENVINO_ASSERT(d.hidden > 0, "RMS hidden dimension must be positive");
    const uint32_t threads = d.reduction_threads > 1 ? d.reduction_threads : 1u;
    std::string input_t = msl_type_from_element(d.input_type == ov::element::dynamic ? d.element_type : d.input_type);
    std::string gamma_t = msl_type_from_element(d.gamma_type == ov::element::dynamic ? d.element_type : d.gamma_type);
    std::string output_t =
        msl_type_from_element(d.output_type == ov::element::dynamic ? d.element_type : d.output_type);
    if (module) {
        if (auto func = get_entry_func(module)) {
            auto ft = func.getFunctionType();
            if (ft.getNumInputs() >= 1) {
                input_t = msl_type_from_mlir(ft.getInput(0));
            }
            if (ft.getNumInputs() >= 2) {
                gamma_t = msl_type_from_mlir(ft.getInput(1));
            }
            if (ft.getNumResults() >= 1) {
                output_t = msl_type_from_mlir(ft.getResult(0));
            }
        }
    }

    std::ostringstream ss;
    ss << "#include <metal_stdlib>\n";
    ss << "using namespace metal;\n";
    ss << "using input_t = " << input_t << ";\n";
    ss << "using gamma_t = " << gamma_t << ";\n";
    ss << "using output_t = " << output_t << ";\n";
    ss << "kernel void rms_kernel(device const input_t* X [[buffer(0)]],\n";
    ss << "                       device const gamma_t* G [[buffer(1)]],\n";
    ss << "                       device output_t* O [[buffer(2)]],\n";
    ss << "                       uint gid [[thread_position_in_grid]],\n";
    ss << "                       uint lane [[thread_index_in_threadgroup]]) {\n";
    if (threads == 1) {
        ss << "    const uint hidden = " << d.hidden << "u;\n";
        ss << "    const uint row = gid / hidden;\n";
        ss << "    const uint col = gid - row * hidden;\n";
        ss << "    const uint base = row * hidden;\n";
        ss << "    float sum = 0.0f;\n";
        ss << "    for (uint k = 0; k < hidden; ++k) {\n";
        ss << "        const float x = static_cast<float>(X[base + k]);\n";
        ss << "        sum += x * x;\n";
        ss << "    }\n";
        ss << "    const uint gidx = " << (d.gamma_size == 1 ? "0u" : "col") << ";\n";
        ss << "    const float inv = rsqrt(sum / static_cast<float>(hidden) + " << d.epsilon << "f);\n";
        ss << "    O[gid] = static_cast<output_t>(static_cast<float>(X[gid]) * inv * static_cast<float>(G[gidx]));\n";
    } else {
        ss << "    const uint threads = " << threads << "u;\n";
        ss << "    const uint hidden = " << d.hidden << "u;\n";
        ss << "    const uint row = gid / threads;\n";
        ss << "    const uint base = row * hidden;\n";
        ss << "    threadgroup float partial[" << threads << "];\n";
        ss << "    float sum = 0.0f;\n";
        ss << "    for (uint k = lane; k < hidden; k += threads) {\n";
        ss << "        const float x = static_cast<float>(X[base + k]);\n";
        ss << "        sum += x * x;\n";
        ss << "    }\n";
        ss << "    partial[lane] = sum;\n";
        ss << "    threadgroup_barrier(mem_flags::mem_threadgroup);\n";
        for (uint32_t stride = threads / 2; stride > 0; stride >>= 1) {
            ss << "    if (lane < " << stride << "u) partial[lane] += partial[lane + " << stride << "u];\n";
            ss << "    threadgroup_barrier(mem_flags::mem_threadgroup);\n";
        }
        ss << "    const float inv = rsqrt(partial[0] / static_cast<float>(hidden) + " << d.epsilon << "f);\n";
        ss << "    for (uint k = lane; k < hidden; k += threads) {\n";
        ss << "        const uint gidx = " << (d.gamma_size == 1 ? "0u" : "k") << ";\n";
        ss << "        O[base + k] = static_cast<output_t>(static_cast<float>(X[base + k]) * inv * static_cast<float>(G[gidx]));\n";
        ss << "    }\n";
    }
    ss << "}\n";
    return ss.str();
}

}  // namespace gfx_plugin
}  // namespace ov
