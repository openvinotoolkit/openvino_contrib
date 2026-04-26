// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "mlir/codegen_common.hpp"

#include <sstream>

namespace ov {
namespace gfx_plugin {

std::string generate_msl_for_range(const RangeCodegenDesc& d, mlir::ModuleOp module) {
    auto type_from_element = [](const ov::element::Type& type) {
        if (type == ov::element::f16) return std::string("half");
        if (type == ov::element::f32) return std::string("float");
        if (type == ov::element::i32) return std::string("int");
        if (type == ov::element::i64) return std::string("long");
        return std::string("float");
    };
    std::string scalar_t = "float";
    std::string start_t = "float";
    std::string stop_t = "float";
    std::string step_t = "float";
    if (d.output_type != ov::element::dynamic || d.element_type != ov::element::dynamic) {
        scalar_t = type_from_element(d.output_type != ov::element::dynamic ? d.output_type : d.element_type);
    }
    if (d.start_type != ov::element::dynamic) {
        start_t = type_from_element(d.start_type);
    }
    if (d.stop_type != ov::element::dynamic) {
        stop_t = type_from_element(d.stop_type);
    }
    if (d.step_type != ov::element::dynamic) {
        step_t = type_from_element(d.step_type);
    }
    if (auto func = get_entry_func(module);
        func && (d.output_type == ov::element::dynamic && d.element_type == ov::element::dynamic)) {
        auto ft = func.getFunctionType();
        if (ft.getNumResults() >= 1) {
            scalar_t = msl_type_from_mlir(ft.getResult(0));
        }
    }
    if (auto func = get_entry_func(module);
        func && (d.start_type == ov::element::dynamic ||
                 d.stop_type == ov::element::dynamic ||
                 d.step_type == ov::element::dynamic)) {
        auto ft = func.getFunctionType();
        if (ft.getNumInputs() >= 3) {
            if (d.start_type == ov::element::dynamic) {
                start_t = msl_type_from_mlir(ft.getInput(0));
            }
            if (d.stop_type == ov::element::dynamic) {
                stop_t = msl_type_from_mlir(ft.getInput(1));
            }
            if (d.step_type == ov::element::dynamic) {
                step_t = msl_type_from_mlir(ft.getInput(2));
            }
        } else {
            start_t = scalar_t;
            stop_t = scalar_t;
            step_t = scalar_t;
        }
    } else if (!module &&
               d.start_type == ov::element::dynamic &&
               d.stop_type == ov::element::dynamic &&
               d.step_type == ov::element::dynamic) {
        start_t = scalar_t;
        stop_t = scalar_t;
        step_t = scalar_t;
    }

    const bool is_half = (scalar_t == "half");
    const bool is_int = (scalar_t == "int" || scalar_t == "long");

    std::ostringstream ss;
    ss << "#include <metal_stdlib>\n";
    ss << "using namespace metal;\n";
    ss << "using scalar_t = " << scalar_t << ";\n";
    ss << "using start_t = " << start_t << ";\n";
    ss << "using stop_t = " << stop_t << ";\n";
    ss << "using step_t = " << step_t << ";\n";
    ss << "kernel void range_kernel(device const start_t* START [[buffer(0)]],\n";
    ss << "                        device const stop_t* STOP [[buffer(1)]],\n";
    ss << "                        device const step_t* STEP [[buffer(2)]],\n";
    ss << "                        device scalar_t* O [[buffer(3)]],\n";
    ss << "                        constant uint& N [[buffer(4)]],\n";
    ss << "                        uint gid [[thread_position_in_grid]]) {\n";
    ss << "    if (gid >= N) return;\n";
    ss << "    (void)STOP;\n";
    if (is_int) {
        ss << "    O[gid] = START[0] + static_cast<scalar_t>(gid) * STEP[0];\n";
    } else if (is_half) {
        ss << "    float v = static_cast<float>(START[0]) + static_cast<float>(gid) * static_cast<float>(STEP[0]);\n";
        ss << "    O[gid] = static_cast<scalar_t>(v);\n";
    } else {
        ss << "    O[gid] = START[0] + static_cast<scalar_t>(gid) * STEP[0];\n";
    }
    ss << "}\n";
    return ss.str();
}

}  // namespace gfx_plugin
}  // namespace ov
