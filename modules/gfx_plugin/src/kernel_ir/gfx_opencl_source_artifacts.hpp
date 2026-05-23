// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//
#pragma once

#include <cstdint>
#include <memory>
#include <optional>
#include <string>
#include <vector>

#include "kernel_ir/gfx_kernel_manifest.hpp"
#include "openvino/core/node.hpp"

namespace ov {
namespace gfx_plugin {

enum class GfxOpenClBaselineOp : uint32_t {
    Identity = 0,
    Add = 1,
    Subtract = 2,
    Multiply = 3,
    Divide = 4,
    Maximum = 5,
    Minimum = 6,
    Power = 7,
    SquaredDifference = 8,
    Relu = 16,
    Sigmoid = 17,
    Tanh = 18,
    Abs = 19,
    Negative = 20,
    Exp = 21,
    Log = 22,
    Sqrt = 23,
    Floor = 24,
    Ceiling = 25,
    Equal = 32,
    NotEqual = 33,
    Greater = 34,
    GreaterEqual = 35,
    Less = 36,
    LessEqual = 37,
};

enum class GfxOpenClBaselineInputMode : uint32_t {
    Direct = 0,
    RhsScalar = 1,
    LhsScalar = 2,
    RhsScalarConstant = 3,
    LhsScalarConstant = 4,
};

enum class GfxOpenClSourceScalarArg : uint32_t {
    ElementCount = 0,
    OpCode = 1,
    InputMode = 2,
    ScalarConstantF32 = 3,
    StaticU32 = 4,
};

enum class GfxOpenClSourceElementCountSource : uint32_t {
    Output0 = 0,
    Input0 = 1,
};

struct GfxOpenClSourceArtifact {
    bool valid = false;
    GfxKernelArtifactRef artifact_ref;
    GfxKernelStageManifest stage_manifest;
    std::string source;
    std::vector<std::string> build_options;
    std::vector<GfxOpenClSourceScalarArg> scalar_args;
    std::vector<uint32_t> static_u32_scalars;
    std::vector<size_t> direct_input_indices;
    uint32_t arg_count = 0;
    uint32_t baseline_local_size = 64;
    uint32_t direct_input_count = 0;
    uint32_t direct_output_count = 1;
    GfxOpenClSourceElementCountSource element_count_source =
        GfxOpenClSourceElementCountSource::Output0;
    GfxOpenClBaselineOp op = GfxOpenClBaselineOp::Identity;
    GfxOpenClBaselineInputMode input_mode = GfxOpenClBaselineInputMode::Direct;
    float scalar_constant_f32 = 0.0f;
};

std::optional<GfxOpenClSourceArtifact> resolve_gfx_opencl_source_artifact(
    const std::shared_ptr<const ov::Node>& node);

std::string gfx_opencl_source_artifact_build_options(
    const GfxOpenClSourceArtifact& artifact);

}  // namespace gfx_plugin
}  // namespace ov
