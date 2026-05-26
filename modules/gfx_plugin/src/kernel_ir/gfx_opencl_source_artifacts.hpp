// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//
#pragma once

#include <cstdint>
#include <memory>
#include <optional>
#include <string>
#include <string_view>
#include <vector>

#include "compiler/executable_bundle.hpp"
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
    Mod = 9,
    FloorMod = 10,
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
    LogicalNot = 48,
    LogicalAnd = 49,
    LogicalOr = 50,
    LogicalXor = 51,
    ReduceLogicalAnd = 64,
    ReduceLogicalOr = 65,
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
    Input0Dim0 = 16,
    Input0Dim1 = 17,
    Input0Dim2 = 18,
    Input0Dim3 = 19,
    Input0Dim4 = 20,
    Input0Dim5 = 21,
    Input0Dim6 = 22,
    Input0Dim7 = 23,
    Input1Dim0 = 24,
    Input1Dim1 = 25,
    Input1Dim2 = 26,
    Input1Dim3 = 27,
    Input1Dim4 = 28,
    Input1Dim5 = 29,
    Input1Dim6 = 30,
    Input1Dim7 = 31,
    Input2Dim0 = 32,
    Input2Dim1 = 33,
    Input2Dim2 = 34,
    Input2Dim3 = 35,
    Input2Dim4 = 36,
    Input2Dim5 = 37,
    Input2Dim6 = 38,
    Input2Dim7 = 39,
    Input3Dim0 = 40,
    Input3Dim1 = 41,
    Input3Dim2 = 42,
    Input3Dim3 = 43,
    Input3Dim4 = 44,
    Input3Dim5 = 45,
    Input3Dim6 = 46,
    Input3Dim7 = 47,
    Output0Dim0 = 64,
    Output0Dim1 = 65,
    Output0Dim2 = 66,
    Output0Dim3 = 67,
    Output0Dim4 = 68,
    Output0Dim5 = 69,
    Output0Dim6 = 70,
    Output0Dim7 = 71,
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
    std::vector<uint32_t> source_static_u32_scalars;
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

class GfxOpenClSourceArtifactPayload final : public compiler::KernelArtifactPayload {
public:
    explicit GfxOpenClSourceArtifactPayload(GfxOpenClSourceArtifact artifact);

    compiler::KernelArtifactPayloadKind payload_kind() const noexcept override;
    std::string_view backend_domain() const noexcept override;
    std::string_view source_id() const noexcept override;
    std::string_view entry_point() const noexcept override;
    bool valid() const noexcept override;

    const GfxOpenClSourceArtifact& artifact() const noexcept {
        return m_artifact;
    }

private:
    GfxOpenClSourceArtifact m_artifact;
};

std::optional<GfxOpenClSourceArtifact> resolve_gfx_opencl_source_artifact(
    const std::shared_ptr<const ov::Node>& node);

std::optional<GfxOpenClSourceArtifact> make_gfx_opencl_concat_chunk_source_artifact(
    const GfxOpenClSourceArtifact& base_artifact,
    uint32_t input_begin,
    uint32_t input_count);

std::optional<GfxOpenClSourceArtifact> make_gfx_opencl_split_chunk_source_artifact(
    const GfxOpenClSourceArtifact& base_artifact,
    uint32_t output_begin,
    uint32_t output_count);

std::string gfx_opencl_source_artifact_build_options(
    const GfxOpenClSourceArtifact& artifact);

}  // namespace gfx_plugin
}  // namespace ov
