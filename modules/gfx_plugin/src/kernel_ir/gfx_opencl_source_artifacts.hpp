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

#include "kernel_ir/gfx_kernel_manifest.hpp"
#include "openvino/core/node.hpp"
#include "common/artifact_payload.hpp"

namespace ov {
namespace gfx_plugin {

struct GfxOpenClSourceArtifact;

enum class GfxOpenClSourceChunkBindingRole : uint32_t {
  DirectInputs = 0,
  DirectOutputs = 1,
};

struct GfxOpenClSourceChunkArtifact {
  uint32_t binding_begin = 0;
  uint32_t binding_count = 0;
  GfxOpenClSourceChunkBindingRole binding_role =
      GfxOpenClSourceChunkBindingRole::DirectInputs;
  uint32_t element_count_multiplier = 1;
  uint32_t element_count_divisor = 1;
  std::shared_ptr<const GfxOpenClSourceArtifact> artifact;
};

enum class GfxOpenClArtifactOp : uint32_t {
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
  Elu = 26,
  GeluErf = 27,
  GeluTanh = 28,
  HSwish = 29,
  HSigmoid = 30,
  SoftPlus = 31,
  Swish = 79,
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
  ReduceSum = 66,
  ReduceMean = 67,
  ReduceMax = 68,
  ReduceMin = 69,
  ReduceProd = 70,
  ReduceL1 = 71,
  ReduceL2 = 72,
  Mish = 80,
  SoftSign = 81,
  Sign = 82,
  Clamp = 83,
  Sin = 84,
  Cos = 85,
  Tan = 86,
  Erf = 87,
  Asin = 88,
  Acos = 89,
  Atan = 90,
  Asinh = 91,
  Acosh = 92,
  Atanh = 93,
  Sinh = 94,
  Cosh = 95,
  RoundEven = 96,
  RoundAway = 97,
  Softmax = 98,
  MaxPool = 99,
  AvgPool = 100,
};

enum class GfxOpenClArtifactInputMode : uint32_t {
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
  StaticF32 = 5,
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
  std::vector<float> static_f32_scalars;
  std::vector<uint32_t> source_static_u32_scalars;
  std::vector<size_t> direct_input_indices;
  uint32_t arg_count = 0;
  uint32_t local_size_hint = 64;
  uint32_t direct_input_count = 0;
  uint32_t direct_output_count = 1;
  uint32_t input_chunk_size = 0;
  uint32_t output_chunk_size = 0;
  std::vector<GfxOpenClSourceChunkArtifact> planned_chunks;
  GfxOpenClSourceElementCountSource element_count_source =
      GfxOpenClSourceElementCountSource::Output0;
  GfxOpenClArtifactOp op = GfxOpenClArtifactOp::Identity;
  GfxOpenClArtifactInputMode input_mode = GfxOpenClArtifactInputMode::Direct;
  float scalar_constant_f32 = 0.0f;
};

class GfxOpenClSourceArtifactPayload final
    : public KernelArtifactPayload {
public:
  explicit GfxOpenClSourceArtifactPayload(GfxOpenClSourceArtifact artifact);

  KernelArtifactPayloadKind payload_kind() const noexcept override;
  std::string_view backend_domain() const noexcept override;
  std::string_view source_id() const noexcept override;
  std::string_view entry_point() const noexcept override;
  bool valid() const noexcept override;

  const GfxOpenClSourceArtifact &artifact() const noexcept {
    return m_artifact;
  }

private:
  GfxOpenClSourceArtifact m_artifact;
};

GfxKernelStageManifest make_opencl_source_manifest(
    GfxKernelStageFamily family, std::string specialization_key,
    std::string entry_point, uint32_t direct_inputs, uint32_t scalar_arg_count,
    uint32_t direct_outputs = 1);

GfxOpenClSourceArtifact make_opencl_source_artifact(
    GfxKernelStageManifest manifest, std::string source_id, std::string source,
    std::vector<GfxOpenClSourceScalarArg> scalar_args,
    std::vector<size_t> direct_input_indices, GfxOpenClArtifactOp op,
    GfxOpenClArtifactInputMode input_mode = GfxOpenClArtifactInputMode::Direct,
    float scalar_constant_f32 = 0.0f,
    std::vector<uint32_t> static_u32_scalars = {},
    GfxOpenClSourceElementCountSource element_count_source =
        GfxOpenClSourceElementCountSource::Output0,
    std::vector<float> static_f32_scalars = {});

std::string gfx_opencl_source_artifact_build_options(
    const GfxOpenClSourceArtifact &artifact);

} // namespace gfx_plugin
} // namespace ov
