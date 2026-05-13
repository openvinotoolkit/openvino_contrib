// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//
#pragma once

#include <cstddef>
#include <cstdint>
#include <memory>
#include <optional>
#include <string>
#include <vector>

#include "kernel_ir/gfx_codegen_backend.hpp"
#include "kernel_ir/gfx_custom_kernel_families.hpp"
#include "mlir/gfx_mlir_kernel_metadata.hpp"
#include "mlir/msl_codegen_apple_msl_dispatch.hpp"
#include "openvino/core/node.hpp"
#include "openvino/core/shape.hpp"
#include "openvino/core/type/element_type.hpp"
#include "openvino/op/constant.hpp"
#include "runtime/gfx_parallelism.hpp"

namespace ov {
namespace gfx_plugin {

struct CompressedMatMulPart {
  std::shared_ptr<const ov::op::v0::Constant> weights;
  std::shared_ptr<const ov::op::v0::Constant> scale;
  size_t n = 0;
  size_t groups = 0;
  size_t group_size = 0;
  size_t k = 0;
};

struct CompressedMatMulInfo {
  std::vector<CompressedMatMulPart> parts;
  std::shared_ptr<const ov::op::v0::Constant> weights;
  std::shared_ptr<const ov::op::v0::Constant> scale;
  ov::element::Type input_type = ov::element::dynamic;
  ov::element::Type output_type = ov::element::dynamic;
  bool signed_weights = true;
  size_t n = 0;
  size_t k = 0;
  size_t groups = 0;
  size_t group_size = 0;
};

std::optional<CompressedMatMulInfo>
detect_compressed_matmul_weights(const std::shared_ptr<const ov::Node> &node);
uint32_t
compressed_matmul_parallel_reduction_threads(const CompressedMatMulInfo &info,
                                             const GfxParallelismCaps &caps);
uint32_t compressed_matmul_output_block(const CompressedMatMulInfo &info,
                                        const GfxParallelismCaps &caps,
                                        uint32_t reduction_threads);
std::vector<uint8_t> pack_compressed_matmul_weights_for_output_block(
    const CompressedMatMulInfo &info, uint32_t output_block);
std::vector<uint8_t>
pack_compressed_matmul_scales(const CompressedMatMulInfo &info);
std::string generate_msl_for_compressed_matmul(const CompressedMatMulInfo &info,
                                               uint32_t reduction_threads,
                                               uint32_t output_block);

GfxMslGeneratedKernelSourcePlan
make_compressed_matmul_msl_kernel_source_plan(const CompressedMatMulInfo &info,
                                              uint32_t reduction_threads,
                                              uint32_t output_block);

} // namespace gfx_plugin
} // namespace ov
