// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "mlir/msl_codegen_compressed_matmul.hpp"

#include "mlir/codegen_common.hpp"
#include "mlir/gfx_backend_custom_kernel_adapter.hpp"
#include "mlir/msl_codegen_apple_msl_dispatch.hpp"
#include "openvino/core/except.hpp"
#include "openvino/core/shape_util.hpp"
#include "openvino/op/concat.hpp"
#include "openvino/op/convert.hpp"
#include "openvino/op/matmul.hpp"
#include "openvino/op/multiply.hpp"
#include "openvino/op/reshape.hpp"

#include <algorithm>
#include <cstring>
#include <sstream>
#include <utility>

namespace ov {
namespace gfx_plugin {
namespace {

std::shared_ptr<const ov::op::v0::Constant>
as_constant_node(const ov::Output<ov::Node> &value) {
  return ov::as_type_ptr<const ov::op::v0::Constant>(
      value.get_node_shared_ptr());
}

std::optional<CompressedMatMulPart>
detect_compressed_matmul_part(const ov::Output<ov::Node> &value,
                              const ov::Shape &b_shape) {
  if (b_shape.size() != 2) {
    return std::nullopt;
  }

  auto source = value.get_node_shared_ptr();
  if (auto convert = ov::as_type_ptr<const ov::op::v0::Convert>(source)) {
    source = convert->input_value(0).get_node_shared_ptr();
  }
  auto reshape = ov::as_type_ptr<const ov::op::v1::Reshape>(source);
  if (!reshape) {
    return std::nullopt;
  }
  auto mul = ov::as_type_ptr<const ov::op::v1::Multiply>(
      reshape->input_value(0).get_node_shared_ptr());
  if (!mul) {
    return std::nullopt;
  }

  std::shared_ptr<const ov::op::v0::Constant> weights;
  std::shared_ptr<const ov::op::v0::Constant> scale;
  for (size_t i = 0; i < mul->get_input_size(); ++i) {
    auto input = mul->input_value(i);
    if (auto convert = ov::as_type_ptr<const ov::op::v0::Convert>(
            input.get_node_shared_ptr())) {
      if (auto constant = as_constant_node(convert->input_value(0))) {
        const auto et = constant->get_element_type();
        if (et == ov::element::i4 || et == ov::element::u4 ||
            et == ov::element::i8 || et == ov::element::u8) {
          weights = constant;
          continue;
        }
      }
    }
    if (auto constant = as_constant_node(input)) {
      if (constant->get_element_type() == ov::element::f16 ||
          constant->get_element_type() == ov::element::f32) {
        scale = constant;
      }
    }
  }
  if (!weights || !scale) {
    return std::nullopt;
  }

  const auto raw_shape = weights->get_shape();
  const auto scale_shape = scale->get_shape();
  if (raw_shape.size() != 3 || scale_shape.size() != 3 ||
      raw_shape[0] != b_shape[0] || scale_shape[0] != raw_shape[0] ||
      scale_shape[1] != raw_shape[1] || scale_shape[2] != 1) {
    return std::nullopt;
  }
  const size_t n = raw_shape[0];
  const size_t groups = raw_shape[1];
  const size_t group_size = raw_shape[2];
  const size_t k = groups * group_size;
  if (n == 0 || k == 0 || b_shape[1] != k) {
    return std::nullopt;
  }

  CompressedMatMulPart part;
  part.weights = weights;
  part.scale = scale;
  part.n = n;
  part.groups = groups;
  part.group_size = group_size;
  part.k = k;
  return part;
}

uint32_t floor_power_of_two(uint32_t value) {
  if (value == 0) {
    return 0;
  }
  uint32_t result = 1;
  while (result <= value / 2) {
    result <<= 1;
  }
  return result;
}

uint8_t read_quantized_weight_value(const ov::op::v0::Constant &weights,
                                    size_t logical_index) {
  const auto et = weights.get_element_type();
  const auto *raw = static_cast<const uint8_t *>(weights.get_data_ptr());
  OPENVINO_ASSERT(raw, "GFX compressed MatMul: empty weight constant");
  if (et == ov::element::i4 || et == ov::element::u4) {
    const uint8_t packed = raw[logical_index >> 1];
    return ((logical_index & 1u) == 0u)
               ? static_cast<uint8_t>(packed & 0x0fu)
               : static_cast<uint8_t>((packed >> 4) & 0x0fu);
  }
  if (et == ov::element::i8 || et == ov::element::u8) {
    return raw[logical_index];
  }
  OPENVINO_THROW("GFX compressed MatMul: unsupported weight type ", et);
}

void write_quantized_weight_value(std::vector<uint8_t> &packed,
                                  ov::element::Type type, size_t logical_index,
                                  uint8_t value) {
  if (type == ov::element::i4 || type == ov::element::u4) {
    const size_t byte_index = logical_index >> 1;
    const uint8_t nibble = static_cast<uint8_t>(value & 0x0fu);
    if ((logical_index & 1u) == 0u) {
      packed[byte_index] =
          static_cast<uint8_t>((packed[byte_index] & 0xf0u) | nibble);
    } else {
      packed[byte_index] = static_cast<uint8_t>(
          (packed[byte_index] & 0x0fu) | static_cast<uint8_t>(nibble << 4));
    }
    return;
  }
  if (type == ov::element::i8 || type == ov::element::u8) {
    packed[logical_index] = value;
    return;
  }
  OPENVINO_THROW("GFX compressed MatMul: unsupported weight type ", type);
}

} // namespace

std::optional<CompressedMatMulInfo>
detect_compressed_matmul_weights(const std::shared_ptr<const ov::Node> &node) {
  auto matmul = ov::as_type_ptr<const ov::op::v0::MatMul>(node);
  if (!matmul || !matmul->get_transpose_b() || matmul->get_input_size() != 2) {
    return std::nullopt;
  }
  if (!matmul->get_input_partial_shape(1).is_static()) {
    return std::nullopt;
  }
  const auto b_shape = matmul->get_input_shape(1);
  if (b_shape.size() != 2) {
    return std::nullopt;
  }

  std::vector<CompressedMatMulPart> parts;
  auto source = matmul->input_value(1).get_node_shared_ptr();
  if (auto convert = ov::as_type_ptr<const ov::op::v0::Convert>(source)) {
    source = convert->input_value(0).get_node_shared_ptr();
  }
  if (auto concat = ov::as_type_ptr<const ov::op::v0::Concat>(source)) {
    if (concat->get_axis() != 0) {
      return std::nullopt;
    }
    size_t total_n = 0;
    for (size_t i = 0; i < concat->get_input_size(); ++i) {
      const auto input = concat->input_value(i);
      if (!input.get_partial_shape().is_static()) {
        return std::nullopt;
      }
      const auto part_shape = input.get_shape();
      auto part = detect_compressed_matmul_part(input, part_shape);
      if (!part) {
        return std::nullopt;
      }
      total_n += part->n;
      parts.push_back(std::move(*part));
    }
    if (parts.empty() || total_n != b_shape[0]) {
      return std::nullopt;
    }
  } else {
    auto part = detect_compressed_matmul_part(matmul->input_value(1), b_shape);
    if (!part) {
      return std::nullopt;
    }
    parts.push_back(std::move(*part));
  }
  if (parts.empty()) {
    return std::nullopt;
  }
  const auto weight_type = parts.front().weights->get_element_type();
  const auto scale_type = parts.front().scale->get_element_type();
  const size_t groups = parts.front().groups;
  const size_t group_size = parts.front().group_size;
  const size_t k = parts.front().k;
  size_t n = 0;
  for (const auto &part : parts) {
    if (!part.weights || !part.scale ||
        part.weights->get_element_type() != weight_type ||
        part.scale->get_element_type() != scale_type || part.groups != groups ||
        part.group_size != group_size || part.k != k) {
      return std::nullopt;
    }
    n += part.n;
  }
  if (n == 0 || k == 0 || b_shape[0] != n || b_shape[1] != k) {
    return std::nullopt;
  }

  CompressedMatMulInfo info;
  info.parts = std::move(parts);
  info.weights = info.parts.front().weights;
  info.scale = info.parts.front().scale;
  info.input_type = matmul->get_input_element_type(0);
  info.output_type = matmul->get_output_element_type(0);
  info.signed_weights =
      weight_type == ov::element::i4 || weight_type == ov::element::i8;
  info.n = n;
  info.k = k;
  info.groups = groups;
  info.group_size = group_size;
  return info;
}

uint32_t
compressed_matmul_parallel_reduction_threads(const CompressedMatMulInfo &info,
                                             const GfxParallelismCaps &caps) {
  if (info.k < 512 || info.n < 16) {
    return 1;
  }

  const uint32_t max_threads = std::max<uint32_t>(
      1u, std::min(std::max<uint32_t>(1u, caps.max_total_threads_per_group),
                   std::max<uint32_t>(1u, caps.max_threads_per_group[0])));
  const uint32_t wave = std::max<uint32_t>(
      1u, std::max(caps.subgroup_size, caps.preferred_simd_width));
  const uint32_t k_tiles = static_cast<uint32_t>((info.k + 1023) / 1024);
  const uint32_t desired = wave * std::max<uint32_t>(2u, k_tiles * 2u);
  const uint32_t threads = floor_power_of_two(std::min(max_threads, desired));
  return threads >= 2 ? threads : 1;
}

uint32_t compressed_matmul_output_block(const CompressedMatMulInfo &info,
                                        const GfxParallelismCaps &caps,
                                        uint32_t reduction_threads) {
  if (reduction_threads < 2 || info.k < 1024 || info.n < 4) {
    return 1;
  }
  const uint32_t max_threads =
      std::max<uint32_t>(1u, caps.max_total_threads_per_group);
  if (max_threads < 128) {
    return 1;
  }
  const uint32_t max_block = 8u;
  return std::min<uint32_t>(max_block,
                            floor_power_of_two(static_cast<uint32_t>(
                                std::min<size_t>(info.n, max_block))));
}

std::vector<uint8_t> pack_compressed_matmul_weights_for_output_block(
    const CompressedMatMulInfo &info, uint32_t output_block) {
  OPENVINO_ASSERT(!info.parts.empty(),
                  "GFX compressed MatMul: no compressed weight parts");
  const auto et = info.weights->get_element_type();
  const size_t col_blocks = (info.n + output_block - 1) / output_block;
  const size_t logical_values = col_blocks * info.k * output_block;
  const size_t bytes = (et == ov::element::i4 || et == ov::element::u4)
                           ? ((logical_values + 1) / 2)
                           : logical_values;
  std::vector<uint8_t> packed(bytes, 0);

  for (size_t col_block = 0; col_block < col_blocks; ++col_block) {
    for (size_t kk = 0; kk < info.k; ++kk) {
      for (uint32_t lane = 0; lane < output_block; ++lane) {
        const size_t col = col_block * output_block + lane;
        const size_t packed_index =
            ((col_block * info.k + kk) * output_block) + lane;
        if (col >= info.n) {
          write_quantized_weight_value(packed, et, packed_index, 0);
          continue;
        }
        size_t part_offset = 0;
        const CompressedMatMulPart *selected_part = nullptr;
        for (const auto &part : info.parts) {
          if (col < part_offset + part.n) {
            selected_part = &part;
            break;
          }
          part_offset += part.n;
        }
        OPENVINO_ASSERT(selected_part,
                        "GFX compressed MatMul: invalid packed weight column");
        const size_t local_col = col - part_offset;
        const size_t source_index = local_col * info.k + kk;
        write_quantized_weight_value(
            packed, et, packed_index,
            read_quantized_weight_value(*selected_part->weights, source_index));
      }
    }
  }
  return packed;
}

std::vector<uint8_t>
pack_compressed_matmul_scales(const CompressedMatMulInfo &info) {
  OPENVINO_ASSERT(!info.parts.empty(),
                  "GFX compressed MatMul: no compressed scale parts");
  const auto scale_type = info.scale->get_element_type();
  const size_t element_count = ov::shape_size(info.scale->get_shape());
  OPENVINO_ASSERT(element_count > 0,
                  "GFX compressed MatMul: empty scale shape");
  const size_t element_bytes = info.scale->get_byte_size() / element_count;
  OPENVINO_ASSERT(element_bytes > 0,
                  "GFX compressed MatMul: invalid scale element size");

  std::vector<uint8_t> packed(info.n * info.groups * element_bytes, 0);
  size_t row_offset = 0;
  for (const auto &part : info.parts) {
    OPENVINO_ASSERT(part.scale && part.scale->get_element_type() == scale_type,
                    "GFX compressed MatMul: inconsistent scale part");
    const auto shape = part.scale->get_shape();
    OPENVINO_ASSERT(shape.size() == 3 && shape[0] == part.n &&
                        shape[1] == info.groups && shape[2] == 1,
                    "GFX compressed MatMul: unsupported scale part shape");
    const size_t bytes = part.n * info.groups * element_bytes;
    const auto *src = static_cast<const uint8_t *>(part.scale->get_data_ptr());
    OPENVINO_ASSERT(src && bytes > 0,
                    "GFX compressed MatMul: empty scale part");
    std::memcpy(packed.data() + row_offset * info.groups * element_bytes, src,
                bytes);
    row_offset += part.n;
  }
  OPENVINO_ASSERT(row_offset == info.n,
                  "GFX compressed MatMul: packed scale size mismatch");
  return packed;
}

std::string generate_msl_for_compressed_matmul(const CompressedMatMulInfo &info,
                                               uint32_t reduction_threads,
                                               uint32_t output_block) {
  const std::string input_scalar =
      msl_type_from_element(info.input_type).empty()
          ? "float"
          : msl_type_from_element(info.input_type);
  const std::string output_scalar =
      msl_type_from_element(info.output_type).empty()
          ? "float"
          : msl_type_from_element(info.output_type);
  const std::string scale_scalar =
      msl_type_from_element(info.scale->get_element_type()).empty()
          ? "half"
          : msl_type_from_element(info.scale->get_element_type());
  const bool is_i4 = info.weights->get_element_type() == ov::element::i4 ||
                     info.weights->get_element_type() == ov::element::u4;

  std::ostringstream ss;
  ss << "#include <metal_stdlib>\n";
  ss << "using namespace metal;\n";
  ss << "constant uint N = " << info.n << ";\n";
  ss << "constant uint K = " << info.k << ";\n";
  ss << "constant uint GROUPS = " << info.groups << ";\n";
  ss << "constant uint GROUP_SIZE = " << info.group_size << ";\n";
  ss << "constant uint REDUCE_THREADS = " << reduction_threads << ";\n";
  ss << "constant uint OUTPUT_BLOCK = " << output_block << ";\n";
  ss << "constant uint COL_BLOCKS = "
     << ((info.n + output_block - 1) / output_block) << ";\n";
  ss << "inline float load_qweight(device const uchar* weights, uint idx) {\n";
  if (is_i4) {
    ss << "  uchar packed = weights[idx >> 1];\n";
    ss << "  uint q = ((idx & 1u) == 0u) ? uint(packed & 0x0fu) : uint(packed "
          ">> 4);\n";
    if (info.signed_weights) {
      ss << "  int s = (q >= 8u) ? int(q) - 16 : int(q);\n";
      ss << "  return float(s);\n";
    } else {
      ss << "  return float(q);\n";
    }
  } else {
    if (info.signed_weights) {
      ss << "  return float(as_type<char>(weights[idx]));\n";
    } else {
      ss << "  return float(weights[idx]);\n";
    }
  }
  ss << "}\n";
  ss << "kernel void compressed_matmul_kernel(\n";
  ss << "  device const " << input_scalar << "* A [[buffer(0)]],\n";
  ss << "  device const uchar* W [[buffer(1)]],\n";
  ss << "  device const " << scale_scalar << "* S [[buffer(2)]],\n";
  ss << "  device " << output_scalar << "* C [[buffer(3)]],\n";
  ss << "  uint gid [[thread_position_in_grid]],\n";
  ss << "  uint lane [[thread_index_in_threadgroup]]) {\n";
  if (reduction_threads <= 1) {
    ss << "  uint block_id = gid;\n";
    ss << "  uint col_block = block_id % COL_BLOCKS;\n";
    ss << "  uint col_base = col_block * OUTPUT_BLOCK;\n";
    ss << "  uint row = block_id / COL_BLOCKS;\n";
    for (uint32_t i = 0; i < output_block; ++i) {
      ss << "  float acc" << i << " = 0.0f;\n";
    }
    ss << "  for (uint group = 0; group < GROUPS; ++group) {\n";
    for (uint32_t i = 0; i < output_block; ++i) {
      ss << "    float scale" << i << " = 0.0f;\n";
      ss << "    if (col_base + " << i << "u < N) scale" << i
         << " = float(S[(col_base + " << i << "u) * GROUPS + group]);\n";
    }
    ss << "    for (uint in_group = 0; in_group < GROUP_SIZE; ++in_group) {\n";
    ss << "      uint kk = group * GROUP_SIZE + in_group;\n";
    ss << "      float a = float(A[row * K + kk]);\n";
    ss << "      uint w_base = ((col_block * K + kk) * OUTPUT_BLOCK);\n";
    for (uint32_t i = 0; i < output_block; ++i) {
      ss << "      if (col_base + " << i << "u < N) acc" << i
         << " += a * load_qweight(W, w_base + " << i << "u) * scale" << i
         << ";\n";
    }
    ss << "    }\n";
    ss << "  }\n";
    for (uint32_t i = 0; i < output_block; ++i) {
      ss << "  if (col_base + " << i << "u < N) C[row * N + col_base + " << i
         << "u] = " << output_scalar << "(acc" << i << ");\n";
    }
    ss << "}\n";
    return ss.str();
  }
  ss << "  uint block_id = gid / REDUCE_THREADS;\n";
  ss << "  uint col_block = block_id % COL_BLOCKS;\n";
  ss << "  uint col_base = col_block * OUTPUT_BLOCK;\n";
  ss << "  uint row = block_id / COL_BLOCKS;\n";
  for (uint32_t i = 0; i < output_block; ++i) {
    ss << "  float acc" << i << " = 0.0f;\n";
  }
  ss << "  for (uint group = 0; group < GROUPS; ++group) {\n";
  for (uint32_t i = 0; i < output_block; ++i) {
    ss << "    float scale" << i << " = 0.0f;\n";
    ss << "    if (col_base + " << i << "u < N) scale" << i
       << " = float(S[(col_base + " << i << "u) * GROUPS + group]);\n";
  }
  ss << "    for (uint in_group = lane; in_group < GROUP_SIZE; in_group += "
        "REDUCE_THREADS) {\n";
  ss << "      uint kk = group * GROUP_SIZE + in_group;\n";
  ss << "      float a = float(A[row * K + kk]);\n";
  ss << "      uint w_base = ((col_block * K + kk) * OUTPUT_BLOCK);\n";
  for (uint32_t i = 0; i < output_block; ++i) {
    ss << "      if (col_base + " << i << "u < N) acc" << i
       << " += a * load_qweight(W, w_base + " << i << "u) * scale" << i
       << ";\n";
  }
  ss << "    }\n";
  ss << "  }\n";
  ss << "  threadgroup float partial[" << (reduction_threads * output_block)
     << "];\n";
  for (uint32_t i = 0; i < output_block; ++i) {
    ss << "  partial[lane * OUTPUT_BLOCK + " << i << "u] = acc" << i << ";\n";
  }
  ss << "  threadgroup_barrier(mem_flags::mem_threadgroup);\n";
  ss << "  for (uint stride = " << (reduction_threads / 2)
     << "; stride > 0; stride >>= 1) {\n";
  ss << "    if (lane < stride) {\n";
  for (uint32_t i = 0; i < output_block; ++i) {
    ss << "      partial[lane * OUTPUT_BLOCK + " << i
       << "u] += partial[(lane + stride) * OUTPUT_BLOCK + " << i << "u];\n";
  }
  ss << "    }\n";
  ss << "    threadgroup_barrier(mem_flags::mem_threadgroup);\n";
  ss << "  }\n";
  ss << "  if (lane == 0) {\n";
  for (uint32_t i = 0; i < output_block; ++i) {
    ss << "    if (col_base + " << i << "u < N) C[row * N + col_base + " << i
       << "u] = " << output_scalar << "(partial[" << i << "u]);\n";
  }
  ss << "  }\n";
  ss << "}\n";
  return ss.str();
}

GfxMslGeneratedKernelSourcePlan
make_compressed_matmul_msl_kernel_source_plan(const CompressedMatMulInfo &info,
                                              uint32_t reduction_threads,
                                              uint32_t output_block) {
  KernelSource source;
  source.entry_point = "compressed_matmul_kernel";
  source.msl_source =
      generate_msl_for_compressed_matmul(info, reduction_threads, output_block);
  return make_msl_generated_custom_kernel_source_plan(std::move(source),
                                                      "CompressedMatMul");
}

} // namespace gfx_plugin
} // namespace ov
