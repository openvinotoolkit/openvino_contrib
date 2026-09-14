// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "runtime/descriptor_const_tensor_materializer.hpp"

#include <algorithm>
#include <sstream>

#include "kernel_ir/gfx_kernel_cache.hpp"
#include "openvino/core/except.hpp"
#include "runtime/executable_descriptor.hpp"
#include "runtime/gpu_buffer_manager.hpp"
#include "runtime/tensor_binding_contract.hpp"

namespace ov {
namespace gfx_plugin {
namespace {

size_t descriptor_const_slot_count(
    const RuntimeStageExecutableDescriptor &descriptor) {
  size_t count = 0;
  for (const auto &tensor : descriptor.const_tensors) {
    count = std::max(count, tensor.source_input_index + 1);
  }
  return count;
}

std::string make_const_tensor_key(std::string_view cache_prefix,
                                  const RuntimeStageExecutableDescriptor &descriptor,
                                  const KernelArtifactConstTensor &tensor) {
  std::ostringstream key;
  key << cache_prefix << "/" << descriptor.kernel_id << "/const/"
      << tensor.source_input_index << "/" << tensor.logical_name << "/"
      << tensor.element_type << "/" << tensor.bytes.size() << "/";
  if (!tensor.bytes.empty()) {
    key << gfx_hash_bytes(tensor.bytes.data(), tensor.bytes.size());
  } else {
    key << "empty";
  }
  return key.str();
}

} // namespace

DescriptorConstTensorSlots materialize_descriptor_const_tensor_slots(
    GpuBufferManager &buffer_manager,
    const RuntimeStageExecutableDescriptor &descriptor,
    std::string_view cache_prefix) {
  DescriptorConstTensorSlots slots;
  const auto slot_count = descriptor_const_slot_count(descriptor);
  slots.buffers.resize(slot_count);
  slots.present.assign(slot_count, false);
  if (descriptor.const_tensors.empty()) {
    return slots;
  }

  OPENVINO_ASSERT(
      buffer_manager.supports_const_cache(),
      "GFX: descriptor-owned const tensors require const cache support for ",
      descriptor.stage_name);

  for (const auto &const_tensor : descriptor.const_tensors) {
    OPENVINO_ASSERT(
        const_tensor.source_input_index < slots.buffers.size(),
        "GFX: descriptor-owned const tensor source input index drift for ",
        descriptor.stage_name);
    OPENVINO_ASSERT(
        !slots.present[const_tensor.source_input_index],
        "GFX: duplicate descriptor-owned const tensor source input index ",
        const_tensor.source_input_index, " for ", descriptor.stage_name);
    auto &tensor = slots.buffers[const_tensor.source_input_index];
    if (!const_tensor.bytes.empty()) {
      const auto element_type = element_type_from_contract(
          const_tensor.element_type);
      OPENVINO_ASSERT(element_type != ov::element::dynamic,
                      "GFX: descriptor-owned const tensor has unsupported "
                      "element type ",
                      const_tensor.element_type, " for ",
                      descriptor.stage_name);
      auto buffer = buffer_manager.wrap_const(
          make_const_tensor_key(cache_prefix, descriptor, const_tensor),
          const_tensor.bytes.data(), const_tensor.bytes.size(), element_type);
      OPENVINO_ASSERT(buffer.valid(),
                      "GFX: failed to materialize descriptor-owned const "
                      "tensor ",
                      const_tensor.source_input_index, " for ",
                      descriptor.stage_name);
      buffer.owned = false;
      tensor.buf = buffer;
      tensor.expected_type = element_type;
    } else {
      tensor.expected_type =
          element_type_from_contract(const_tensor.element_type);
    }
    tensor.shape = const_tensor.shape;
    tensor.prefer_private = false;
    slots.present[const_tensor.source_input_index] = true;
  }
  return slots;
}

std::vector<GpuTensor *> descriptor_const_tensor_args(
    DescriptorConstTensorSlots &slots, size_t expected_count) {
  std::vector<GpuTensor *> tensors;
  tensors.reserve(expected_count);
  for (size_t input_idx = 0;
       input_idx < slots.buffers.size() && tensors.size() < expected_count;
       ++input_idx) {
    if (input_idx < slots.present.size() && slots.present[input_idx] &&
        slots.buffers[input_idx].buf.valid()) {
      tensors.push_back(&slots.buffers[input_idx]);
    }
  }
  return tensors;
}

} // namespace gfx_plugin
} // namespace ov
