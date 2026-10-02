// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//
#pragma once

#include <cstddef>
#include <memory>
#include <string>
#include <vector>

#include "runtime/executable_descriptor.hpp"

namespace ov {
namespace gfx_plugin {

class GpuBufferManager;
class GpuStage;
struct GpuTensor;

class ResourceBindingTable final {
public:
  static ResourceBindingTable for_stage(const std::vector<GpuTensor *> &inputs,
                                        const std::vector<GpuTensor *> &outputs);
  static ResourceBindingTable
  for_stage(const std::vector<GpuTensor *> &inputs,
            const std::vector<GpuTensor *> &outputs,
            const RuntimeStageExecutableDescriptor &descriptor);

  const std::vector<GpuTensor *> &inputs() const noexcept { return m_inputs; }
  const std::vector<GpuTensor *> &outputs() const noexcept { return m_outputs; }
  const std::vector<std::string> &input_region_ids() const noexcept {
    return m_input_region_ids;
  }
  const std::vector<std::string> &output_region_ids() const noexcept {
    return m_output_region_ids;
  }

  bool compatible_with(const RuntimeStageExecutableDescriptor &descriptor) const
      noexcept;

private:
  std::vector<GpuTensor *> m_inputs;
  std::vector<GpuTensor *> m_outputs;
  std::vector<std::string> m_input_region_ids;
  std::vector<std::string> m_output_region_ids;
};

class PreparedKernelExecutable final {
public:
  explicit PreparedKernelExecutable(
      const RuntimeStageExecutableDescriptor &descriptor);

  const RuntimeStageExecutableDescriptor &descriptor() const noexcept {
    return *m_descriptor;
  }
  const ResourceBindingTable &resource_bindings() const noexcept {
    return m_bindings;
  }
  bool prepared() const noexcept { return m_prepared; }

  void prepare(GpuStage &stage, GpuBufferManager *buffer_manager,
               ResourceBindingTable bindings);
  void bind(ResourceBindingTable bindings);

private:
  const RuntimeStageExecutableDescriptor *m_descriptor = nullptr;
  ResourceBindingTable m_bindings;
  bool m_prepared = false;
};

class RuntimeSession final {
public:
  explicit RuntimeSession(
      std::shared_ptr<const RuntimeExecutableDescriptor> descriptor);

  const RuntimeExecutableDescriptor &descriptor() const noexcept {
    return *m_descriptor;
  }
  size_t stage_count() const noexcept;

  ResourceBindingTable
  make_binding_table(size_t stage_index, const std::vector<GpuTensor *> &inputs,
                     const std::vector<GpuTensor *> &outputs) const;

  PreparedKernelExecutable prepare_stage(size_t stage_index, GpuStage &stage,
                                         GpuBufferManager *buffer_manager,
                                         ResourceBindingTable bindings) const;

  const RuntimeStageExecutableDescriptor &stage_descriptor(size_t stage_index)
      const;

private:
  std::shared_ptr<const RuntimeExecutableDescriptor> m_descriptor;
};

} // namespace gfx_plugin
} // namespace ov
