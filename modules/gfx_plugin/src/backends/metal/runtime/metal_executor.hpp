// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//
#pragma once

#include <memory>
#include <string>
#include <vector>

#include "backends/metal/runtime/memory/buffer.hpp"
#include "runtime/executable_descriptor.hpp"
#include "runtime/gpu_backend_base.hpp"
#include "runtime/gpu_stage.hpp"

namespace ov {
namespace gfx_plugin {

class MetalProfiler;

class ICompiledKernel;
class GfxKernelSourcePayload;

// Metal runtime stage for compiler-owned MSL source artifacts.
class MetalStage final : public GpuStage {
public:
  MetalStage(const RuntimeStageExecutableDescriptor &descriptor,
             MetalDeviceHandle device, MetalCommandQueueHandle queue,
             std::shared_ptr<const ov::Node> source_node = {});

  void init(GpuBufferManager *buffer_manager) override;
  void prepare_runtime_handle(GpuBufferManager *buffer_manager) override;
  void execute(GpuCommandBufferHandle command_buffer) override;

  void set_inputs(const std::vector<GpuTensor *> &inputs) override;
  void set_output(GpuTensor *output) override;
  void set_output_refs(const std::vector<GpuTensor *> &outputs) override;
  void
  set_outputs(const std::vector<std::unique_ptr<GpuTensor>> &outputs) override;

  bool fuse_activation(ActivationKind kind, float alpha) override;
  bool fuse_input_activation(size_t input_idx, ActivationKind kind,
                             float alpha) override;
  bool fuse_residual_add() override;
  bool fuse_batchnorm(const BatchNormParams &params) override;
  bool fuse_bias(const BiasParams &params) override;

  void enable_profiling(bool enable) override;
  void set_profiler(void *profiler, uint32_t node_id,
                    const std::string &node_name,
                    const std::string &node_type) override;

  const std::string &name() const override { return m_name; }
  const std::string &type() const override { return m_type; }
  std::unique_ptr<GpuStage> clone() const override;

private:
  struct ProfileState;
  struct ConstBufferSet;

  void ensure_prepared();
  std::vector<GpuTensor *> resolve_outputs() const;
  std::vector<KernelArg>
  materialize_source_args(const std::vector<GpuTensor *> &outputs);
  std::vector<KernelArg>
  materialize_role_ordered_source_args(const GfxKernelSourcePayload &payload,
                                       const std::vector<GpuTensor *> &outputs);
  std::vector<int32_t> refresh_runtime_param_buffers(
      const GfxKernelSourcePayload &payload,
      const std::vector<GpuTensor *> &outputs,
      const std::vector<int32_t> &compiler_scalar_args);
  void prepare_constant_input_buffers();
  GpuTensor *resolve_input_tensor(size_t input_idx) const;
  KernelDispatch
  make_source_dispatch(const std::vector<GpuTensor *> &outputs) const;
  KernelExecutionHooks *prepare_profiling(ProfileState &state,
                                          KernelExecutionHooks &hooks);
  void finalize_profiling(const ProfileState &state);

  MetalDeviceHandle m_device = nullptr;
  [[maybe_unused]] MetalCommandQueueHandle m_queue = nullptr;
  RuntimeStageExecutableDescriptor m_descriptor;
  std::shared_ptr<ICompiledKernel> m_kernel;
  GpuBufferManager *m_buffer_manager = nullptr;
  std::shared_ptr<const ov::Node> m_node;
  std::string m_name;
  std::string m_type;
  std::vector<GpuTensor *> m_inputs;
  std::vector<GpuTensor *> m_outputs;
  GpuTensor *m_output = nullptr;
  std::shared_ptr<ConstBufferSet> m_const_buffers;
  std::vector<GpuTensor> m_kernel_extra_inputs;
  std::vector<int32_t> m_scalar_storage;
  bool m_profiling_enabled = false;
  void *m_profiler = nullptr;
  uint32_t m_profile_node_id = 0;
  std::string m_profile_node_name;
  std::string m_profile_node_type;
};

} // namespace gfx_plugin
} // namespace ov
