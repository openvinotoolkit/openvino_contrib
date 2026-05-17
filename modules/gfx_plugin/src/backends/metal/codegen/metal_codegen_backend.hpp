// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//
#pragma once

#include <memory>
#include <mutex>
#include <string>
#include <vector>

#include "kernel_ir/gfx_codegen_backend.hpp"
#include "backends/metal/runtime/memory/buffer.hpp"
#include "runtime/gfx_mpsrt_model.hpp"

#ifdef __OBJC__
#import <Metal/Metal.h>
#endif

namespace ov {
namespace gfx_plugin {

class MetalBindingSchema;
class MetalDeviceReuseContext;
namespace mpsrt {
struct MpsrtModel;
}  // namespace mpsrt
namespace metal {
namespace mpsrt {
class MpsrtContext;
struct MpsrtPreparedModel;
}  // namespace mpsrt
}  // namespace metal

class MetalCodegenBackend final : public ICodegenBackend {
public:
    explicit MetalCodegenBackend(MetalDeviceHandle device);

    std::shared_ptr<ICompiledKernel> compile(const KernelSource& source,
                                             std::string* log = nullptr) override;

private:
    MetalDeviceHandle m_device = nullptr;
    std::shared_ptr<MetalDeviceReuseContext> m_reuse_context;
};

class MetalCompiledKernel final : public CompiledKernelBase,
                                  public std::enable_shared_from_this<MetalCompiledKernel> {
public:
    explicit MetalCompiledKernel(MetalDeviceHandle device, void* pipeline, uint32_t arg_count = 0);
    MetalCompiledKernel(MetalDeviceHandle device,
                        void* pipeline,
                        std::shared_ptr<const KernelBindingPlan> binding_plan);
    MetalCompiledKernel(MetalDeviceHandle device,
                        void* pipeline,
                        std::shared_ptr<const KernelBindingPlan> binding_plan,
                        std::shared_ptr<void> prepared_binding_cache,
                        std::shared_ptr<MetalBindingSchema> binding_schema);

    size_t clamp_threadgroup_size(size_t desired) const override;
    std::shared_ptr<ICompiledKernel> fork() const override;
    void prewarm_bindings(const std::vector<KernelArg>& args) override;
    void set_mpsrt_model(std::shared_ptr<const mpsrt::MpsrtModel> model);
    void set_mpsrt_msl_source(std::string msl_source);
    const mpsrt::MpsrtModel* mpsrt_model() const;
    bool register_mpsrt_const_tensor_data(GfxMpsrtValue value,
                                          GfxMpsrtTensorAbiDesc desc,
                                          const void* data,
                                          size_t bytes,
                                          std::string* log = nullptr);
    void execute(GpuCommandBufferHandle command_buffer,
                 const KernelDispatch& dispatch,
                 const std::vector<KernelArg>& args,
                 const KernelExecutionHooks* hooks = nullptr) override;
    const void* shared_binding_schema_identity() const;

private:
    enum class MpsrtPreparedModelCacheKind {
        None,
        ContextExecution,
        SingleMslDispatch,
    };

    struct MpsrtPreparedModelCacheSlot {
        std::shared_ptr<metal::mpsrt::MpsrtPreparedModel> model;
        MpsrtPreparedModelCacheKind kind = MpsrtPreparedModelCacheKind::None;
    };

    void reset_mpsrt_prepared_model_cache();
    static const char* mpsrt_prepared_model_cache_kind_name(MpsrtPreparedModelCacheKind kind);
    MpsrtPreparedModelCacheKind resolve_mpsrt_prepared_model_cache_kind() const;
    std::shared_ptr<const metal::mpsrt::MpsrtPreparedModel> get_or_prepare_mpsrt_model(
        MpsrtPreparedModelCacheKind kind,
        std::string* error,
        bool* cache_hit,
        const KernelExecutionHooks* hooks = nullptr);

    static constexpr size_t kMpsrtPreparedModelCacheSlotCount = 3;

    MetalDeviceHandle m_device = nullptr;
    void* m_pipeline = nullptr;
    std::shared_ptr<MetalBindingSchema> m_binding_schema;
    std::shared_ptr<const mpsrt::MpsrtModel> m_mpsrt_model;
    std::shared_ptr<metal::mpsrt::MpsrtContext> m_mpsrt_context;
    std::mutex m_mpsrt_prepared_model_cache_mutex;
    std::vector<MpsrtPreparedModelCacheSlot> m_mpsrt_prepared_model_cache_slots;
    size_t m_mpsrt_prepared_model_cache_next_slot = 0;
    std::string m_mpsrt_msl_source;
};

}  // namespace gfx_plugin
}  // namespace ov
