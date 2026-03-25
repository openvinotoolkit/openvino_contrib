// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#import "backends/metal/codegen/metal_codegen_backend.hpp"

#include <exception>
#include <mutex>
#include <unordered_map>
#include <vector>

#include "backends/metal/codegen/metal_compiler.hpp"
#include "mlir/mlir_passes.hpp"
#include "openvino/core/except.hpp"
#include "backends/metal/runtime/op_utils.hpp"
#include "kernel_ir/gfx_kernel_cache.hpp"

namespace ov {
namespace gfx_plugin {

class MetalBindingSchema final {
public:
    explicit MetalBindingSchema(uint32_t arg_count) : m_arg_count(arg_count) {}

    uint32_t arg_count() const {
        return m_arg_count;
    }

private:
    uint32_t m_arg_count = 0;
};

class MetalDeviceReuseContext final {
public:
    explicit MetalDeviceReuseContext(MetalDeviceHandle device) : m_device(device) {}

    std::shared_ptr<MetalBindingSchema> acquire_binding_schema(uint32_t arg_count) {
        std::lock_guard<std::mutex> lock(m_mutex);
        if (auto it = m_binding_schemas.find(arg_count); it != m_binding_schemas.end()) {
            if (auto schema = it->second.lock()) {
                return schema;
            }
        }
        auto schema = std::make_shared<MetalBindingSchema>(arg_count);
        m_binding_schemas[arg_count] = schema;
        return schema;
    }

private:
    MetalDeviceHandle m_device = nullptr;
    std::mutex m_mutex;
    std::unordered_map<uint32_t, std::weak_ptr<MetalBindingSchema>> m_binding_schemas;
};

class MetalDeviceReuseRegistry final {
public:
    static MetalDeviceReuseRegistry& instance() {
        static MetalDeviceReuseRegistry registry;
        return registry;
    }

    std::shared_ptr<MetalDeviceReuseContext> acquire(MetalDeviceHandle device) {
        std::lock_guard<std::mutex> lock(m_mutex);
        const auto key = reinterpret_cast<uintptr_t>(device);
        if (auto it = m_contexts.find(key); it != m_contexts.end()) {
            if (auto context = it->second.lock()) {
                return context;
            }
        }
        auto context = std::make_shared<MetalDeviceReuseContext>(device);
        m_contexts[key] = context;
        return context;
    }

private:
    std::mutex m_mutex;
    std::unordered_map<uintptr_t, std::weak_ptr<MetalDeviceReuseContext>> m_contexts;
};

namespace {
inline id<MTLBuffer> to_mtl(const GpuBuffer& buf) {
    return (__bridge id<MTLBuffer>)buf.buffer;
}

class MetalPreparedState final {
public:
    explicit MetalPreparedState(const KernelBindingTable& table) {
        const auto& bindings = table.buffers;
        buffers.reserve(bindings.size());
        offsets.reserve(bindings.size());
        for (const auto& binding : bindings) {
            buffers.push_back(to_mtl(binding.buffer));
            offsets.push_back(binding.offset);
        }
    }

    std::vector<id<MTLBuffer>> buffers;
    std::vector<size_t> offsets;
};
}  // namespace

MetalCodegenBackend::MetalCodegenBackend(MetalDeviceHandle device)
    : m_device(device),
      m_reuse_context(MetalDeviceReuseRegistry::instance().acquire(device)) {}

std::shared_ptr<ICompiledKernel> MetalCodegenBackend::compile(const KernelSource& source,
                                                              std::string* log) {
    std::string local_log;
    std::string* log_ptr = log ? log : &local_log;
    if (source.module && source.msl_source.empty() && source.msl_generator) {
        try {
            run_mlir_pipeline(source.module, /*use_alloca=*/true, /*use_parallel_loops=*/false);
        } catch (const std::exception& e) {
            if (log_ptr) {
                *log_ptr = std::string("MLIR preprocessing failed: ") + e.what();
            }
            return nullptr;
        }
    }
    std::string msl = resolve_msl_source(source, log_ptr);
    OPENVINO_ASSERT(!msl.empty(), "MetalCodegenBackend: missing MSL source");
    OPENVINO_ASSERT(!source.entry_point.empty(), "MetalCodegenBackend: missing entry point");

    const uint32_t arg_count = source.signature.arg_count;
    const uintptr_t device_key = reinterpret_cast<uintptr_t>(m_device);
    auto shared_prepared_cache = acquire_shared_prepared_binding_cache(GpuBackend::Metal, device_key, arg_count);
    auto binding_schema = m_reuse_context->acquire_binding_schema(arg_count);
    return lookup_or_compile_kernel(GpuBackend::Metal,
                                    device_key,
                                    msl.data(),
                                    msl.size(),
                                    source.entry_point,
                                    arg_count,
                                    [&]() -> std::shared_ptr<ICompiledKernel> {
                                        MetalKernelCompiler compiler((id<MTLDevice>)m_device);
                                        std::string local_log;
                                        std::string& compile_log = log ? *log : local_log;
                                        id<MTLComputePipelineState> pipeline =
                                            compiler.compile_msl_from_source(msl,
                                                                             source.entry_point.c_str(),
                                                                             compile_log);
                                        if (!pipeline) {
                                            return nullptr;
                                        }
                                        auto binding_plan = std::make_shared<KernelBindingPlan>(arg_count);
                                        return std::make_shared<MetalCompiledKernel>(m_device,
                                                                                     (void*)pipeline,
                                                                                     std::move(binding_plan),
                                                                                     shared_prepared_cache,
                                                                                     binding_schema);
                                    });
}

MetalCompiledKernel::MetalCompiledKernel(MetalDeviceHandle device, void* pipeline, uint32_t arg_count)
    : CompiledKernelBase(arg_count), m_device(device), m_pipeline(pipeline) {}

MetalCompiledKernel::MetalCompiledKernel(MetalDeviceHandle device,
                                         void* pipeline,
                                         std::shared_ptr<const KernelBindingPlan> binding_plan)
    : CompiledKernelBase(std::move(binding_plan)), m_device(device), m_pipeline(pipeline) {}

MetalCompiledKernel::MetalCompiledKernel(MetalDeviceHandle device,
                                         void* pipeline,
                                         std::shared_ptr<const KernelBindingPlan> binding_plan,
                                         std::shared_ptr<void> prepared_binding_cache,
                                         std::shared_ptr<MetalBindingSchema> binding_schema)
    : CompiledKernelBase(std::move(binding_plan), std::move(prepared_binding_cache)),
      m_device(device),
      m_pipeline(pipeline),
      m_binding_schema(std::move(binding_schema)) {}

size_t MetalCompiledKernel::clamp_threadgroup_size(size_t desired) const {
    return metal_clamp_tg_size(m_pipeline, desired);
}

std::shared_ptr<ICompiledKernel> MetalCompiledKernel::fork() const {
    return std::make_shared<MetalCompiledKernel>(m_device,
                                                 m_pipeline,
                                                 binding_plan(),
                                                 prepared_binding_cache(),
                                                 m_binding_schema);
}

const void* MetalCompiledKernel::shared_binding_schema_identity() const {
    return m_binding_schema.get();
}

void MetalCompiledKernel::execute(GpuCommandBufferHandle command_buffer,
                                  const KernelDispatch& dispatch,
                                  const std::vector<KernelArg>& args,
                                  const KernelExecutionHooks* hooks) {
    id<MTLCommandBuffer> cb = static_cast<id<MTLCommandBuffer>>(command_buffer);
    OPENVINO_ASSERT(cb, "MetalCompiledKernel: command buffer is null");
    OPENVINO_ASSERT(m_pipeline, "MetalCompiledKernel: pipeline is null");
    auto prepared_base = get_or_create_prepared_bindings(args, "MetalCompiledKernel");
    auto prepared = prepared_base->get_or_create_backend_state<MetalPreparedState>(
        reinterpret_cast<uintptr_t>(m_binding_schema.get() ? m_binding_schema.get() : m_device),
        [&]() {
            return std::make_shared<MetalPreparedState>(prepared_base->binding_table());
        });

    id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];
    [enc setComputePipelineState:(id<MTLComputePipelineState>)m_pipeline];

    for (size_t index = 0; index < prepared->buffers.size(); ++index) {
        [enc setBuffer:prepared->buffers[index] offset:prepared->offsets[index] atIndex:index];
    }

    if (hooks && hooks->on_begin) {
        hooks->on_begin(enc);
    }

    const size_t grid_x = dispatch.grid[0];
    const size_t grid_y = dispatch.grid[1];
    const size_t grid_z = dispatch.grid[2];
    if (grid_x == 0 || grid_y == 0 || grid_z == 0) {
        if (hooks && hooks->on_end) {
            hooks->on_end(enc);
        }
        [enc endEncoding];
        return;
    }

    MTLSize grid = MTLSizeMake(grid_x, grid_y, grid_z);
    MTLSize tg = MTLSizeMake(dispatch.threads_per_group[0],
                             dispatch.threads_per_group[1],
                             dispatch.threads_per_group[2]);
    [enc dispatchThreads:grid threadsPerThreadgroup:tg];

    if (hooks && hooks->on_end) {
        hooks->on_end(enc);
    }
    [enc endEncoding];
}

}  // namespace gfx_plugin
}  // namespace ov
