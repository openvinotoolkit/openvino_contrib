// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#import "backends/metal/codegen/metal_codegen_backend.hpp"

#include <chrono>
#include <exception>
#include <mutex>
#include <sstream>
#include <unordered_map>
#include <vector>

#include "backends/metal/codegen/metal_compiler.hpp"
#include "mlir/mlir_passes.hpp"
#include "openvino/core/except.hpp"
#include "backends/metal/runtime/op_utils.hpp"
#include "kernel_ir/gfx_kernel_cache.hpp"
#include "runtime/gfx_compile_profiling.hpp"
#include "runtime/gfx_logger.hpp"

#include "llvm/Support/raw_ostream.h"

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

std::string make_resolved_msl_cache_key(const KernelSource& source) {
    if (!source.module) {
        return {};
    }
    std::string module_text;
    llvm::raw_string_ostream os(module_text);
    auto module = source.module;
    module.print(os);
    os.flush();

    std::ostringstream key;
    key << source.entry_point << '\n'
        << source.signature.arg_count << ':'
        << source.signature.output_arg_count << '\n'
        << module_text;
    return key.str();
}

class MetalResolvedMslCache final {
public:
    static MetalResolvedMslCache& instance() {
        static MetalResolvedMslCache cache;
        return cache;
    }

    bool lookup(const std::string& key, std::string& msl) {
        std::lock_guard<std::mutex> lock(m_mutex);
        auto it = m_cache.find(key);
        if (it == m_cache.end()) {
            return false;
        }
        msl = it->second;
        return true;
    }

    void store(std::string key, std::string msl) {
        if (key.empty() || msl.empty()) {
            return;
        }
        std::lock_guard<std::mutex> lock(m_mutex);
        m_cache.emplace(std::move(key), std::move(msl));
    }

private:
    std::mutex m_mutex;
    std::unordered_map<std::string, std::string> m_cache;
};

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
    if (gfx_log_debug_enabled()) {
        gfx_log_debug("MetalCodegen") << "compile entry=" << source.entry_point
                                       << " arg_count=" << source.signature.arg_count
                                       << " has_module=" << (source.module ? "yes" : "no")
                                       << " has_msl=" << (!source.msl_source.empty() ? "yes" : "no")
                                       << " has_generator=" << (source.msl_generator ? "yes" : "no");
    }
    std::string msl;
    std::string resolved_msl_cache_key;
    const bool can_cache_resolved_msl = source.module && source.msl_source.empty() && source.msl_generator;
    if (can_cache_resolved_msl) {
        const auto cache_key_start = current_compile_trace() ? std::chrono::steady_clock::now()
                                                             : std::chrono::steady_clock::time_point{};
        resolved_msl_cache_key = make_resolved_msl_cache_key(source);
        if (current_compile_trace()) {
            add_compile_segment(
                "metal_resolved_msl_cache_key",
                static_cast<uint64_t>(std::chrono::duration_cast<std::chrono::microseconds>(
                                          std::chrono::steady_clock::now() - cache_key_start)
                                          .count()));
        }
        if (MetalResolvedMslCache::instance().lookup(resolved_msl_cache_key, msl)) {
            if (current_compile_trace()) {
                increment_compile_counter("metal_resolved_msl_cache_hit_count");
                add_compile_segment("metal_resolved_msl_cache_hit", 0);
            }
        }
    }

    if (msl.empty() && source.module && source.msl_source.empty() && source.msl_generator) {
        const auto mlir_preprocess_start = current_compile_trace() ? std::chrono::steady_clock::now()
                                                                   : std::chrono::steady_clock::time_point{};
        try {
            if (gfx_log_debug_enabled()) {
                gfx_log_debug("MetalCodegen") << "before run_mlir_pipeline entry=" << source.entry_point;
            }
            run_mlir_pipeline(source.module, /*use_alloca=*/true, /*use_parallel_loops=*/false);
            if (gfx_log_debug_enabled()) {
                gfx_log_debug("MetalCodegen") << "after run_mlir_pipeline entry=" << source.entry_point;
            }
            if (current_compile_trace()) {
                increment_compile_counter("metal_mlir_preprocess_count");
                add_compile_segment(
                    "metal_mlir_preprocess",
                    static_cast<uint64_t>(std::chrono::duration_cast<std::chrono::microseconds>(
                                              std::chrono::steady_clock::now() - mlir_preprocess_start)
                                              .count()));
            }
        } catch (const std::exception& e) {
            if (log_ptr) {
                *log_ptr = std::string("MLIR preprocessing failed: ") + e.what();
            }
            return nullptr;
        }
    }
    if (msl.empty()) {
        const auto resolve_msl_start =
            current_compile_trace() ? std::chrono::steady_clock::now() : std::chrono::steady_clock::time_point{};
        if (gfx_log_debug_enabled()) {
            gfx_log_debug("MetalCodegen") << "before resolve_msl_source entry=" << source.entry_point;
        }
        msl = resolve_msl_source(source, log_ptr);
        if (gfx_log_debug_enabled()) {
            gfx_log_debug("MetalCodegen") << "after resolve_msl_source entry=" << source.entry_point
                                           << " msl_size=" << msl.size();
        }
        if (current_compile_trace()) {
            increment_compile_counter("metal_resolve_msl_count");
            add_compile_segment(
                "metal_resolve_msl",
                static_cast<uint64_t>(std::chrono::duration_cast<std::chrono::microseconds>(
                                          std::chrono::steady_clock::now() - resolve_msl_start)
                                          .count()));
        }
        if (can_cache_resolved_msl && !resolved_msl_cache_key.empty()) {
            MetalResolvedMslCache::instance().store(std::move(resolved_msl_cache_key), msl);
        }
    }
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
                                        const auto backend_compile_start = current_compile_trace()
                                                                               ? std::chrono::steady_clock::now()
                                                                               : std::chrono::steady_clock::time_point{};
                                        id<MTLComputePipelineState> pipeline =
                                            compiler.compile_msl_from_source(msl,
                                                                             source.entry_point.c_str(),
                                                                             compile_log);
                                        if (current_compile_trace()) {
                                            increment_compile_counter("metal_backend_compile_count");
                                            add_compile_segment(
                                                "metal_backend_compile",
                                                static_cast<uint64_t>(std::chrono::duration_cast<std::chrono::microseconds>(
                                                                          std::chrono::steady_clock::now() -
                                                                          backend_compile_start)
                                                                          .count()));
                                        }
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

void MetalCompiledKernel::prewarm_bindings(const std::vector<KernelArg>& args) {
    auto prepared_base = get_or_create_prepared_bindings(args, "MetalCompiledKernel prewarm");
    (void)prepared_base->get_or_create_backend_state<MetalPreparedState>(
        reinterpret_cast<uintptr_t>(m_binding_schema.get() ? m_binding_schema.get() : m_device),
        [&]() {
            return std::make_shared<MetalPreparedState>(prepared_base->binding_table());
        });
}

void MetalCompiledKernel::execute(GpuCommandBufferHandle command_buffer,
                                  const KernelDispatch& dispatch,
                                  const std::vector<KernelArg>& args,
                                  const KernelExecutionHooks* hooks) {
    id<MTLCommandBuffer> cb = static_cast<id<MTLCommandBuffer>>(command_buffer);
    OPENVINO_ASSERT(cb, "MetalCompiledKernel: command buffer is null");
    OPENVINO_ASSERT(m_pipeline, "MetalCompiledKernel: pipeline is null");
    auto prepared_base = get_or_create_prepared_bindings(args, "MetalCompiledKernel");
    const bool trace_bindings = hooks && (hooks->on_segment || hooks->on_counter);
    const auto binding_start = trace_bindings ? std::chrono::steady_clock::now() : std::chrono::steady_clock::time_point{};
    bool prepared_state_created = false;
    auto prepared = prepared_base->get_or_create_backend_state<MetalPreparedState>(
        reinterpret_cast<uintptr_t>(m_binding_schema.get() ? m_binding_schema.get() : m_device),
        [&]() {
            prepared_state_created = true;
            return std::make_shared<MetalPreparedState>(prepared_base->binding_table());
        });
    if (trace_bindings && prepared_state_created) {
        const auto binding_cpu_us = std::chrono::duration_cast<std::chrono::microseconds>(
            std::chrono::steady_clock::now() - binding_start);
        if (hooks->on_counter) {
            hooks->on_counter("binding_prepare_count", 1);
            hooks->on_counter("buffer_bind_count", static_cast<uint64_t>(prepared->buffers.size()));
        }
        if (hooks->on_segment) {
            hooks->on_segment("binding_prepare",
                              "metal_prepared_state",
                              binding_cpu_us,
                              0,
                              0,
                              0,
                              0,
                              0,
                              0,
                              -1,
                              0,
                              reinterpret_cast<uint64_t>(cb));
        }
    } else if (hooks && hooks->on_counter) {
        hooks->on_counter("prepared_binding_cache_hit_count", 1);
    }

    const bool trace_encoder_setup = hooks && (hooks->on_segment || hooks->on_counter);
    const auto encoder_setup_start =
        trace_encoder_setup ? std::chrono::steady_clock::now() : std::chrono::steady_clock::time_point{};
    id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];
    const auto pipeline_bind_start =
        trace_encoder_setup ? std::chrono::steady_clock::now() : std::chrono::steady_clock::time_point{};
    [enc setComputePipelineState:(id<MTLComputePipelineState>)m_pipeline];
    const auto after_pipeline_bind =
        trace_encoder_setup ? std::chrono::steady_clock::now() : std::chrono::steady_clock::time_point{};

    const auto buffer_bind_start =
        trace_encoder_setup ? std::chrono::steady_clock::now() : std::chrono::steady_clock::time_point{};
    for (size_t index = 0; index < prepared->buffers.size(); ++index) {
        [enc setBuffer:prepared->buffers[index] offset:prepared->offsets[index] atIndex:index];
    }
    if (trace_encoder_setup) {
        const auto encoder_cpu_us = std::chrono::duration_cast<std::chrono::microseconds>(
            std::chrono::steady_clock::now() - encoder_setup_start);
        const auto pipeline_bind_cpu_us =
            std::chrono::duration_cast<std::chrono::microseconds>(after_pipeline_bind - pipeline_bind_start);
        const auto buffer_bind_cpu_us =
            std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::steady_clock::now() - buffer_bind_start);
        if (hooks->on_counter) {
            hooks->on_counter("encoder_setup_count", 1);
            hooks->on_counter("pipeline_bind_count", 1);
        }
        if (hooks->on_segment) {
            hooks->on_segment("descriptor_update",
                              "metal_pipeline_bind",
                              pipeline_bind_cpu_us,
                              0,
                              0,
                              0,
                              0,
                              0,
                              0,
                              -1,
                              0,
                              reinterpret_cast<uint64_t>(cb));
            hooks->on_segment("descriptor_update",
                              "metal_buffer_bind",
                              buffer_bind_cpu_us,
                              0,
                              static_cast<uint32_t>(prepared->buffers.size()),
                              0,
                              0,
                              0,
                              0,
                              -1,
                              0,
                              reinterpret_cast<uint64_t>(cb));
            if (encoder_cpu_us > pipeline_bind_cpu_us + buffer_bind_cpu_us) {
                hooks->on_segment("descriptor_update",
                                  "metal_encoder_setup_overhead",
                                  encoder_cpu_us - pipeline_bind_cpu_us - buffer_bind_cpu_us,
                                  0,
                                  0,
                                  0,
                                  0,
                                  0,
                                  0,
                                  -1,
                                  0,
                                  reinterpret_cast<uint64_t>(cb));
            }
        }
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
