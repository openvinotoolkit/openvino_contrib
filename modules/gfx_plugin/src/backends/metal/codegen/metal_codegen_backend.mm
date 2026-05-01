// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#import "backends/metal/codegen/metal_codegen_backend.hpp"

#include <algorithm>
#include <chrono>
#include <exception>
#include <mutex>
#include <sstream>
#include <unordered_map>
#include <vector>

#include "backends/metal/codegen/metal_compiler.hpp"
#include "backends/metal/runtime/metal_command_encoder.hpp"
#include "backends/metal/runtime/mpsrt/mpsrt_request.hpp"
#include "mlir/mlir_passes.hpp"
#include "openvino/core/except.hpp"
#include "backends/metal/runtime/op_utils.hpp"
#include "kernel_ir/gfx_kernel_cache.hpp"
#include "mlir/gfx_mpsrt_metadata.hpp"
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

uint32_t resolve_kernel_output_arg_count(const KernelSource& source) {
    if (source.signature.output_arg_count != 0) {
        return source.signature.output_arg_count;
    }
    if (!source.module) {
        return 0;
    }
    if (auto attr = source.module->getAttrOfType<mlir::IntegerAttr>("gfx.kernel_output_arg_count")) {
        return static_cast<uint32_t>(std::max<int64_t>(attr.getInt(), 0));
    }
    return 0;
}

bool set_error(std::string* error, const std::string& message) {
    if (error) {
        *error = message;
    }
    return false;
}

bool make_mpsrt_external_io_bindings(const metal::mpsrt::MpsrtModel& model,
                                     const std::vector<void*>& buffer_ptrs,
                                     const std::vector<size_t>& offsets,
                                     uint32_t output_arg_count,
                                     std::vector<metal::mpsrt::MpsrtBoundBuffer>& input_buffers,
                                     std::vector<metal::mpsrt::MpsrtBoundBuffer>& output_buffers,
                                     std::string* error) {
    const auto bound_buffers = metal::mpsrt::make_mpsrt_bound_buffers(buffer_ptrs, offsets);
    const size_t input_count = model.input_values.size();
    const size_t output_count = model.output_values.size();
    if (bound_buffers.size() < input_count) {
        return set_error(error, "GFX MPSRT: runtime buffers do not cover model inputs");
    }

    input_buffers.assign(bound_buffers.begin(), bound_buffers.begin() + input_count);
    output_buffers.clear();
    if (output_count == 0) {
        return true;
    }

    if (output_arg_count != 0) {
        if (output_arg_count != output_count) {
            return set_error(error, "GFX MPSRT: output arg count does not match model outputs");
        }
        if (bound_buffers.size() < output_count) {
            return set_error(error, "GFX MPSRT: runtime buffers do not cover model outputs");
        }
        output_buffers.assign(bound_buffers.end() - output_count, bound_buffers.end());
        return true;
    }

    if (bound_buffers.size() >= input_count + output_count) {
        output_buffers.assign(bound_buffers.begin() + input_count,
                              bound_buffers.begin() + input_count + output_count);
        return true;
    }

    if (bound_buffers.size() == input_count && input_count == output_count) {
        output_buffers = input_buffers;
        return true;
    }

    return set_error(error, "GFX MPSRT: cannot infer model output bindings from runtime buffers");
}

void record_mpsrt_plan_counters(mlir::ModuleOp module) {
    if (!module || !current_compile_trace()) {
        return;
    }

    GfxMpsrtModuleBuilderPlan module_plan;
    if (!build_module_mpsrt_builder_plan(module, module_plan)) {
        return;
    }
    const auto& plan = module_plan.stage_plan;
    const auto& builder_plan = module_plan.builder_plan;

    switch (plan.stage.domain) {
        case GfxStageBackendDomain::AppleMps:
            increment_compile_counter("mpsrt_plan_apple_mps_count");
            break;
        case GfxStageBackendDomain::AppleMsl:
            increment_compile_counter("mpsrt_plan_apple_msl_count");
            break;
        case GfxStageBackendDomain::Spirv:
            increment_compile_counter("mpsrt_plan_spirv_count");
            break;
        case GfxStageBackendDomain::Unknown:
        default:
            break;
    }
    increment_compile_counter(std::string("mpsrt_stage_kind_") + gfx_mpsrt_stage_kind_name(plan.stage.kind));
    increment_compile_counter(std::string("mpsrt_builder_symbol_") + plan.stage.builder_symbol + "_count");
    increment_compile_counter("mpsrt_builder_record_count", static_cast<uint64_t>(builder_plan.records.size()));
    increment_compile_counter("mpsrt_builder_encode_record_count");
    if (!plan.stage.dispatch_kernel_family.empty()) {
        increment_compile_counter(std::string("mpsrt_dispatch_family_") +
                                  plan.stage.dispatch_kernel_family +
                                  "_count");
    }
    if (plan.stage.dispatch_kernel_family_id != 0) {
        increment_compile_counter(std::string("mpsrt_dispatch_family_id_") +
                                  std::to_string(plan.stage.dispatch_kernel_family_id) +
                                  "_count");
    }
    if (!plan.stage.dispatch_entry_point.empty()) {
        increment_compile_counter(std::string("mpsrt_dispatch_entry_") +
                                  plan.stage.dispatch_entry_point +
                                  "_count");
    }
    if (plan.stage.dispatch_threads_per_threadgroup != 0) {
        increment_compile_counter(std::string("mpsrt_dispatch_tg_") +
                                  std::to_string(plan.stage.dispatch_threads_per_threadgroup) +
                                  "_count");
    }
    if (plan.stage.dispatch_flags != GfxMpsrtMslDispatchFlagNone) {
        increment_compile_counter("mpsrt_dispatch_flags",
                                  static_cast<uint64_t>(plan.stage.dispatch_flags));
    }
    if (plan.stage.dispatch_precompiled_kernel_required) {
        increment_compile_counter("mpsrt_dispatch_precompiled_kernel_required_count");
    }
    for (const auto& record : builder_plan.records) {
        if (record.stage_kind == GfxMpsrtStageKind::MSLDispatch &&
            record.msl_dispatch_desc.kernel_family != 0) {
            increment_compile_counter("mpsrt_msl_dispatch_descriptor_count");
            increment_compile_counter(std::string("mpsrt_msl_dispatch_descriptor_family_id_") +
                                      std::to_string(record.msl_dispatch_desc.kernel_family) +
                                      "_count");
        }
    }
    increment_compile_counter(std::string("mpsrt_storage_") +
                              gfx_mpsrt_storage_name(plan.stage.output_storage) +
                              "_count");
    if (plan.stage.uses_vendor_primitive) {
        increment_compile_counter("mpsrt_vendor_primitive_stage_count");
    }
    if (plan.stage.uses_custom_kernel) {
        increment_compile_counter("mpsrt_custom_kernel_stage_count");
    }
    uint64_t input_bytes = 0;
    uint64_t output_bytes = 0;
    for (const auto& desc : plan.inputs) {
        input_bytes += desc.byte_length;
    }
    for (const auto& desc : plan.outputs) {
        output_bytes += desc.byte_length;
    }
    increment_compile_counter("mpsrt_input_descriptor_count", static_cast<uint64_t>(plan.inputs.size()));
    increment_compile_counter("mpsrt_output_descriptor_count", static_cast<uint64_t>(plan.outputs.size()));
    increment_compile_counter("mpsrt_input_byte_length", input_bytes);
    increment_compile_counter("mpsrt_output_byte_length", output_bytes);
}

std::shared_ptr<const metal::mpsrt::MpsrtModel> build_metal_mpsrt_runtime_model(mlir::ModuleOp module,
                                                                               uint32_t arg_count,
                                                                               uint32_t output_arg_count) {
    if (!module) {
        return nullptr;
    }

    GfxMpsrtModuleBuilderPlan module_plan;
    if (!build_module_mpsrt_builder_plan(module, module_plan)) {
        return nullptr;
    }

    metal::mpsrt::MpsrtModel model;
    std::string error;
    if (!metal::mpsrt::build_mpsrt_model_from_builder_plan(module_plan.builder_plan, model, &error)) {
        OPENVINO_THROW("GFX Metal MPSRT: failed to build runtime model: ", error);
    }
    const uint32_t mpsrt_arg_count = module_plan.builder_plan.external_buffer_count != 0
                                         ? module_plan.builder_plan.external_buffer_count
                                         : arg_count;
    const uint32_t mpsrt_output_arg_count = module_plan.builder_plan.external_buffer_abi_valid
                                                ? module_plan.builder_plan.external_output_buffer_count
                                                : output_arg_count;
    if (!metal::mpsrt::adapt_mpsrt_model_to_external_buffer_abi(model,
                                                                mpsrt_arg_count,
                                                                mpsrt_output_arg_count,
                                                                &error)) {
        OPENVINO_THROW("GFX Metal MPSRT: failed to adapt runtime model ABI: ", error);
    }

    if (current_compile_trace()) {
        increment_compile_counter("mpsrt_runtime_model_prepare_count");
        if (module_plan.builder_plan.external_buffer_abi_valid) {
            increment_compile_counter("mpsrt_runtime_model_mlir_external_buffer_abi_count");
        }
        increment_compile_counter("mpsrt_runtime_model_stage_count", static_cast<uint64_t>(model.stages.size()));
        increment_compile_counter("mpsrt_runtime_model_tensor_count", static_cast<uint64_t>(model.tensors.size()));
        for (const auto& stage : model.stages) {
            increment_compile_counter(std::string("mpsrt_runtime_model_stage_kind_") +
                                      gfx_mpsrt_stage_kind_name(stage.kind) +
                                      "_count");
            if (stage.kind == GfxMpsrtStageKind::MSLDispatch) {
                increment_compile_counter("mpsrt_runtime_model_msl_dispatch_stage_count");
            }
        }
    }

    return std::make_shared<metal::mpsrt::MpsrtModel>(std::move(model));
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
        buffer_ptrs.reserve(bindings.size());
        offsets.reserve(bindings.size());
        for (const auto& binding : bindings) {
            auto* buffer = to_mtl(binding.buffer);
            buffers.push_back(buffer);
            buffer_ptrs.push_back(buffer);
            offsets.push_back(binding.offset);
        }
    }

    std::vector<id<MTLBuffer>> buffers;
    std::vector<void*> buffer_ptrs;
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
    record_mpsrt_plan_counters(source.module);
    const uint32_t arg_count = source.signature.arg_count;
    const uint32_t output_arg_count = resolve_kernel_output_arg_count(source);
    auto mpsrt_model = build_metal_mpsrt_runtime_model(source.module, arg_count, output_arg_count);
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

    const uintptr_t device_key = reinterpret_cast<uintptr_t>(m_device);
    auto shared_prepared_cache = acquire_shared_prepared_binding_cache(GpuBackend::Metal, device_key, arg_count);
    auto binding_schema = m_reuse_context->acquire_binding_schema(arg_count);
    auto kernel = lookup_or_compile_kernel(GpuBackend::Metal,
                                           device_key,
                                           msl.data(),
                                           msl.size(),
                                           source.entry_point,
                                           arg_count,
                                           [&]() -> std::shared_ptr<ICompiledKernel> {
                                               MetalKernelCompiler compiler((id<MTLDevice>)m_device);
                                               std::string local_log;
                                               std::string& compile_log = log ? *log : local_log;
                                               const auto backend_compile_start =
                                                   current_compile_trace()
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
                                                       static_cast<uint64_t>(
                                                           std::chrono::duration_cast<std::chrono::microseconds>(
                                                               std::chrono::steady_clock::now() -
                                                               backend_compile_start)
                                                               .count()));
                                               }
                                               if (!pipeline) {
                                                   return nullptr;
                                               }
                                               auto binding_plan = std::make_shared<KernelBindingPlan>(
                                                   arg_count,
                                                   output_arg_count);
                                               return std::make_shared<MetalCompiledKernel>(m_device,
                                                                                            (void*)pipeline,
                                                                                            std::move(binding_plan),
                                                                                            shared_prepared_cache,
                                                                                            binding_schema);
                                           });
    if (auto metal_kernel = std::dynamic_pointer_cast<MetalCompiledKernel>(kernel)) {
        metal_kernel->set_mpsrt_model(std::move(mpsrt_model));
    }
    return kernel;
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
    auto kernel = std::make_shared<MetalCompiledKernel>(m_device,
                                                        m_pipeline,
                                                        binding_plan(),
                                                        prepared_binding_cache(),
                                                        m_binding_schema);
    kernel->set_mpsrt_model(m_mpsrt_model);
    return kernel;
}

const void* MetalCompiledKernel::shared_binding_schema_identity() const {
    return m_binding_schema.get();
}

void MetalCompiledKernel::set_mpsrt_model(std::shared_ptr<const metal::mpsrt::MpsrtModel> model) {
    m_mpsrt_model = std::move(model);
}

const metal::mpsrt::MpsrtModel* MetalCompiledKernel::mpsrt_model() const {
    return m_mpsrt_model.get();
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

    if (m_mpsrt_model && m_mpsrt_model->stages.size() == 1 &&
        m_mpsrt_model->stages.front().kind == GfxMpsrtStageKind::MSLDispatch) {
        const auto prepared_mpsrt =
            metal::mpsrt::make_prepared_msl_dispatch_from_pipeline(m_mpsrt_model->stages.front(),
                                                                    0,
                                                                    static_cast<id<MTLComputePipelineState>>(m_pipeline));
        std::string mpsrt_error;
        std::vector<metal::mpsrt::MpsrtBoundBuffer> input_buffers;
        std::vector<metal::mpsrt::MpsrtBoundBuffer> output_buffers;
        const auto external_buffers = metal::mpsrt::make_mpsrt_bound_buffers(prepared->buffer_ptrs,
                                                                             prepared->offsets);

        std::vector<id<MTLBuffer>> transient_buffers;
        auto transient_allocator = [&](const metal::mpsrt::MpsrtRuntimeTensor& tensor) {
            const auto byte_length = static_cast<NSUInteger>(tensor.desc.byte_length);
            if (byte_length == 0) {
                return metal::mpsrt::MpsrtBoundBuffer{};
            }
            id<MTLBuffer> buffer =
                [static_cast<id<MTLDevice>>(m_device) newBufferWithLength:byte_length
                                                                  options:MTLResourceStorageModePrivate];
            transient_buffers.push_back(buffer);
            return metal::mpsrt::MpsrtBoundBuffer{(__bridge void*)buffer,
                                                  static_cast<size_t>(tensor.desc.byte_offset)};
        };
        metal::mpsrt::MpsrtTensorBindings bindings;
        metal::mpsrt::MpsrtBindingBuildResult binding_result;
        bool bindings_built = false;
        if (external_buffers.size() == m_mpsrt_model->external_values.size()) {
            bindings_built =
                metal::mpsrt::build_mpsrt_external_tensor_bindings(*m_mpsrt_model,
                                                                   external_buffers,
                                                                   transient_allocator,
                                                                   bindings,
                                                                   &binding_result,
                                                                   &mpsrt_error);
        } else {
            const bool external_bound =
                make_mpsrt_external_io_bindings(*m_mpsrt_model,
                                                prepared->buffer_ptrs,
                                                prepared->offsets,
                                                binding_plan()->output_arg_count(),
                                                input_buffers,
                                                output_buffers,
                                                &mpsrt_error);
            OPENVINO_ASSERT(external_bound, mpsrt_error);
            bindings_built =
                metal::mpsrt::build_mpsrt_tensor_bindings(*m_mpsrt_model,
                                                          input_buffers,
                                                          output_buffers,
                                                          transient_allocator,
                                                          bindings,
                                                          &binding_result,
                                                          &mpsrt_error);
        }
        OPENVINO_ASSERT(bindings_built, mpsrt_error);
        if (hooks && hooks->on_counter) {
            hooks->on_counter("mpsrt_binding_external_input_count",
                              static_cast<uint64_t>(binding_result.external_inputs_bound));
            hooks->on_counter("mpsrt_binding_external_output_count",
                              static_cast<uint64_t>(binding_result.external_outputs_bound));
            hooks->on_counter("mpsrt_binding_transient_alloc_count",
                              static_cast<uint64_t>(binding_result.transient_buffers_allocated));
        }

        metal::mpsrt::MpsrtPreparedModel prepared_model;
        prepared_model.msl_dispatches.push_back(prepared_mpsrt);
        std::vector<KernelDispatch> stage_dispatches = {dispatch};
        metal::mpsrt::MpsrtRequest request;
        metal::mpsrt::MpsrtModelEncodeResult encode_result;
        const bool encoded =
            request.encode_prepared_model(command_buffer,
                                          *m_mpsrt_model,
                                          prepared_model,
                                          stage_dispatches,
                                          bindings,
                                          hooks,
                                          &encode_result,
                                          &mpsrt_error);
        OPENVINO_ASSERT(encoded, mpsrt_error);
        return;
    }

    const bool trace_encoder_setup = hooks && (hooks->on_segment || hooks->on_counter);
    const auto encoder_setup_start =
        trace_encoder_setup ? std::chrono::steady_clock::now() : std::chrono::steady_clock::time_point{};
    bool encoder_created = false;
    id<MTLComputeCommandEncoder> enc =
        static_cast<id<MTLComputeCommandEncoder>>(metal_get_or_create_compute_encoder(command_buffer, &encoder_created));
    OPENVINO_ASSERT(enc, "MetalCompiledKernel: failed to create compute encoder");
    const auto pipeline_bind_start =
        trace_encoder_setup ? std::chrono::steady_clock::now() : std::chrono::steady_clock::time_point{};
    const bool pipeline_bound =
        metal_set_compute_pipeline_if_needed(command_buffer,
                                             reinterpret_cast<GpuCommandEncoderHandle>(enc),
                                             m_pipeline);
    const auto after_pipeline_bind =
        trace_encoder_setup ? std::chrono::steady_clock::now() : std::chrono::steady_clock::time_point{};

    const auto buffer_bind_start =
        trace_encoder_setup ? std::chrono::steady_clock::now() : std::chrono::steady_clock::time_point{};
    const size_t bound_buffers =
        metal_bind_compute_buffers_if_needed(command_buffer,
                                             reinterpret_cast<GpuCommandEncoderHandle>(enc),
                                             prepared->buffer_ptrs,
                                             prepared->offsets);
    if (trace_encoder_setup) {
        const auto encoder_cpu_us = std::chrono::duration_cast<std::chrono::microseconds>(
            std::chrono::steady_clock::now() - encoder_setup_start);
        const auto pipeline_bind_cpu_us =
            std::chrono::duration_cast<std::chrono::microseconds>(after_pipeline_bind - pipeline_bind_start);
        const auto buffer_bind_cpu_us =
            std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::steady_clock::now() - buffer_bind_start);
        if (hooks->on_counter) {
            if (encoder_created) {
                hooks->on_counter("encoder_setup_count", 1);
            }
            if (pipeline_bound) {
                hooks->on_counter("pipeline_bind_count", 1);
            }
            hooks->on_counter("buffer_bind_count", static_cast<uint64_t>(bound_buffers));
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
                              static_cast<uint32_t>(bound_buffers),
                              0,
                              0,
                              0,
                              0,
                              -1,
                              0,
                              reinterpret_cast<uint64_t>(cb));
            if (encoder_created && encoder_cpu_us > pipeline_bind_cpu_us + buffer_bind_cpu_us) {
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
}

}  // namespace gfx_plugin
}  // namespace ov
