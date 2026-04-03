// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//
#pragma once

#include <functional>
#include <string>

#include "openvino/core/node.hpp"
#include "openvino/core/shape.hpp"
#include "kernel_ir/gfx_codegen_backend.hpp"
#include "kernel_ir/gfx_kernel_signature.hpp"
#include "kernel_ir/gfx_kernel_dispatch.hpp"
#include "mlir/gfx_mlir_kernel_metadata.hpp"

namespace ov {
namespace gfx_plugin {

uint32_t infer_kernel_arg_count(mlir::ModuleOp module, const std::string& entry_point);

class KernelPlan {
public:
    KernelPlan(mlir::ModuleOp module, std::string entry_point, uint32_t arg_count)
        : m_module(module),
          m_entry_point(std::move(entry_point)),
          m_arg_count(arg_count) {}

    mlir::ModuleOp module() const { return m_module; }
    const std::string& entry_point() const { return m_entry_point; }
    uint32_t arg_count() const { return m_arg_count; }

    KernelSource to_source() const {
        const uint32_t inferred = m_arg_count ? m_arg_count : infer_kernel_arg_count(m_module, m_entry_point);
        return make_kernel_source_from_mlir(m_module, m_entry_point, inferred);
    }

    KernelSource to_source_with_msl(std::string msl_source) const {
        KernelSource src = to_source();
        src.msl_source = std::move(msl_source);
        return src;
    }

    KernelSource to_source_with_msl_generator(std::function<std::string(mlir::ModuleOp)> generator) const {
        KernelSource src = to_source();
        src.msl_generator = std::move(generator);
        return src;
    }

    KernelSource to_source_with_spirv(std::vector<uint32_t> spirv_binary) const {
        KernelSource src = to_source();
        src.spirv_binary = std::move(spirv_binary);
        return src;
    }

    KernelSource to_source_with_spirv_generator(
        std::function<std::vector<uint32_t>(mlir::ModuleOp)> generator) const {
        KernelSource src = to_source();
        src.spirv_generator = std::move(generator);
        return src;
    }

    KernelRuntimeMetadata runtime_metadata(const KernelArgMappingInfo& mapping,
                                           const std::shared_ptr<const ov::Node>& node,
                                           size_t outputs_hint = 0) const {
        return extract_kernel_runtime_metadata(m_module, mapping, node, outputs_hint);
    }

    static KernelDispatch make_default_dispatch(const ov::Shape& shape,
                                                const ICompiledKernel& kernel) {
        // Linear MLIR kernels are indexed by the global invocation id, so using
        // a reasonable workgroup size reduces dispatch group counts without
        // changing semantics. This is required for large mobile-Vulkan tensors
        // where a 1-thread group would overflow practical dispatch limits.
        constexpr size_t kDefaultLinearThreadsPerGroup = 64;
        return gfx_plugin::make_default_dispatch(shape,
                                                 kernel.clamp_threadgroup_size(kDefaultLinearThreadsPerGroup));
    }

private:
    mlir::ModuleOp m_module;
    std::string m_entry_point;
    uint32_t m_arg_count = 0;
};

inline uint32_t infer_kernel_arg_count(mlir::ModuleOp module, const std::string& entry_point) {
    const auto sig = infer_kernel_signature(module, entry_point);
    const size_t fallback = static_cast<size_t>(sig.total());
    return static_cast<uint32_t>(infer_kernel_arg_count_from_module(module, fallback));
}

}  // namespace gfx_plugin
}  // namespace ov
