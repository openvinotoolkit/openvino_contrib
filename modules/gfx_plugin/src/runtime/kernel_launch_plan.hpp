// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//
#pragma once

#include <cstdint>
#include <string_view>
#include <vector>

#include "kernel_ir/gfx_kernel_manifest.hpp"
#include "openvino/core/except.hpp"
#include "runtime/gpu_backend_base.hpp"
#include "runtime/gpu_tensor.hpp"

namespace ov {
namespace gfx_plugin {

template <typename ScalarT>
struct KernelLaunchPlan {
    std::vector<KernelArg> args;
    std::vector<ScalarT> scalar_storage;
};

template <typename ScalarT, typename ResolveInputFn>
inline KernelLaunchPlan<ScalarT> build_role_ordered_kernel_launch_plan(
    const std::vector<GfxKernelBufferRole>& roles,
    const std::vector<size_t>& direct_input_indices,
    const std::vector<ScalarT>& scalar_values,
    const std::vector<GpuTensor*>& outputs,
    const std::vector<GpuTensor*>& const_tensors,
    const std::vector<GpuTensor>& runtime_params,
    ResolveInputFn&& resolve_input,
    std::string_view stage_name) {
    const std::string_view label = stage_name.empty() ? std::string_view("<unknown>") : stage_name;
    OPENVINO_ASSERT(!roles.empty(), "GFX: kernel launch plan requires manifest ABI roles for ", label);

    KernelLaunchPlan<ScalarT> plan;
    plan.args.reserve(roles.size());
    plan.scalar_storage.reserve(scalar_values.size());

    size_t input_idx = 0;
    size_t output_idx = 0;
    size_t scalar_idx = 0;
    size_t const_idx = 0;
    size_t runtime_idx = 0;

    for (size_t arg_idx = 0; arg_idx < roles.size(); ++arg_idx) {
        switch (roles[arg_idx]) {
        case GfxKernelBufferRole::TensorInput: {
            OPENVINO_ASSERT(direct_input_indices.empty() || input_idx < direct_input_indices.size(),
                            "GFX: tensor input ABI has no direct input mapping for ",
                            label);
            const size_t node_input_idx =
                direct_input_indices.empty() ? input_idx : direct_input_indices[input_idx];
            GpuTensor* input = resolve_input(node_input_idx);
            OPENVINO_ASSERT(input && input->buf.valid(),
                            "GFX: missing tensor input ",
                            node_input_idx,
                            " for kernel launch plan in ",
                            label);
            plan.args.push_back(make_buffer_arg(static_cast<uint32_t>(arg_idx), input->buf));
            ++input_idx;
            break;
        }
        case GfxKernelBufferRole::TensorOutput: {
            OPENVINO_ASSERT(output_idx < outputs.size() && outputs[output_idx] &&
                                outputs[output_idx]->buf.valid(),
                            "GFX: missing tensor output ",
                            output_idx,
                            " for kernel launch plan in ",
                            label);
            plan.args.push_back(make_buffer_arg(static_cast<uint32_t>(arg_idx),
                                                outputs[output_idx]->buf));
            ++output_idx;
            break;
        }
        case GfxKernelBufferRole::ScalarParam: {
            OPENVINO_ASSERT(scalar_idx < scalar_values.size(),
                            "GFX: missing scalar parameter ",
                            scalar_idx,
                            " for kernel launch plan in ",
                            label);
            plan.scalar_storage.push_back(scalar_values[scalar_idx++]);
            plan.args.push_back(make_bytes_arg(static_cast<uint32_t>(arg_idx),
                                               &plan.scalar_storage.back(),
                                               sizeof(plan.scalar_storage.back())));
            break;
        }
        case GfxKernelBufferRole::ConstTensor: {
            OPENVINO_ASSERT(const_idx < const_tensors.size() && const_tensors[const_idx] &&
                                const_tensors[const_idx]->buf.valid(),
                            "GFX: missing const tensor ",
                            const_idx,
                            " for kernel launch plan in ",
                            label);
            plan.args.push_back(make_buffer_arg(static_cast<uint32_t>(arg_idx),
                                                const_tensors[const_idx]->buf));
            ++const_idx;
            break;
        }
        case GfxKernelBufferRole::RuntimeParams: {
            OPENVINO_ASSERT(runtime_idx < runtime_params.size() &&
                                runtime_params[runtime_idx].buf.valid(),
                            "GFX: missing runtime-parameter buffer ",
                            runtime_idx,
                            " for kernel launch plan in ",
                            label);
            plan.args.push_back(make_buffer_arg(static_cast<uint32_t>(arg_idx),
                                                runtime_params[runtime_idx].buf));
            ++runtime_idx;
            break;
        }
        case GfxKernelBufferRole::Unknown:
        default:
            OPENVINO_THROW("GFX: unsupported kernel launch plan ABI role in ", label);
        }
    }

    if (!direct_input_indices.empty()) {
        OPENVINO_ASSERT(input_idx == direct_input_indices.size(),
                        "GFX: kernel launch plan input count mismatch for ",
                        label);
    }
    OPENVINO_ASSERT(output_idx == outputs.size(),
                    "GFX: kernel launch plan output count mismatch for ",
                    label);
    OPENVINO_ASSERT(scalar_idx == scalar_values.size(),
                    "GFX: kernel launch plan scalar count mismatch for ",
                    label);
    OPENVINO_ASSERT(const_idx == const_tensors.size(),
                    "GFX: kernel launch plan const tensor count mismatch for ",
                    label);
    OPENVINO_ASSERT(runtime_idx == runtime_params.size(),
                    "GFX: kernel launch plan runtime-parameter count mismatch for ",
                    label);
    OPENVINO_ASSERT(kernel_args_dense(plan.args),
                    "GFX: kernel launch plan args must be densely indexed for ",
                    label);
    return plan;
}

}  // namespace gfx_plugin
}  // namespace ov
