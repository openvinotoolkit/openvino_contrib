// Copyright (C) 2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <memory>
#include <memory_manager/model/details/cuda_memory_utils.hpp>
#include <vector>

#include "dnn_be.hpp"
#include "event.hpp"
#include "ie_common.h"

namespace CUDA {

inline std::vector<std::shared_ptr<DnnBEExecutionPlan>> getAllExecutionPlansFromHeuristics(
    const std::shared_ptr<DnnBEOperationGraphDescriptor>& graph, const CUDA::DnnHandle& dnnHandle) {
    std::vector<std::shared_ptr<DnnBEExecutionPlan>> plans;
#if (CUDNN_VERSION >= 8200)
    {
        auto heuristics =
            CUDA::DnnBEEngineHeuristicsDescriptorBuilder().setOpGraph(graph).setMode(CUDNN_HEUR_MODE_INSTANT).build();

        std::vector<std::shared_ptr<CUDA::DnnBEEngineConfigDescriptor>> configs = heuristics->getEngineConfigs();
        for (const auto& config : configs) {
            try {
                auto plan = CUDA::DnnBEExecutionPlanBuilder().setDnnHandle(dnnHandle).setEngineConfig(config).build();
                plans.push_back(std::move(plan));
            } catch (const InferenceEngine::Exception&) {
                continue;
            }
        }
    }
#else
    {
        const auto num_engines = graph->getEngineCount();
        for (int64_t i = 0; i < num_engines; ++i) {
            try {
                auto engine = CUDA::DnnBEEngineBuilder().setOpEngineGraph(graph).setGlobalIndex(i).build();
                auto engineConfig = CUDA::DnnBEEngineConfigDescriptorBuilder().setEngine(engine).build();
                auto plan =
                    CUDA::DnnBEExecutionPlanBuilder().setDnnHandle(dnnHandle).setEngineConfig(engineConfig).build();
                plans.push_back(std::move(plan));
            } catch (const InferenceEngine::Exception&) {
                continue;
            }
        }
    }
#endif
    return std::move(plans);
}

template <size_t NumBenchmarks>
std::shared_ptr<CUDA::DnnBEExecutionPlan> performBenchmarks(
    const CUDA::DnnHandle& dnnHandle,
    const std::vector<std::shared_ptr<CUDA::DnnBEExecutionPlan>>& plans,
    CUDA::DnnBEVariantPackBuilder& variantPackBuilder) {
    auto getDescendSortedWorkspaceSizes = [](const std::vector<std::shared_ptr<CUDA::DnnBEExecutionPlan>>& plans) {
        std::vector<size_t> workspace_sizes{};
        std::transform(plans.begin(), plans.end(), std::back_inserter(workspace_sizes), [](const auto& p) {
            return p->getWorkspaceSize();
        });
        std::sort(workspace_sizes.begin(), workspace_sizes.end(), std::greater<size_t>{});
        return workspace_sizes;
    };

    auto tryAllocateMaxWorkspace =
        [](const std::vector<size_t>& workspace_sizes) -> std::optional<std::pair<CUDA::DefaultAllocation, size_t>> {
        for (const auto workspace_size : workspace_sizes) {
            try {
                const auto aligned_workspace_size = CUDAPlugin::applyAllignment(workspace_size);
                CUDA::DefaultAllocation workspace = CUDA::DefaultStream::stream().malloc(aligned_workspace_size);
                return std::make_pair<CUDA::DefaultAllocation, size_t>(std::move(workspace),
                                                                       static_cast<size_t>(workspace_size));
            } catch (...) {
                // NOTE: If not enough memory try another workspace size
            }
        }
        return std::nullopt;
    };

    auto filterPlansByWorkspaceSize = [](const std::vector<std::shared_ptr<CUDA::DnnBEExecutionPlan>>& plans,
                                         const size_t max_workspace_size) {
        auto filtered_plans = plans;
        auto erased =
            std::remove_if(filtered_plans.begin(), filtered_plans.end(), [max_workspace_size](const auto& plan) {
                return plan->getWorkspaceSize() > max_workspace_size;
            });
        filtered_plans.erase(erased, filtered_plans.end());
        return filtered_plans;
    };

    const auto& workspace_sizes = getDescendSortedWorkspaceSizes(plans);
    auto max_workspace = tryAllocateMaxWorkspace(workspace_sizes);
    auto [workspace, max_workspace_size] = max_workspace.value();
    auto filtered_plans = filterPlansByWorkspaceSize(plans, max_workspace_size);

    if (max_workspace) {
        variantPackBuilder.setWorkspase(workspace.get());
    }
    auto variantPack = variantPackBuilder.build();

    auto executeBenchmarkStep = [&](auto& plan) {
        throwIfError(::cudnnBackendExecute(dnnHandle.get(), plan->get(), variantPack->get()));
    };

    CUDA::Event start, stop;
    CUDA::Device{}.synchronize();

    auto stream = dnnHandle.getStream();

    std::vector<float> time;
    for (auto& plan : filtered_plans) {
        // Warmup
        executeBenchmarkStep(plan);

        start.record(stream);
        for (size_t i = 0; i < NumBenchmarks; ++i) {
            executeBenchmarkStep(plan);
        }
        stop.record(stream);
        stop.synchronize();
        const float time_ms = stop.elapsedSince(start) / NumBenchmarks;
        time.push_back(time_ms);
    }
    const auto min_time = std::min_element(time.begin(), time.end());
    const auto best_time_index = std::distance(time.begin(), min_time);

    return filtered_plans[best_time_index];
}

inline std::vector<size_t> generateStrides(gsl::span<const size_t> dim, cudnnTensorFormat_t filterFormat) {
    std::vector<size_t> strides(dim.size());
    // For INT8x4 and INT8x32 we still compute standard strides here to input
    // into the cuDNN functions. We will manually scale by resizeFactor in the cpu ref.
    if (filterFormat == CUDNN_TENSOR_NCHW) {
        strides[strides.size() - 1] = 1;
        for (int64_t d = strides.size() - 2; d >= 0; d--) {
            strides[d] = strides[d + 1] * dim[d + 1];
        }
    } else {
        // Here we assume that the format is CUDNN_TENSOR_NHWC
        strides[1] = 1;
        strides[strides.size() - 1] = strides[1] * dim[1];
        for (int64_t d = strides.size() - 2; d >= 2; d--) {
            strides[d] = strides[d + 1] * dim[d + 1];
        }
        strides[0] = strides[2] * dim[2];
    }
    return strides;
}

}  // namespace CUDA
