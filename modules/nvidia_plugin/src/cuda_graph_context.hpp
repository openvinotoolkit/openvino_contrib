// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <cuda/graph.hpp>

#include "cuda_tensor_mapping_context.hpp"

namespace ov {
namespace nvidia_gpu {

class CudaGraphInfo {
public:
    void add_parameter(const std::string& tensorName,
                       const CUDA::Stream& stream,
                       CUDA::DevicePointer<void*> dst,
                       const void* src,
                       std::size_t size);

    void add_result(const std::string& tensorName,
                    const CUDA::Stream& stream,
                    void* dst,
                    CUDA::DevicePointer<const void*> src,
                    std::size_t size);

    void add_transfer(const CUDA::Stream& stream,
                      CUDA::DevicePointer<void*> dst,
                      CUDA::DevicePointer<const void*> src,
                      std::size_t size);

    template <typename... Args>
    void add_kernel(const CUDA::Stream& stream, void* kernel, dim3 gridDim, dim3 blockDim, Args&&... args) {
        CUDA::CaptureInfo captureInfo{stream};
        kernelNodes_.emplace_back(captureInfo.addKernelNode(kernel, gridDim, blockDim, std::forward<Args>(args)...));
    }

    template <typename... Args>
    void update_kernel(std::size_t index, Args&&... args) {
        kernelNodes_[index].update_params(graphExec_.value(), std::forward<Args>(args)...);
    }

    bool is_initialized() const;

    void update_capture(const TensorMappingContext& context);

    std::size_t get_params_count() const { return parameterNodes_.size(); }
    std::size_t get_results_count() const { return resultNodes_.size(); }
    std::size_t get_transfers_count() const { return transferNodes_.size(); }
    std::size_t get_kernels_count() const { return kernelNodes_.size(); }

    void set_graph(const CUDA::Graph& graph);
    void set_params_graph(const CUDA::Graph& graph);
    void set_results_graph(const CUDA::Graph& graph);

    void launch(const CUDA::Stream& stream) const;
    void launch_params_graph(const CUDA::Stream& stream) const;
    void launch_results_graph(const CUDA::Stream& stream) const;

    friend bool operator==(const CudaGraphInfo& lhs, const CudaGraphInfo& rhs);
    friend bool operator!=(const CudaGraphInfo& lhs, const CudaGraphInfo& rhs);

private:
    std::optional<CUDA::Graph> graph_{};
    std::optional<CUDA::GraphExec> graphExec_{};

    std::optional<CUDA::Graph> paramsGraph_{};
    std::optional<CUDA::GraphExec> paramsGraphExec_{};

    std::optional<CUDA::Graph> resultsGraph_{};
    std::optional<CUDA::GraphExec> resultsGraphExec_{};

    std::map<std::string, CUDA::UploadNode> parameterNodes_;
    std::map<std::string, CUDA::DownloadNode> resultNodes_;

    std::vector<CUDA::TransferNode> transferNodes_;
    std::vector<CUDA::KernelNode> kernelNodes_;
};

class CudaGraphContext {
public:
    void reset();

    void add_parameter(const std::string& tensorName,
                       const CUDA::Stream& stream,
                       CUDA::DevicePointer<void*> dst,
                       const void* src,
                       std::size_t size);

    void add_result(const std::string& tensorName,
                    const CUDA::Stream& stream,
                    void* dst,
                    CUDA::DevicePointer<const void*> src,
                    std::size_t size);

    void set_current_graph(const CUDA::Graph& graph);

    bool is_initialized() const;

    void update_capture(const TensorMappingContext& context);

    void add_new_graph_info();

    const CudaGraphInfo& get_current_graph_info() const;
    CudaGraphInfo& get_current_graph_info();

    void select_current_graph(std::size_t index);

    std::size_t get_params_count() const;
    std::size_t get_results_count() const;

    std::size_t get_graphs_count() const;

    friend bool operator==(const CudaGraphContext& lhs, const CudaGraphContext& rhs);
    friend bool operator!=(const CudaGraphContext& lhs, const CudaGraphContext& rhs);

private:
    std::vector<CudaGraphInfo> graph_infos_{};
    std::size_t currentGraphIndex_ = 0;
};

bool operator==(const CudaGraphInfo& lhs, const CudaGraphInfo& rhs);

bool operator!=(const CudaGraphInfo& lhs, const CudaGraphInfo& rhs);

bool operator==(const CudaGraphContext& lhs, const CudaGraphContext& rhs);

bool operator!=(const CudaGraphContext& lhs, const CudaGraphContext& rhs);

}  // namespace nvidia_gpu
}  // namespace ov
