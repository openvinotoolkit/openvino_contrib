// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <cuda/graph.hpp>

#include "cuda_tensor_mapping_context.hpp"

namespace ov {
namespace nvidia_gpu {

class TiCudaGraphInfo {
public:
    void add_transfer(const CUDA::Stream& stream,
                      CUDA::DevicePointer<void*> dst,
                      CUDA::DevicePointer<const void*> src,
                      std::size_t size);

    template <typename... Args>
    void add_kernel(const CUDA::Stream& stream, void* kernel, dim3 gridDim, dim3 blockDim, Args&&... args) {
        CUDA::CaptureInfo captureInfo{stream};
        kernelNodes_.emplace_back(captureInfo.addKernelNode(kernel, gridDim, blockDim, std::forward<Args>(args)...));
    }

    void set_params_graph(const CUDA::Graph& graph);
    void set_body_graph(const CUDA::Graph& graph);
    void set_results_graph(const CUDA::Graph& graph);

    template <typename... Args>
    void update_kernel(std::size_t index, Args&&... args) {
        kernelNodes_[index].update_params(bodyGraphExec_.value(), std::forward<Args>(args)...);
    }

    void launch_params_graph(const CUDA::Stream& stream) const;
    void launch_body_graph(const CUDA::Stream& stream) const;
    void launch_results_graph(const CUDA::Stream& stream) const;

    std::size_t get_transfers_count() const;
    std::size_t get_kernels_count() const;

private:
    std::optional<CUDA::Graph> paramsGraph_{};
    std::optional<CUDA::GraphExec> paramsGraphExec_{};

    std::optional<CUDA::Graph> bodyGraph_{};
    std::optional<CUDA::GraphExec> bodyGraphExec_{};

    std::optional<CUDA::Graph> resultsGraph_{};
    std::optional<CUDA::GraphExec> resultsGraphExec_{};

    std::vector<CUDA::TransferNode> transferNodes_;
    std::vector<CUDA::KernelNode> kernelNodes_;
};

class CudaGraphContext {
public:
    void reset();

    void start_next_graph_addition();

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

    void add_graph(const CUDA::Graph& graph);

    void add_ti_graph(const std::string& ti_op_name, const CUDA::Graph& graph);

    TiCudaGraphInfo& get_ti_graph(const std::string& ti_op_name) const;

    bool is_initialized() const;

    void update_capture(const TensorMappingContext& context);

    void launch(std::size_t index, const CUDA::Stream& stream) const;

    std::size_t get_params_count() const;
    std::size_t get_results_count() const;

    std::size_t get_graphs_count() const;

    friend bool operator==(const CudaGraphContext& lhs, const CudaGraphContext& rhs);
    friend bool operator!=(const CudaGraphContext& lhs, const CudaGraphContext& rhs);

private:
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

        void set_graph(const CUDA::Graph& graph);

        bool is_initialized() const;

        void update_capture(const TensorMappingContext& context);

        void launch(const CUDA::Stream& stream) const;

        std::size_t get_params_count() const;
        std::size_t get_results_count() const;

        friend bool operator==(const CudaGraphInfo& lhs, const CudaGraphInfo& rhs);
        friend bool operator!=(const CudaGraphInfo& lhs, const CudaGraphInfo& rhs);

    private:
        std::optional<CUDA::Graph> graph_{};
        std::optional<CUDA::GraphExec> graphExec_{};
        std::map<std::string, CUDA::UploadNode> parameterNodes_;
        std::map<std::string, CUDA::DownloadNode> resultNodes_;

    };

    friend bool operator==(const CudaGraphInfo& lhs, const CudaGraphInfo& rhs);
    friend bool operator!=(const CudaGraphInfo& lhs, const CudaGraphInfo& rhs);

private:
    std::vector<CudaGraphInfo> graphs_{};
    mutable std::unordered_map<std::string, TiCudaGraphInfo> ti_graphs_;
    mutable std::size_t currentGraphIndex_ = 0;
};

bool operator==(const CudaGraphContext::CudaGraphInfo& lhs, const CudaGraphContext::CudaGraphInfo& rhs);

bool operator!=(const CudaGraphContext::CudaGraphInfo& lhs, const CudaGraphContext::CudaGraphInfo& rhs);

bool operator==(const CudaGraphContext& lhs, const CudaGraphContext& rhs);

bool operator!=(const CudaGraphContext& lhs, const CudaGraphContext& rhs);

}  // namespace nvidia_gpu
}  // namespace ov
