// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <cuda/graph.hpp>
#include "cuda_tensor_mapping_context.hpp"

namespace ov {
namespace nvidia_gpu {

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

    std::vector<CudaGraphInfo> graphs_{};
    mutable std::size_t currentGraphIndex_ = 0;
};

bool operator==(const CudaGraphContext::CudaGraphInfo& lhs, const CudaGraphContext::CudaGraphInfo& rhs);

bool operator!=(const CudaGraphContext::CudaGraphInfo& lhs, const CudaGraphContext::CudaGraphInfo& rhs);

bool operator==(const CudaGraphContext& lhs, const CudaGraphContext& rhs);

bool operator!=(const CudaGraphContext& lhs, const CudaGraphContext& rhs);

}  // namespace nvidia_gpu
}  // namespace ov
