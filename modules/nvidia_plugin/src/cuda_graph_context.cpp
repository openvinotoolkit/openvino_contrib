// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "cuda_graph_context.hpp"

namespace ov {
namespace nvidia_gpu {

void CudaGraphInfo::reset() {
    graph_.reset();
    graphExec_.reset();
    parameterNodes_.clear();
    resultNodes_.clear();
    transferNodes_.clear();
    kernelNodes_.clear();
}

void CudaGraphInfo::add_parameter(const std::string& tensorName,
                                  const CUDA::Stream& stream,
                                  CUDA::DevicePointer<void*> dst,
                                  const void* src,
                                  std::size_t size) {
    CUDA::CaptureInfo captureInfo{stream};
    parameterNodes_.emplace(tensorName, captureInfo.addUploadNode(dst, src, size));
}

void CudaGraphInfo::add_result(const std::string& tensorName,
                               const CUDA::Stream& stream,
                               void* dst,
                               CUDA::DevicePointer<const void*> src,
                               std::size_t size) {
    CUDA::CaptureInfo captureInfo{stream};
    resultNodes_.emplace(tensorName, captureInfo.addDownloadNode(dst, src, size));
}

void CudaGraphInfo::add_transfer(const CUDA::Stream& stream,
                                 CUDA::DevicePointer<void*> dst,
                                 CUDA::DevicePointer<const void*> src,
                                 std::size_t size) {
    CUDA::CaptureInfo captureInfo{stream};
    transferNodes_.emplace_back(captureInfo.addTransferNode(dst, src, size));
}

bool CudaGraphInfo::is_initialized() const { return graph_.has_value() && graphExec_.has_value(); }

void CudaGraphInfo::update_capture(const TensorMappingContext& context) {
    for (auto&& [tensorName, node] : parameterNodes_) {
        node.update_src(graphExec_.value(), (context.get_input_tensor(tensorName)->data()));
    }
    for (auto&& [tensorName, node] : resultNodes_) {
        node.update_dst(graphExec_.value(), context.get_output_tensor(tensorName)->data());
    }
}

std::size_t CudaGraphInfo::get_graphs_count() const { return is_initialized() ? 1 : 0; }

void CudaGraphInfo::launch(const CUDA::Stream& stream) const { graphExec_.value().launch(stream); }

void CudaGraphPack::reset() {
    graphs_.clear();
    currentGraphIndex_ = 0;
}

void CudaGraphPack::add_parameter(const std::string& tensorName,
                                     const CUDA::Stream& stream,
                                     CUDA::DevicePointer<void*> dst,
                                     const void* src,
                                     std::size_t size) {
    OPENVINO_ASSERT(currentGraphIndex_ < graphs_.size(), "Graph index/vector size incosistency");
    graphs_[currentGraphIndex_]->add_parameter(tensorName, stream, dst, src, size);
}

void CudaGraphPack::add_result(const std::string& tensorName,
                                  const CUDA::Stream& stream,
                                  void* dst,
                                  CUDA::DevicePointer<const void*> src,
                                  std::size_t size) {
    OPENVINO_ASSERT(currentGraphIndex_ < graphs_.size(), "Graph index/vector size incosistency");
    graphs_[currentGraphIndex_]->add_result(tensorName, stream, dst, src, size);
}

void CudaGraphPack::add_transfer(const CUDA::Stream& stream,
                                 CUDA::DevicePointer<void*> dst,
                                 CUDA::DevicePointer<const void*> src,
                                 std::size_t size) {
    graphs_[currentGraphIndex_]->add_transfer(stream, dst, src, size);
}

void CudaGraphPack::set_current_graph(const CUDA::Graph& graph) {
    OPENVINO_ASSERT(currentGraphIndex_ < graphs_.size(), "Graph index/vector size incosistency");
    graphs_[currentGraphIndex_]->set_current_graph(graph);
}

bool CudaGraphPack::is_initialized() const {
    const auto size = graphs_.size();
    return size != 0 && graphs_[size - 1]->is_initialized();
}

void CudaGraphPack::update_capture(const TensorMappingContext& context) {
    for (currentGraphIndex_ = 0; currentGraphIndex_ < graphs_.size(); ++currentGraphIndex_) {
        graphs_[currentGraphIndex_]->update_capture(context);
    }
}

ICudaGraphInfo& CudaGraphPack::add(std::shared_ptr<ICudaGraphInfo> ptr) {
    currentGraphIndex_ = graphs_.size();
    graphs_.emplace_back(ptr);
    return *graphs_.back();
}

ICudaGraphInfo& CudaGraphPack::get_current_graph() { return *graphs_[currentGraphIndex_]; }

void CudaGraphPack::select_current_graph(std::size_t index) {
    OPENVINO_ASSERT(index < graphs_.size(), "Graph index/vector size incosistency");
    currentGraphIndex_ = index;
}

std::size_t CudaGraphPack::get_params_count() const {
    return std::accumulate(
        graphs_.begin(), graphs_.end(), static_cast<std::size_t>(0), [](auto sum, const auto& graph) {
            return sum + graph->get_params_count();
        });
}

std::size_t CudaGraphPack::get_results_count() const {
    return std::accumulate(
        graphs_.begin(), graphs_.end(), static_cast<std::size_t>(0), [](auto sum, const auto& graph) {
            return sum + graph->get_results_count();
        });
}

std::size_t CudaGraphPack::get_transfers_count() const {
    return std::accumulate(
        graphs_.begin(), graphs_.end(), static_cast<std::size_t>(0), [](auto sum, const auto& graph) {
            return sum + graph->get_transfers_count();
        });
}

std::size_t CudaGraphPack::get_kernels_count() const {
    return std::accumulate(
        graphs_.begin(), graphs_.end(), static_cast<std::size_t>(0), [](auto sum, const auto& graph) {
            return sum + graph->get_kernels_count();
        });
}

std::size_t CudaGraphPack::get_graphs_count() const {
    return std::accumulate(
        graphs_.begin(), graphs_.end(), static_cast<std::size_t>(0), [](auto sum, const auto& graph) {
            return sum + graph->get_graphs_count();
        });
}

void CudaGraphPack::launch(const CUDA::Stream& stream) const { graphs_[currentGraphIndex_]->launch(stream); }

}  // namespace nvidia_gpu
}  // namespace ov
