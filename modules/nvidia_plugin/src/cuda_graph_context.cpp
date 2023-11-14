// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "cuda_graph_context.hpp"

namespace ov {
namespace nvidia_gpu {

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

void CudaGraphInfo::set_graph(const CUDA::Graph& graph) {
    graph_.emplace(graph);
    graphExec_.emplace(graph);
}

void CudaGraphInfo::set_params_graph(const CUDA::Graph& graph) {
    paramsGraph_.emplace(graph);
    paramsGraphExec_.emplace(graph);
}

void CudaGraphInfo::set_results_graph(const CUDA::Graph& graph) {
    resultsGraph_.emplace(graph);
    resultsGraphExec_.emplace(graph);
}

void CudaGraphInfo::launch(const CUDA::Stream& stream) const { graphExec_.value().launch(stream); }

void CudaGraphInfo::launch_params_graph(const CUDA::Stream& stream) const { paramsGraphExec_.value().launch(stream); }

void CudaGraphInfo::launch_results_graph(const CUDA::Stream& stream) const { resultsGraphExec_.value().launch(stream); }

bool operator==(const CudaGraphInfo& lhs, const CudaGraphInfo& rhs) {
    return lhs.graph_ == rhs.graph_ && lhs.graphExec_ == rhs.graphExec_ && lhs.parameterNodes_ == rhs.parameterNodes_ &&
           lhs.resultNodes_ == rhs.resultNodes_ && lhs.transferNodes_ == rhs.transferNodes_ &&
           lhs.kernelNodes_ == rhs.kernelNodes_;
}

bool operator!=(const CudaGraphInfo& lhs, const CudaGraphInfo& rhs) { return !(lhs == rhs); }

void CudaGraphContext::reset() {
    graph_infos_.clear();
    currentGraphIndex_ = 0;
}

void CudaGraphContext::add_parameter(const std::string& tensorName,
                                     const CUDA::Stream& stream,
                                     CUDA::DevicePointer<void*> dst,
                                     const void* src,
                                     std::size_t size) {
    OPENVINO_ASSERT(currentGraphIndex_ < graph_infos_.size(), "Graph index/vector size incosistency");
    graph_infos_[currentGraphIndex_].add_parameter(tensorName, stream, dst, src, size);
}

void CudaGraphContext::add_result(const std::string& tensorName,
                                  const CUDA::Stream& stream,
                                  void* dst,
                                  CUDA::DevicePointer<const void*> src,
                                  std::size_t size) {
    OPENVINO_ASSERT(currentGraphIndex_ < graph_infos_.size(), "Graph index/vector size incosistency");
    graph_infos_[currentGraphIndex_].add_result(tensorName, stream, dst, src, size);
}

void CudaGraphContext::set_current_graph(const CUDA::Graph& graph) {
    OPENVINO_ASSERT(currentGraphIndex_ < graph_infos_.size(), "Graph index/vector size incosistency");
    graph_infos_[currentGraphIndex_].set_graph(graph);
}

bool CudaGraphContext::is_initialized() const {
    const auto size = graph_infos_.size();
    return size != 0 && graph_infos_[size - 1].is_initialized();
}

void CudaGraphContext::update_capture(const TensorMappingContext& context) {
    for (currentGraphIndex_ = 0; currentGraphIndex_ < graph_infos_.size(); ++currentGraphIndex_) {
        graph_infos_[currentGraphIndex_].update_capture(context);
    }
}

void CudaGraphContext::add_new_graph_info() {
    currentGraphIndex_ = graph_infos_.size();
    graph_infos_.emplace_back();
}

const CudaGraphInfo& CudaGraphContext::get_current_graph_info() const { return graph_infos_[currentGraphIndex_]; }

CudaGraphInfo& CudaGraphContext::get_current_graph_info() { return graph_infos_[currentGraphIndex_]; }

void CudaGraphContext::select_current_graph(std::size_t index) {
    OPENVINO_ASSERT(index < graph_infos_.size(), "Graph index/vector size incosistency");
    currentGraphIndex_ = index;
}

std::size_t CudaGraphContext::get_params_count() const {
    std::size_t res = 0;
    for (const auto& graph : graph_infos_) {
        res += graph.get_params_count();
    }
    return res;
}

std::size_t CudaGraphContext::get_results_count() const {
    std::size_t res = 0;
    for (const auto& graph : graph_infos_) {
        res += graph.get_results_count();
    }
    return res;
}

std::size_t CudaGraphContext::get_graphs_count() const { return graph_infos_.size(); }

bool operator==(const CudaGraphContext& lhs, const CudaGraphContext& rhs) { return lhs.graph_infos_ == rhs.graph_infos_; }

bool operator!=(const CudaGraphContext& lhs, const CudaGraphContext& rhs) { return !(lhs == rhs); }

}  // namespace nvidia_gpu
}  // namespace ov
