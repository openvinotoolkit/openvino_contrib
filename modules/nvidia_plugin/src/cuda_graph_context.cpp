// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "cuda_graph_context.hpp"

namespace ov {
namespace nvidia_gpu {

void CudaGraphContext::reset() {
    graphs_.clear();
    currentGraphIndex_ = 0;
}

void CudaGraphContext::startNextGraphAddition() {
    currentGraphIndex_ = graphs_.size();
    graphs_.emplace_back();
}

void CudaGraphContext::addParameter(const std::string& tensorName,
                                    const CUDA::Stream& stream,
                                    CUDA::DevicePointer<void*> dst,
                                    const void* src,
                                    std::size_t size) {
    if (currentGraphIndex_ >= graphs_.size()) {
        throw_ov_exception("Graph index/vector size incosistency");
    }
    graphs_[currentGraphIndex_].addParameter(tensorName, stream, dst, src, size);
}

void CudaGraphContext::addResult(const std::string& tensorName,
                                 const CUDA::Stream& stream,
                                 void* dst,
                                 CUDA::DevicePointer<const void*> src,
                                 std::size_t size) {
    if (currentGraphIndex_ >= graphs_.size()) {
        throw_ov_exception("Graph index/vector size incosistency");
    }
    graphs_[currentGraphIndex_].addResult(tensorName, stream, dst, src, size);
}

void CudaGraphContext::addGraph(const CUDA::Graph& graph) {
    if (currentGraphIndex_ >= graphs_.size()) {
        throw_ov_exception("Graph index/vector size incosistency");
    }
    graphs_[currentGraphIndex_].setGraph(graph);
}

bool CudaGraphContext::isInitialized() const {
    const auto size = graphs_.size();
    return size != 0 && graphs_[size - 1].isInitialized();
}

void CudaGraphContext::updateCapture(const TensorMappingContext& context) {
    for (currentGraphIndex_ = 0; currentGraphIndex_ < graphs_.size(); ++currentGraphIndex_) {
        graphs_[currentGraphIndex_].updateCapture(context);
    }
}

void CudaGraphContext::launch(std::size_t index, const CUDA::Stream& stream) const {
    currentGraphIndex_ = index;
    if (currentGraphIndex_ >= graphs_.size()) {
        throw_ov_exception("Graph index/vector size incosistency");
    }
    graphs_[currentGraphIndex_].launch(stream);
}

std::size_t CudaGraphContext::getParamsCount() const {
    std::size_t res = 0;
    for (const auto& graph : graphs_) {
        res += graph.getParamsCount();
    }
    return res;
}

std::size_t CudaGraphContext::getResultsCount() const {
    std::size_t res = 0;
    for (const auto& graph : graphs_) {
        res += graph.getResultsCount();
    }
    return res;
}

std::size_t CudaGraphContext::getGraphsCount() const { return graphs_.size(); }

void CudaGraphContext::CudaGraphInfo::addParameter(const std::string& tensorName,
                                                   const CUDA::Stream& stream,
                                                   CUDA::DevicePointer<void*> dst,
                                                   const void* src,
                                                   std::size_t size) {
    CUDA::CaptureInfo captureInfo{stream};
    parameterNodes_.emplace(tensorName, captureInfo.addUploadNode(dst, src, size));
}

void CudaGraphContext::CudaGraphInfo::addResult(const std::string& tensorName,
                                                const CUDA::Stream& stream,
                                                void* dst,
                                                CUDA::DevicePointer<const void*> src,
                                                std::size_t size) {
    CUDA::CaptureInfo captureInfo{stream};
    resultNodes_.emplace(tensorName, captureInfo.addDownloadNode(dst, src, size));
}

void CudaGraphContext::CudaGraphInfo::setGraph(const CUDA::Graph& graph) {
    graph_.emplace(graph);
    graphExec_.emplace(graph);
}

bool CudaGraphContext::CudaGraphInfo::isInitialized() const { return graph_.has_value() && graphExec_.has_value(); }

void CudaGraphContext::CudaGraphInfo::updateCapture(const TensorMappingContext& context) {
    for (auto&& [tensorName, node] : parameterNodes_) {
        node.update_src(graphExec_.value(), (context.get_input_tensor(tensorName)->data()));
    }
    for (auto&& [tensorName, node] : resultNodes_) {
        node.update_dst(graphExec_.value(), context.get_output_tensor(tensorName)->data());
    }
}

void CudaGraphContext::CudaGraphInfo::launch(const CUDA::Stream& stream) const { graphExec_.value().launch(stream); }

std::size_t CudaGraphContext::CudaGraphInfo::getParamsCount() const { return parameterNodes_.size(); }

std::size_t CudaGraphContext::CudaGraphInfo::getResultsCount() const { return resultNodes_.size(); }

bool operator==(const CudaGraphContext::CudaGraphInfo& lhs, const CudaGraphContext::CudaGraphInfo& rhs) {
    return lhs.graph_ == rhs.graph_ && lhs.graphExec_ == rhs.graphExec_ && lhs.parameterNodes_ == rhs.parameterNodes_ &&
           lhs.resultNodes_ == rhs.resultNodes_;
}

bool operator!=(const CudaGraphContext::CudaGraphInfo& lhs, const CudaGraphContext::CudaGraphInfo& rhs) {
    return !(lhs == rhs);
}

bool operator==(const CudaGraphContext& lhs, const CudaGraphContext& rhs) { return lhs.graphs_ == rhs.graphs_; }

bool operator!=(const CudaGraphContext& lhs, const CudaGraphContext& rhs) { return !(lhs == rhs); }

}  // namespace nvidia_gpu
}  // namespace ov
