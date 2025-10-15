// Copyright (C) 2021-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <cuda.h>
#include "graph.hpp"
#include "openvino/core/except.hpp"
#include <fmt/format.h>

namespace CUDA {

Graph::Graph(unsigned int flags) :
        Graph { createNativeWithFlags(flags) } {
}

Graph::Graph(cudaGraph_t graph) :
        Handle { createFromNative, cudaGraphDestroy, graph } {
}

cudaError_t Graph::createFromNative(cudaGraph_t* pGraph, const cudaGraph_t anotherGraph) {
    *pGraph = anotherGraph;
    return cudaSuccess;
}

cudaGraph_t Graph::createNativeWithFlags(unsigned int flags) {
    cudaGraph_t g;
    throwIfError(cudaGraphCreate(&g, flags));
    return g;
}

bool operator==(const Graph& rhs, const Graph& lhs) { return rhs.get() == lhs.get(); }

GraphExec::GraphExec(const Graph& g)
#if !defined(NDEBUG) || defined(_DEBUG)
    try
#endif
    : Handle(cudaGraphInstantiate,
             cudaGraphExecDestroy,
             g.get(),
             static_cast<cudaGraphNode_t*>(nullptr),
#if !defined(NDEBUG) || defined(_DEBUG)
             errorMsg_,
             kErrorStringLen)
#else
             static_cast<char*>(nullptr),
             static_cast<size_t>(0ul))
#endif
{
}
#if !defined(NDEBUG) || defined(_DEBUG)
catch (std::exception& e) {
    OPENVINO_THROW(e.what(), ": ", errorMsg_);
}
#endif

#if defined(CUDA_VERSION) && CUDA_VERSION >= 12020
cudaGraphExecUpdateResultInfo GraphExec::update(const Graph& g) const {
    cudaGraphExecUpdateResultInfo res;
    throwIfError(cudaGraphExecUpdate(get(), g.get(), &res));
    return res;
}
#else
cudaGraphExecUpdateResult GraphExec::update(const Graph& g) const {
    cudaGraphExecUpdateResult res;
    throwIfError(cudaGraphExecUpdate(get(), g.get(), nullptr, &res));
    return res;
}
#endif

void GraphExec::launch(const Stream& stream) const {
    throwIfError(cudaGraphLaunch(get(), stream.get()));
}

bool operator==(const GraphExec& lhs, const GraphExec& rhs) { return rhs.get() == lhs.get(); }

GraphCapture::GraphCaptureScope::GraphCaptureScope(GraphCapture& graphCapture) : graphCapture_{graphCapture} {
    throwIfError(cudaStreamBeginCapture(graphCapture_.stream_.get(), cudaStreamCaptureModeThreadLocal));
}

GraphCapture::GraphCaptureScope::~GraphCaptureScope() {
    graphCapture_.capturedError_ = cudaStreamEndCapture(graphCapture_.stream_.get(), &graphCapture_.cudaGraph_);
}

GraphCapture::GraphCapture(const Stream& capturedStream) :
        stream_ { capturedStream } {
}

GraphCapture::GraphCaptureScope GraphCapture::getScope() {
    graph_.reset();
    cudaGraph_ = nullptr;
    capturedError_ = cudaSuccess;
    return GraphCapture::GraphCaptureScope { *this };
}

const Graph& GraphCapture::getGraph() {
    throwIfError(capturedError_);
    if (!graph_ && cudaGraph_ != nullptr) {
        graph_ = std::make_optional<Graph>( Graph{ cudaGraph_ });
    }
    return graph_.value();
}

CaptureInfo::CaptureInfo(const Stream& capturedStream) : stream_{capturedStream} {
    throwIfError(cudaStreamGetCaptureInfo_v2(capturedStream.get(), &captureStatus_, nullptr,
            &capturingGraph_, &deps_, &depCount_));
}

UploadNode CaptureInfo::addUploadNode(DevicePointer<void*> dst, const void* src, std::size_t size) {
    cudaGraphNode_t newNode;
    throwIfError(cudaGraphAddMemcpyNode1D(&newNode, capturingGraph_, deps_, depCount_,
            dst.get(), src, size, cudaMemcpyHostToDevice));
    throwIfError(cudaStreamUpdateCaptureDependencies(stream_.get(), &newNode, 1, 1));
    return UploadNode{newNode, dst, src, size};
}

DownloadNode CaptureInfo::addDownloadNode(void* dst, DevicePointer<const void*> src,
                                                       std::size_t size) {
    cudaGraphNode_t newNode;
    throwIfError(cudaGraphAddMemcpyNode1D(&newNode, capturingGraph_, deps_, depCount_,
            dst, src.get(), size, cudaMemcpyDeviceToHost));
    throwIfError(cudaStreamUpdateCaptureDependencies(stream_.get(), &newNode, 1, 1));
    return DownloadNode{newNode, dst, src, size};
}

TransferNode CaptureInfo::addTransferNode(CUDA::DevicePointer<void*> dst,
                                          CUDA::DevicePointer<const void*> src,
                                          std::size_t size) {
    cudaGraphNode_t newNode;
    throwIfError(cudaGraphAddMemcpyNode1D(
        &newNode, capturingGraph_, deps_, depCount_, dst.get(), src.get(), size, cudaMemcpyDeviceToDevice));
    throwIfError(cudaStreamUpdateCaptureDependencies(stream_.get(), &newNode, 1, 1));
    return TransferNode{newNode, dst, src, size};
}

void UploadNode::update_src(const GraphExec& exec, const void* src) {
    if (src_ != src) {
        throwIfError(cudaGraphExecMemcpyNodeSetParams1D(exec.get(), node_,
                dst_.get(), src, size_, cudaMemcpyHostToDevice));
        src_ = src;
    }
}

UploadNode::UploadNode(cudaGraphNode_t node, DevicePointer<void*> dst, const void* src, std::size_t size)
    : node_{node},
      dst_{dst},
      src_{src},
      size_{size} {
}

void DownloadNode::update_dst(const GraphExec& exec, void* dst) {
    if (dst_ != dst) {
        throwIfError(cudaGraphExecMemcpyNodeSetParams1D(exec.get(), node_,
                dst, src_.get(), size_, cudaMemcpyDeviceToHost));
        dst_ = dst;
    }
}

DownloadNode::DownloadNode(cudaGraphNode_t node, void* dst, DevicePointer<const void*> src, std::size_t size)
    : node_{node}, dst_{dst}, src_{src}, size_{size} {}

void CUDA::TransferNode::update_ptrs(const GraphExec& exec,
                                     CUDA::DevicePointer<void*> dst,
                                     CUDA::DevicePointer<const void*> src) {
    if (dst_ != dst || src_ != src) {
        dst_ = dst;
        src_ = src;
        throwIfError(cudaGraphExecMemcpyNodeSetParams1D(
            exec.get(), node_, dst_.get(), src_.get(), size_, cudaMemcpyDeviceToDevice));
    }
}

CUDA::TransferNode::TransferNode(cudaGraphNode_t node,
                                 CUDA::DevicePointer<void*> dst,
                                 CUDA::DevicePointer<const void*> src,
                                 std::size_t size)
    : node_{node}, dst_{dst}, src_{src}, size_{size} {}

CUDA::KernelNode::KernelNode(cudaGraphNode_t node, CUDA::NodeParams&& params) : node_{node}, node_params_{params} {}

bool UploadNode::operator==(const UploadNode& rhs) const {
    return size_ == rhs.size_ && src_ == rhs.src_ && dst_.get() == rhs.dst_.get() && node_ == rhs.node_;
}

bool DownloadNode::operator==(const DownloadNode& rhs) const {
    return size_ == rhs.size_ && src_.get() == rhs.src_.get() && dst_ == rhs.dst_ && node_ == rhs.node_;
}

bool CUDA::TransferNode::operator==(const TransferNode& rhs) const {
    return size_ == rhs.size_ && src_.get() == rhs.src_.get() && dst_.get() == rhs.dst_.get() && node_ == rhs.node_;
}

bool KernelNode::operator==(const KernelNode& rhs) const {
    return node_ == rhs.node_ && node_params_ == rhs.node_params_;
}
}  // namespace CUDA
