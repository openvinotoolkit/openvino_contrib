// Copyright (C) 2021-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "graph.hpp"
#include <ie_common.h>
#include <fmt/format.h>

namespace CUDA {

Graph::Graph(unsigned int flags) :
        Graph { createNativeWithFlags(flags) } {
}

Graph::Graph(cudaGraph_t graph) :
        Handle { createFromNative, cudaGraphDestroy, graph } {
}

cudaError_t Graph::createFromNative(cudaGraph_t *pGraph, const cudaGraph_t anotherGraph) {
    *pGraph = anotherGraph;
    return cudaSuccess;
}

cudaGraph_t Graph::createNativeWithFlags(unsigned int flags) {
    cudaGraph_t g;
    throwIfError(cudaGraphCreate(&g, flags));
    return g;
}

// clang-format off
GraphExec::GraphExec(const Graph &g)
#if !defined(NDEBUG) || defined(_DEBUG)
try
#endif
:
Handle(cudaGraphInstantiate, cudaGraphExecDestroy, g.get(), static_cast<cudaGraphNode_t*>(nullptr),
#if !defined(NDEBUG) || defined(_DEBUG)
       errorMsg_, kErrorStringLen)
#else
       static_cast<char*>(nullptr), static_cast<size_t>(0ul))
#endif
{
}
#if !defined(NDEBUG) || defined(_DEBUG)
catch (std::exception &e) {
    throw InferenceEngine::GeneralError { fmt::format("{}: {}", e.what(), errorMsg_) };
}
#endif
// clang-format on

cudaGraphExecUpdateResult GraphExec::update(const Graph &g) {
    cudaGraphExecUpdateResult res;
    throwIfError(cudaGraphExecUpdate(get(), g.get(), nullptr, &res));
    return res;
}

void GraphExec::launch(const Stream &stream) {
    throwIfError(cudaGraphLaunch(get(), stream.get()));
}

GraphCapture::GraphCaptureScope::GraphCaptureScope(GraphCapture &graphCapture) :
        graphCapture_ { graphCapture } {
    throwIfError(cudaStreamBeginCapture(graphCapture_.stream_.get(), cudaStreamCaptureModeGlobal));
}

GraphCapture::GraphCaptureScope::~GraphCaptureScope() {
    graphCapture_.capturedError_ = cudaStreamEndCapture(graphCapture_.stream_.get(), &graphCapture_.cudaGraph_);
}

GraphCapture::GraphCapture(const Stream &capturedStream) :
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

} // namespace CUDA
