// Copyright (C) 2021-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "runtime.hpp"
#include <optional>

namespace CUDA {

class GraphCapture;

class Graph: public Handle<cudaGraph_t> {
public:
    Graph(unsigned int flags);

    friend GraphCapture;

private:
    Graph(cudaGraph_t graph);

    static cudaError_t createFromNative(cudaGraph_t *pGraph, cudaGraph_t anotherGraph);

    static cudaGraph_t createNativeWithFlags(unsigned int flags);
};


class GraphExec: public Handle<cudaGraphExec_t> {
public:
    GraphExec(const Graph& g);

    cudaGraphExecUpdateResult update(const Graph& g);

    void launch(const Stream& stream);

#if !defined(NDEBUG) || defined(_DEBUG)
private:
    static constexpr std::size_t kErrorStringLen = 1024;
    char errorMsg_[kErrorStringLen];
#endif
};


class GraphCapture {
public:
    class GraphCaptureScope {
    public:
        GraphCaptureScope(GraphCapture& graphCapture);

        GraphCaptureScope(const GraphCaptureScope&) = delete;

        GraphCaptureScope& operator=(const GraphCaptureScope&) = delete;

        ~GraphCaptureScope();

    private:
        GraphCapture& graphCapture_;
    };

    GraphCapture(const Stream& capturedStream);

    [[nodiscard]] GraphCaptureScope getScope();

    [[nodiscard]] const Graph& getGraph();

private:
    Stream stream_;
    cudaGraph_t cudaGraph_ {};
    cudaError_t capturedError_ {cudaSuccess};
    std::optional<Graph> graph_ {};
};

}// namespace CUDA
