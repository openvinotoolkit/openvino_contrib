// Copyright (C) 2021-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <optional>

#include "runtime.hpp"

namespace CUDA {

class GraphCapture;
class CaptureInfo;

class Graph : public Handle<cudaGraph_t> {
public:
    Graph(unsigned int flags);

    friend bool operator==(const Graph& lhs, const Graph& rhs);

    friend GraphCapture;

private:
    Graph(cudaGraph_t graph);

    static cudaError_t createFromNative(cudaGraph_t* pGraph, cudaGraph_t anotherGraph);

    static cudaGraph_t createNativeWithFlags(unsigned int flags);
};

bool operator==(const Graph& rhs, const Graph& lhs);

class GraphExec : public Handle<cudaGraphExec_t> {
public:
    GraphExec(const Graph& g);

    cudaGraphExecUpdateResult update(const Graph& g) const;

    void launch(const Stream& stream) const;

    friend bool operator==(const GraphExec& lhs, const GraphExec& rhs);

#if !defined(NDEBUG) || defined(_DEBUG)
private:
    static constexpr std::size_t kErrorStringLen = 1024;
    char errorMsg_[kErrorStringLen];
#endif
};

bool operator==(const GraphExec& lhs, const GraphExec& rhs);

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
    cudaGraph_t cudaGraph_{};
    cudaError_t capturedError_{cudaSuccess};
    std::optional<Graph> graph_{};
};

class UploadNode {
    friend CaptureInfo;

public:
    void update_src(const GraphExec& exec, const void* src);
    bool operator==(const UploadNode& rhs) const;

private:
    UploadNode(cudaGraphNode_t node, CUDA::DevicePointer<void*> dst, const void* src, std::size_t size);
    cudaGraphNode_t node_;
    CUDA::DevicePointer<void*> dst_;
    const void* src_;
    std::size_t size_;
};

class DownloadNode {
    friend CaptureInfo;

public:
    void update_dst(const GraphExec& exec, void* dst);
    bool operator==(const DownloadNode& rhs) const;

private:
    DownloadNode(cudaGraphNode_t node, void* dst, CUDA::DevicePointer<const void*> src, std::size_t size);
    cudaGraphNode_t node_;
    void* dst_;
    CUDA::DevicePointer<const void*> src_;
    std::size_t size_;
};

class CaptureInfo {
public:
    CaptureInfo(const Stream& capturedStream);
    UploadNode addUploadNode(CUDA::DevicePointer<void*> dst, const void* src, std::size_t size);
    DownloadNode addDownloadNode(void* dst, CUDA::DevicePointer<const void*> src, std::size_t size);

private:
    const Stream& stream_;
    cudaGraph_t capturingGraph_;
    cudaStreamCaptureStatus captureStatus_;
    const cudaGraphNode_t* deps_;
    size_t depCount_;
};

}  // namespace CUDA
