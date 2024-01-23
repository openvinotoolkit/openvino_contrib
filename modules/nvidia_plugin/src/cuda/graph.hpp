// Copyright (C) 2021-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <optional>

#include <cuda/node_params.hpp>
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

#if defined(CUDA_VERSION) && CUDA_VERSION >= 12020
    cudaGraphExecUpdateResultInfo update(const Graph& g) const;
#else
    cudaGraphExecUpdateResult update(const Graph& g) const;
#endif

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

class TransferNode {
    friend CaptureInfo;

public:
    void update_ptrs(const GraphExec& exec, CUDA::DevicePointer<void*> dst, CUDA::DevicePointer<const void*> src);
    bool operator==(const TransferNode& rhs) const;

private:
    TransferNode(cudaGraphNode_t node,
                 CUDA::DevicePointer<void*> dst,
                 CUDA::DevicePointer<const void*> src,
                 std::size_t size);

    cudaGraphNode_t node_;
    CUDA::DevicePointer<void*> dst_;
    CUDA::DevicePointer<const void*> src_;
    std::size_t size_;
};

bool operator==(const cudaKernelNodeParams& lhs, const cudaKernelNodeParams& rhs);

class KernelNode {
    friend CaptureInfo;

public:
    template <typename... Args>
    void update_params(const GraphExec& exec, Args&&... args) {
        node_params_.reset_args();
        node_params_.add_args(std::forward<Args>(args)...);
        throwIfError(cudaGraphExecKernelNodeSetParams(exec.get(), node_, &node_params_.get_knp()));
    }

    bool operator==(const KernelNode& rhs) const;

private:
    KernelNode(cudaGraphNode_t node, CUDA::NodeParams&& params);

    cudaGraphNode_t node_;
    CUDA::NodeParams node_params_;
};

class CaptureInfo {
public:
    CaptureInfo(const Stream& capturedStream);
    UploadNode addUploadNode(CUDA::DevicePointer<void*> dst, const void* src, std::size_t size);
    DownloadNode addDownloadNode(void* dst, CUDA::DevicePointer<const void*> src, std::size_t size);
    TransferNode addTransferNode(CUDA::DevicePointer<void*> dst,
                                 CUDA::DevicePointer<const void*> src,
                                 std::size_t size);
    template <typename... Args>
    KernelNode addKernelNode(void* kernel, dim3 gridDim, dim3 blockDim, Args&&... args);

private:
    const Stream& stream_;
    cudaGraph_t capturingGraph_;
    cudaStreamCaptureStatus captureStatus_;
    const cudaGraphNode_t* deps_;
    size_t depCount_;
};

template <typename... Args>
KernelNode CaptureInfo::addKernelNode(void* kernel, dim3 gridDim, dim3 blockDim, Args&&... args) {
    cudaGraphNode_t newNode;
    CUDA::NodeParams params{kernel, gridDim, blockDim};
    params.add_args(std::forward<Args>(args)...);
    throwIfError(cudaGraphAddKernelNode(&newNode, capturingGraph_, deps_, depCount_, &params.get_knp()));
    throwIfError(cudaStreamUpdateCaptureDependencies(stream_.get(), &newNode, 1, 1));
    return KernelNode{newNode, std::move(params)};
}

}  // namespace CUDA
