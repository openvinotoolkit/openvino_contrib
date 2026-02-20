// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <cuda/graph.hpp>

#include "cuda_tensor_mapping_context.hpp"

namespace ov {
namespace nvidia_gpu {

class ICudaGraphInfo {

public:
    virtual ~ICudaGraphInfo() = 0;

    virtual void reset() = 0;

    virtual void add_parameter(const std::string& tensorName,
                               const CUDA::Stream& stream,
                               CUDA::DevicePointer<void*> dst,
                               const void* src,
                               std::size_t size) = 0;

    virtual void add_result(const std::string& tensorName,
                            const CUDA::Stream& stream,
                            void* dst,
                            CUDA::DevicePointer<const void*> src,
                            std::size_t size) = 0;

    virtual void add_transfer(const CUDA::Stream& stream,
                              CUDA::DevicePointer<void*> dst,
                              CUDA::DevicePointer<const void*> src,
                              std::size_t size) = 0;

    template <typename... Args>
    void add_kernel(const CUDA::Stream& stream, void* kernel, dim3 gridDim, dim3 blockDim, Args&&... args) {
        CUDA::CaptureInfo captureInfo{stream};
        get_kernels().emplace_back(captureInfo.addKernelNode(kernel, gridDim, blockDim, std::forward<Args>(args)...));
    }

    template <typename... Args>
    void update_kernel(std::size_t index, Args&&... args) {
        get_kernels()[index].update_params(get_graph_exec().value(), std::forward<Args>(args)...);
    }

    virtual void set_current_graph(const CUDA::Graph& graph) = 0;

    virtual bool is_initialized() const = 0;
    virtual bool is_nested() const = 0;

    virtual void update_capture(const TensorMappingContext& context) = 0;

    virtual ICudaGraphInfo& add(std::shared_ptr<ICudaGraphInfo> ptr) = 0;

    virtual ICudaGraphInfo& get_current_graph() = 0;

    virtual void select_current_graph(std::size_t index) = 0;

    virtual std::size_t get_params_count() const = 0;
    virtual std::size_t get_results_count() const = 0;
    virtual std::size_t get_transfers_count() const = 0;
    virtual std::size_t get_kernels_count() const = 0;

    virtual std::size_t get_graphs_count() const = 0;

    virtual void launch(const CUDA::Stream& stream) const = 0;

    virtual std::vector<CUDA::KernelNode>& get_kernels() = 0;
    virtual std::optional<CUDA::GraphExec>& get_graph_exec() = 0;
};

inline ICudaGraphInfo::~ICudaGraphInfo() = default;

class CudaGraphInfo : public ICudaGraphInfo {
public:
    CudaGraphInfo() = default;
    CudaGraphInfo(const CudaGraphInfo&) = delete;
    CudaGraphInfo& operator=(const CudaGraphInfo&) = delete;

    static std::shared_ptr<ICudaGraphInfo> create() { return std::make_shared<CudaGraphInfo>(); }

    void reset() override;

    void add_parameter(const std::string& tensorName,
                       const CUDA::Stream& stream,
                       CUDA::DevicePointer<void*> dst,
                       const void* src,
                       std::size_t size) override;

    void add_result(const std::string& tensorName,
                    const CUDA::Stream& stream,
                    void* dst,
                    CUDA::DevicePointer<const void*> src,
                    std::size_t size) override;

    void add_transfer(const CUDA::Stream& stream,
                      CUDA::DevicePointer<void*> dst,
                      CUDA::DevicePointer<const void*> src,
                      std::size_t size) override;

    void set_current_graph(const CUDA::Graph& graph) override {
        graph_.emplace(graph);
        graphExec_.emplace(graph);
    }

    bool is_initialized() const override;
    bool is_nested() const override { return false; };

    void update_capture(const TensorMappingContext& context) override;

    ICudaGraphInfo& add(std::shared_ptr<ICudaGraphInfo> ptr) override {
        OPENVINO_THROW("add() called for CudaGraphInfo");
    }

    ICudaGraphInfo& get_current_graph() override { return *this; }

    void select_current_graph(std::size_t index) override {
        OPENVINO_THROW("select_current_graph() called for CudaGraphInfo");
    }

    std::size_t get_params_count() const override { return parameterNodes_.size(); }
    std::size_t get_results_count() const override { return resultNodes_.size(); }
    std::size_t get_transfers_count() const override { return transferNodes_.size(); }
    std::size_t get_kernels_count() const override { return kernelNodes_.size(); }

    std::size_t get_graphs_count() const override;

    void launch(const CUDA::Stream& stream) const override;

    std::vector<CUDA::KernelNode>& get_kernels() override { return kernelNodes_; };
    std::optional<CUDA::GraphExec>& get_graph_exec() override { return graphExec_; };

    const std::map<std::string, CUDA::UploadNode>& get_parameter_nodes() const { return parameterNodes_; }
    const std::map<std::string, CUDA::DownloadNode>& get_result_nodes() const { return resultNodes_; }

private:
    std::optional<CUDA::Graph> graph_{};
    std::optional<CUDA::GraphExec> graphExec_{};

    std::map<std::string, CUDA::UploadNode> parameterNodes_;
    std::map<std::string, CUDA::DownloadNode> resultNodes_;

    std::vector<CUDA::TransferNode> transferNodes_;
    std::vector<CUDA::KernelNode> kernelNodes_;
};

class CudaGraphPack : public ICudaGraphInfo {
public:
    CudaGraphPack() = default;
    CudaGraphPack(const CudaGraphPack&) = delete;
    CudaGraphPack& operator=(const CudaGraphPack&) = delete;

    static std::shared_ptr<ICudaGraphInfo> create() { return std::make_shared<CudaGraphPack>(); }

    void reset() override;

    void add_parameter(const std::string& tensorName,
                       const CUDA::Stream& stream,
                       CUDA::DevicePointer<void*> dst,
                       const void* src,
                       std::size_t size) override;

    void add_result(const std::string& tensorName,
                    const CUDA::Stream& stream,
                    void* dst,
                    CUDA::DevicePointer<const void*> src,
                    std::size_t size) override;

    void add_transfer(const CUDA::Stream& stream,
                      CUDA::DevicePointer<void*> dst,
                      CUDA::DevicePointer<const void*> src,
                      std::size_t size) override;

    void set_current_graph(const CUDA::Graph& graph) override;

    bool is_initialized() const override;
    bool is_nested() const override { return true; };

    void update_capture(const TensorMappingContext& context) override;

    ICudaGraphInfo& add(std::shared_ptr<ICudaGraphInfo> ptr) override;

    ICudaGraphInfo& get_current_graph() override;

    void select_current_graph(std::size_t index) override;

    std::size_t get_params_count() const override;
    std::size_t get_results_count() const override;
    std::size_t get_transfers_count() const override;
    std::size_t get_kernels_count() const override;

    std::size_t get_graphs_count() const override;

    void launch(const CUDA::Stream& stream) const override;

    std::vector<CUDA::KernelNode>& get_kernels() override { return graphs_[currentGraphIndex_]->get_kernels(); };
    std::optional<CUDA::GraphExec>& get_graph_exec() override { return graphs_[currentGraphIndex_]->get_graph_exec(); };

private:
    std::vector<std::shared_ptr<ICudaGraphInfo>> graphs_{};
    std::size_t currentGraphIndex_ = 0;
};

using CudaGraphContext = CudaGraphPack;

}  // namespace nvidia_gpu
}  // namespace ov
