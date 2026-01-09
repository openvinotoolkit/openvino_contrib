// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <gsl/span>
#include <memory>
#include <memory_manager/cuda_device_mem_block.hpp>
#include <memory_manager/model/cuda_immutable_memory_model_builder.hpp>
#include <memory_manager/model/cuda_memory_model.hpp>
#include <memory_manager/model/cuda_memory_model_builder.hpp>
#include <string>
#include <unordered_map>
#include <vector>

#include "memory_manager/cuda_workbuffers.hpp"
#include "openvino/core/node.hpp"

namespace ov {
namespace nvidia_gpu {

/**
 * Extracts intermediate buffer ids from intermediate representation.
 * Holds information about buffers size and lifespan.
 * Provides this information for a buffer by it's id.
 */
class OperationBuffersExtractor {
public:
    using NodePtr = std::shared_ptr<ov::Node>;
    using Byte = char;
    static constexpr char kOutputNumberSeparator = '_';

    /**
     * c-tor
     * @param [in] ordered_nodes Subgraph to execute represenation.
     * Nodes are ordered in their execution order.
     * @param [in] is_stable_params Makes input parameters alive for whole graph's life time
     * @param [in] is_stable_results Makes output results alive for till end of the graph's life time
     * @throws ov::Exception if the given subgraph is bad formed
     */
    OperationBuffersExtractor(gsl::span<const NodePtr> ordered_nodes,
                              bool is_stable_params = false,
                              bool is_stable_results = false);

    /**
     * Provides input tensors ids of the given ngraph node
     * @param node ngraph node for which input tensors ids should be provided
     * @returns Input tensors ids
     */
    std::vector<TensorID> inputTensorIds(const ov::Node& node) const;

    /**
     * Provides output tensors ids of the given ngraph node
     * @param node ngraph node for which output tensors ids should be provided
     * @returns Output tensors ids
     */
    std::vector<TensorID> outputTensorIds(const ov::Node& node) const;

    /**
     * Provides lifespan start of the given mutable buffer
     * @param buffer_id Identifier of a buffer.
     * Can be obtained via InputBufferIds or OutputBufferIds
     * @returns Lifespan start of the given buffer
     * @throws ov::Exception
     * if buffer with the provided index doesn't exist
     */
    int mutableBufferLifespanStart(BufferID buffer_id) const;

    /**
     * Provides lifespan end of the given mutable buffer
     * @param buffer_id Identifier of a buffer.
     * Can be obtained via InputBufferIds or OutputBufferIds
     * @returns Lifespan end of the given buffer
     * @throws ov::Exception
     * if buffer with the provided index doesn't exist
     */
    int mutableBufferLifespanEnd(BufferID buffer_id) const;

    /**
     * Provides size of the given mutable buffer
     * @param buffer_id Identifier of a buffer.
     * Can be obtained via InputBufferIds or OutputBufferIds
     * @returns Size of the given buffer
     * @throws ov::Exception
     * if buffer with the provided index doesn't exist
     */
    std::size_t mutableBufferSize(BufferID buffer_id) const;

    /**
     * Provides mutable buffer content
     * @param buffer_id Identifier of a buffer.
     * @returns mutable buffer content
     * @throws ov::Exception
     * if buffer with the provided index doesn't exist
     */
    gsl::span<const Byte> immutableBuffer(BufferID buffer_id) const;

    /**
     * @returns mutable buffers ids
     */
    std::vector<BufferID> mutableBuffersIds() const;

    /**
     * @returns immutable buffers ids
     */
    std::vector<BufferID> immutableBuffersIds() const;

    /**
     * Handles work buffers request for the named operation
     * @param node_idx node index
     * @param request workbuffer request
     * @returns workbuffer ids
     */
    WorkbufferIds processWorkbufferRequest(int node_idx, const WorkbufferRequest& request);

    /**
     * @returns sizes of immutable workbuffers
     */
    const std::unordered_map<BufferID, size_t>& immutableWorkbufferSizes() const { return immutable_workbuffers_; }

    /**
     * Initialize constant memory
     * @param memory_block Memory block to initialize
     */
    void initConstantMemory(DeviceMemBlock::Ptr memory_block) const;

    /**
     * Create constant memory model
     * @return MemoryModel for constants
     */
    MemoryModel::Ptr createConstantMemoryModel() const;

    /**
     * Create mutable memory model
     * @return MemoryModel for mutable buffers
     */
    MemoryModel::Ptr createMutableMemoryModel() const;

    /**
     * Create immutable memory model
     * @return MemoryModel for immutable buffers
     */
    MemoryModel::Ptr createImmutableMemoryModel() const;

    /**
     * Provides tensor size for the given node like object
     * @param node Node like object to process
     * @returns Tensor size in bytes for the given node
     */
    template <typename TNode>
    static std::size_t GetTensorByteSize(const TNode& node) {
        return node.get_element_type().size() * std::max(std::size_t(1), shape_size(node.get_shape()));
    }

    /**
     * Checks whether the given node changes tensor shape only and
     * doesn't change tensor data itself. For such nodes, input and output
     * data tensors will reuse the same buffer allocation.
     */
    static bool isReshapeOnlyNode(const ov::Node& node);

private:
    /**
     * Internal buffer representation
     */
    struct BufferDesc {
        BufferDesc(int lifespan_start, int lifespan_end, std::size_t size)
            : lifespan_start{lifespan_start}, lifespan_end{lifespan_end}, size{size} {}

        int lifespan_start;
        int lifespan_end;
        std::size_t size;
    };

    /**
     * Encapsulates mutable tensors extraction for the ov::Parameter node
     * @param node ov::Parameter node from which tensors to be extracted
     */
    void extractParameterTensors(const NodePtr& node, int node_idx);

    /**
     * Encapsulates mutable tensors extraction for the ov::Result node
     * @param node ov::Result node from which tensors to be extracted
     */
    void extractResultTensors(const NodePtr& node);

    /**
     * Encapsulates mutable tensors extraction for the Reshape like nodes
     * (nodes that checked by @isReshapeOnlyNode(...))
     * @param node Reshape like node from which tensors to be extracted
     */
    void extractReshapeTensors(const NodePtr& node, int node_idx);

    /**
     * Encapsulates mutable tensors extraction for the given node
     * @param node ngraph node from which tensors to be extracted
     * @param node_idx Current node index
     */
    void extractMutableTensors(const NodePtr& node, int node_idx);

    /**
     * Merge mutable tensors in one tensor for ConcatOptimized node
     * @param node ConcatOptimized node (custom node)
     * @param node_idx Current node index
     */
    void mergeConcatMutableTensors(const NodePtr& node, int node_idx);

    /**
     * Encapsulates immutable tensors extraction for the given node
     * @param node ngraph node from which tensors to be extracted
     */
    void extractImmutableTensors(const NodePtr& node);

    /**
     * Provides internal tensor name
     * @param [in] output Output to process
     * @returns internal tensor name
     */
    template <class Node>
    static inline std::string GetTensorNameInternal(const ov::Output<Node>& output) {
        return output.get_node()->get_name() + kOutputNumberSeparator + std::to_string(output.get_index());
    }

    /**
     * Provides internal tensor name
     * @param [in] input Input to process
     * @returns internal tensor name
     */
    template <class Node>
    static inline std::string GetTensorNameInternal(const ov::Input<Node>& input) {
        const auto output = input.get_source_output();
        return output.get_node()->get_name() + kOutputNumberSeparator + std::to_string(output.get_index());
    }

    /**
     * Checks whether the given node is a parameter node
     */
    static bool IsParameterNode(const ov::Node& node);

    /**
     * Checks whether the given node is a result node
     */
    static bool IsResultNode(const ov::Node& node);

    /**
     * Checks whether the given node is a constant node
     */
    static bool IsConstantNode(const ov::Node& node);

    /**
     * Checks whether the given node is a ConcatOptimized node (concat optimized)
     */
    static bool IsConcatOptimizedNode(const ov::Node& node);

    /**
     * Exception helper
     */
    static void ThrowBufferSizesAreNotMatchError(const ov::Input<ov::Node>& input);

    /**
     * Exception helper
     */
    static void ThrowGraphIsBadFormedError(const ov::Input<ov::Node>& input);

private:
    std::unordered_map<BufferID, BufferDesc> mutable_buffers_;
    std::unordered_map<BufferID, size_t> mutable_tensor_sizes_;
    std::unordered_map<BufferID, gsl::span<const Byte>> immutable_buffers_;
    std::unordered_map<BufferID, size_t> immutable_workbuffers_;
    std::unordered_map<std::string, TensorID::Ptr> tensor_names_;
    unsigned next_buffer_id_{};
    const bool is_stable_params_ = false;
    const bool is_stable_results_ = false;
    const unsigned long num_ordered_nodes_ = 0;
};

}  // namespace nvidia_gpu
}  // namespace ov
