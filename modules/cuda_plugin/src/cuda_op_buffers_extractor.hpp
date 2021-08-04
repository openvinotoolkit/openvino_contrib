// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <gsl/span>
#include <memory>
#include <memory_manager/model/cuda_memory_model.hpp>
#include <ngraph/node.hpp>
#include <string>
#include <unordered_map>
#include <vector>

#include "memory_manager/cuda_workbuffers.hpp"

namespace CUDAPlugin {

/**
 * Extracts intermediate buffer ids from intermediate representation.
 * Holds information about buffers size and lifespan.
 * Provides this information for a buffer by it's id.
 */
class OperationBuffersExtractor {
   public:
    using NodePtr = std::shared_ptr<ngraph::Node>;
    using Byte = char;
    static constexpr char kOutputNumberSeparator = '_';

    /**
     * c-tor
     * @param [in] ordered_nodes Subgraph to execute represenation.
     * Nodes are ordered in their execution order.
     * @throws InferenceEngineException if the given subgraph is bad formed
     */
    OperationBuffersExtractor(gsl::span<const NodePtr> ordered_nodes);

    /**
     * Provides input tensors ids of the given ngraph node
     * @param node ngraph node for which input tensors ids should be provided
     * @returns Input tensors ids
     */
    std::vector<TensorID> inputTensorIds(const ngraph::Node& node) const;

    /**
     * Provides output tensors ids of the given ngraph node
     * @param node ngraph node for which output tensors ids should be provided
     * @returns Output tensors ids
     */
    std::vector<TensorID> outputTensorIds(const ngraph::Node& node) const;

    /**
     * Provides lifespan start of the given mutable buffer
     * @param buffer_id Identifier of a buffer.
     * Can be obtained via InputBufferIds or OutputBufferIds
     * @returns Lifespan start of the given buffer
     * @throws InferenceEngine::details::InferenceEngineException
     * if buffer with the provided index doesn't exist
     */
    int mutableBufferLifespanStart(BufferID buffer_id) const;

    /**
     * Provides lifespan end of the given mutable buffer
     * @param buffer_id Identifier of a buffer.
     * Can be obtained via InputBufferIds or OutputBufferIds
     * @returns Lifespan end of the given buffer
     * @throws InferenceEngine::details::InferenceEngineException
     * if buffer with the provided index doesn't exist
     */
    int mutableBufferLifespanEnd(BufferID buffer_id) const;

    /**
     * Provides size of the given mutable buffer
     * @param buffer_id Identifier of a buffer.
     * Can be obtained via InputBufferIds or OutputBufferIds
     * @returns Size of the given buffer
     * @throws InferenceEngine::details::InferenceEngineException
     * if buffer with the provided index doesn't exist
     */
    std::size_t mutableBufferSize(BufferID buffer_id) const;

    /**
     * Provides mutable buffer content
     * @param buffer_id Identifier of a buffer.
     * @returns mutable buffer content
     * @throws InferenceEngine::details::InferenceEngineException
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
    const std::unordered_map<BufferID, size_t>& immutableWorkbufferSizes() const {
        return immutable_workbuffers_;
    }
   private:
    /**
     * Internal buffer representation
     */
    struct BufferDesc {
        BufferDesc(int lifespan_start, int lifespan_end, std::size_t size) :
            lifespan_start {lifespan_start},
            lifespan_end {lifespan_end},
            size {size}
        {}

        int lifespan_start;
        int lifespan_end;
        std::size_t size;
    };

    /**
     * Encapsulates mutable tensors extraction for the given node
     * @param node ngraph node from which tensors to be extracted
     * @param node_idx Current node index
     */
    void extractMutableTensors(const NodePtr& node, int node_idx);

    /**
     * Merge mutable tensors in one buffer for ConcatOptimized node
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
     * Provides tensor size for the given output
     * @param output Output to process
     * @returns Tensor size in bytes for the given output
     */
    static std::size_t GetTensorByteSize(const ngraph::Output<ngraph::Node>& input);

    /**
     * Provides tensor size for the given input
     * @param input Input to process
     * @returns Tensor size in bytes for the given input
     */
    static std::size_t GetTensorByteSize(const ngraph::Input<ngraph::Node>& input);

    /**
     * Provides internal tensor name
     * @param [in] output Output to process
     * @returns internal tensor name
     */
    template<class Node>
    static inline std::string GetTensorNameInternal(const ngraph::Output<Node>& output) {
        return output.get_node()->get_name() + kOutputNumberSeparator + std::to_string(output.get_index());
    }

    /**
     * Provides internal tensor name
     * @param [in] input Input to process
     * @returns internal tensor name
     */
    template<class Node>
    static inline std::string GetTensorNameInternal(const ngraph::Input<Node>& input) {
        const auto output = input.get_source_output();
        return output.get_node()->get_name() + kOutputNumberSeparator + std::to_string(output.get_index());
    }

    /**
     * Checks whether the given node is a result node
     */
    static bool IsResultNode(const ngraph::Node& node);

    /**
     * Checks whether the given node is a constant node
     */
    static bool IsConstantNode(const ngraph::Node& node);

    /**
     * Checks whether the given node is a ConcatOptimized node (concat optimized)
     */
    static bool IsConcatOptimizedNode(const ngraph::Node& node);

    /**
     * Checks whether the given node changes tensor shape only and
     * doesn't change tensor data itself. For such nodes, input and output
     * data tensors will reuse the same buffer allocation.
     */
    static bool isReshapeOnlyNode(const ngraph::Node& node);

    /**
     * Exception helper
     */
    static void ThrowBufferSizesAreNotMatchError(const ngraph::Input<ngraph::Node>& input);

    /**
     * Exception helper
     */
    static void ThrowGraphIsBadFormedError(const ngraph::Input<ngraph::Node>& input);
   private:
    std::unordered_map<BufferID, BufferDesc> mutable_buffers_;
    std::unordered_map<BufferID, gsl::span<const Byte>> immutable_buffers_;
    std::unordered_map<BufferID, size_t> immutable_workbuffers_;
    std::unordered_map<std::string, TensorID> tensor_names_;
    unsigned next_buffer_id_{};
};

} // namespace CUDAPlugin
