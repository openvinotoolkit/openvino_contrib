// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <ngraph/node.hpp>
#include <memory>
#include <string>
#include <unordered_map>
#include <vector>
#include <gsl/span>

namespace CUDAPlugin {

/**
 * Extracts intermediate buffer indices from intermediate representation.
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
     * Provides input buffers indices of the given ngraph node
     * @param node ngraph node for which input buffers indices should be provided
     * @returns Input buffers indices
     */
    std::vector<unsigned> inputBufferIndices(const ngraph::Node& node) const;

    /**
     * Provides output buffers indices of the given ngraph node
     * @param node ngraph node for which output buffers indices should be provided
     * @returns Output buffers indices
     */
    std::vector<unsigned> outputBufferIndices(const ngraph::Node& node) const;

    /**
     * Provides lifespan start of the given mutable buffer
     * @param buffer_index Index of a buffer.
     * Can be obtained via InputBufferIndices or OutputBufferIndices
     * @returns Lifespan start of the given buffer
     * @throws InferenceEngine::details::InferenceEngineException
     * if buffer with the provided index doesn't exist
     */
    int mutableBufferLifespanStart(unsigned buffer_index) const;

    /**
     * Provides lifespan end of the given mutable buffer
     * @param buffer_index Index of a buffer.
     * Can be obtained via InputBufferIndices or OutputBufferIndices
     * @returns Lifespan end of the given buffer
     * @throws InferenceEngine::details::InferenceEngineException
     * if buffer with the provided index doesn't exist
     */
    int mutableBufferLifespanEnd(unsigned buffer_index) const;

    /**
     * Provides size of the given mutable buffer
     * @param buffer_index Index of a buffer.
     * Can be obtained via InputBufferIndices or OutputBufferIndices
     * @returns Size of the given buffer
     * @throws InferenceEngine::details::InferenceEngineException
     * if buffer with the provided index doesn't exist
     */
    std::size_t mutableBufferSize(unsigned buffer_index) const;

    /**
     * Provides mutable buffer content
     * @param buffer_index Index of a buffer.
     * @returns mutable buffer content
     * @throws InferenceEngine::details::InferenceEngineException
     * if buffer with the provided index doesn't exist
     */
    gsl::span<const Byte> immutableBuffer(unsigned buffer_index) const;

    /**
     * @returns mutable buffers indices
     */
    std::vector<unsigned> mutableBuffersIndices() const;

    /**
     * @returns immutable buffers indices
     */
    std::vector<unsigned> immutableBuffersIndices() const;

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
     * Encapsulates mutable buffers extraction for the given node
     * @param node ngraph node from which buffers to be extracted
     * @param [in] [out] buffer_idx Current buffer index.
     * Should be incremented if new buffer was added.
     * @param node_idx Current node index
     */
    void extractMutableBuffers(const NodePtr& node, int node_idx, unsigned& buffer_idx);

    /**
     * Encapsulates immutable buffers extraction for the given node
     * @param node ngraph node from which buffers to be extracted
     * @param [in] [out] buffer_idx Current buffer index.
     * Should be incremented if new buffer was added.
     * @param node_idx Current node index
     */
    void extractImmutableBuffers(const NodePtr& node, unsigned& buffer_idx);

    /**
     * Provides buffer size for the given output
     * @param output Output to process
     * @returns Buffer size for the given output
     */
    static std::size_t GetBufferSize(const ngraph::Output<ngraph::Node>& output);

    /**
     * Provides buffer size for the given input
     * @param input Input to process
     * @returns Buffer size for the given input
     */
    static std::size_t GetBufferSize(const ngraph::Input<ngraph::Node>& output);

    /**
     * Provides internal buffer name
     * @param [in] output Output to process
     * @returns internal buffer name
     */
    template<class Node>
    static inline std::string GetBufferNameInternal(const ngraph::Output<Node>& output) {
        return output.get_node()->get_name() + kOutputNumberSeparator + std::to_string(output.get_index());
    }

    /**
     * Provides internal buffer name
     * @param [in] input Input to process
     * @returns internal buffer name
     */
    template<class Node>
    static inline std::string GetBufferNameInternal(const ngraph::Input<Node>& input) {
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
    std::unordered_map<unsigned, BufferDesc> mutable_buffers_;
    std::unordered_map<unsigned, gsl::span<const Byte>> immutable_buffers_;
    std::unordered_map<std::string, unsigned> buffer_names_;
};

} // namespace CUDAPlugin
