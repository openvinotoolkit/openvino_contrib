// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "detection_output.hpp"

#include <cuda_operation_registry.hpp>

#include "converters.hpp"

namespace CUDAPlugin {

DetectionOutputOp::DetectionOutputOp(const CreationContext& context,
                                     const NodeOp& node,
                                     IndexCollection&& inputIds,
                                     IndexCollection&& outputIds)
    : OperationBase{context, node, move(inputIds), move(outputIds)}, element_type_{node.get_input_element_type(0)} {
    const auto& ngraph_attrs = node.get_attrs();
    kernel::DetectionOutput::Attrs kernel_attrs;

    auto locShape = node.get_input_shape(0);
    auto priorsShape = node.get_input_shape(2);
    auto outShape = node.get_output_shape(0);

    kernel_attrs.num_images = locShape[0];
    kernel_attrs.offset = ngraph_attrs.normalized ? 0 : 1;
    kernel_attrs.prior_size = ngraph_attrs.normalized ? 4 : 5;
    kernel_attrs.num_priors = priorsShape[2] / kernel_attrs.prior_size;
    Ensures(locShape[0] == priorsShape[0]);
    kernel_attrs.num_results = outShape[2];
    kernel_attrs.out_total_size = shape_size(outShape);
    kernel_attrs.num_loc_classes = ngraph_attrs.share_location ? 1 : static_cast<size_t>(ngraph_attrs.num_classes);
    kernel_attrs.num_classes = ngraph_attrs.num_classes;

    kernel_attrs.background_label_id = ngraph_attrs.background_label_id;
    kernel_attrs.top_k = ngraph_attrs.top_k;
    kernel_attrs.variance_encoded_in_target = ngraph_attrs.variance_encoded_in_target;
    kernel_attrs.keep_top_k = ngraph_attrs.keep_top_k[0];
    kernel_attrs.code_type = ngraph_attrs.code_type == "caffe.PriorBoxParameter.CORNER"
                                 ? kernel::DetectionOutput::Attrs::CodeType::Caffe_PriorBoxParameter_CORNER
                                 : kernel::DetectionOutput::Attrs::CodeType::Caffe_PriorBoxParameter_CENTER_SIZE;
    kernel_attrs.share_location = ngraph_attrs.share_location;
    kernel_attrs.nms_threshold = ngraph_attrs.nms_threshold;
    kernel_attrs.confidence_threshold = ngraph_attrs.confidence_threshold;
    kernel_attrs.clip_after_nms = ngraph_attrs.clip_after_nms;
    kernel_attrs.clip_before_nms = ngraph_attrs.clip_before_nms;
    kernel_attrs.decrease_label_id = ngraph_attrs.decrease_label_id;
    kernel_attrs.normalized = ngraph_attrs.normalized;
    kernel_attrs.input_height = ngraph_attrs.input_height;
    kernel_attrs.input_width = ngraph_attrs.input_width;
    kernel_attrs.objectness_score = ngraph_attrs.objectness_score;
    const size_t max_threads_per_block = context.device().props().maxThreadsPerBlock;
    if (node.inputs().size() == 5) {
        kernel_ =
            std::make_optional<kernel::DetectionOutput>(convertDataType<CUDAPlugin::kernel::Type_t>(element_type_),
                                                        max_threads_per_block,
                                                        ov::shape_size(node.get_input_shape(0)),
                                                        ov::shape_size(node.get_input_shape(1)),
                                                        ov::shape_size(node.get_input_shape(2)),
                                                        ov::shape_size(node.get_input_shape(3)),
                                                        ov::shape_size(node.get_input_shape(4)),
                                                        ov::shape_size(node.get_output_shape(0)),
                                                        kernel_attrs);
    } else {
        kernel_ =
            std::make_optional<kernel::DetectionOutput>(convertDataType<CUDAPlugin::kernel::Type_t>(element_type_),
                                                        max_threads_per_block,
                                                        ov::shape_size(node.get_input_shape(0)),
                                                        ov::shape_size(node.get_input_shape(1)),
                                                        ov::shape_size(node.get_input_shape(2)),
                                                        0,
                                                        0,
                                                        ov::shape_size(node.get_output_shape(0)),
                                                        kernel_attrs);
    }
}

void DetectionOutputOp::Execute(const InferenceRequestContext& context,
                                Inputs inputTensors,
                                Outputs outputTensors,
                                const Workbuffers& workbuffers) const {
    const auto& stream = context.getThreadContext().stream();
    if (inputTensors.size() == 5) {
        (*kernel_)(stream,
                   inputTensors[0],
                   inputTensors[1],
                   inputTensors[2],
                   inputTensors[3].get(),
                   inputTensors[4].get(),
                   workbuffers.mutable_buffers,
                   outputTensors[0]);
    } else {
        (*kernel_)(stream,
                   inputTensors[0],
                   inputTensors[1],
                   inputTensors[2],
                   nullptr,
                   nullptr,
                   workbuffers.mutable_buffers,
                   outputTensors[0]);
    }
}

void DetectionOutputOp::InitSharedImmutableWorkbuffers(const Buffers& buffers) {
    kernel_.value().initSharedImmutableWorkbuffers(buffers);
}

WorkbufferRequest DetectionOutputOp::GetWorkBufferRequest() const {
    return {kernel_.value().getImmutableWorkbufferSizes(), kernel_.value().getMutableWorkbufferSizes()};
}

OPERATION_REGISTER(DetectionOutputOp, DetectionOutput);
}  // namespace CUDAPlugin
