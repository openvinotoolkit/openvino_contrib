// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#import <MetalPerformanceShadersGraph/MPSGraphPoolingOps.h>

#include "ops/pooling.hpp"

namespace ov::metal_plugin::ops {

MetalNode* build_max_pool(NodeContext& ctx, const ov::op::v1::MaxPool& node) {
    const auto& kernel = node.get_kernel();
    const auto& strides = node.get_strides();
    const auto& pads_begin = node.get_pads_begin();
    const auto& pads_end = node.get_pads_end();
    if (kernel.size() != 2 || strides.size() != 2 || pads_begin.size() != 2 || pads_end.size() != 2) {
        OPENVINO_THROW("MaxPool: only 2D pooling with explicit pads/strides is supported");
    }
    auto rank = node.get_input_partial_shape(0).rank();
    if (!(rank.is_static() && rank.get_length() == 4)) {
        OPENVINO_THROW("MaxPool: only rank-4 NCHW inputs are supported");
    }
    auto* src_node = ctx.get_node(node.input_value(0).get_node());
    std::vector<MetalNode*> deps{src_node};
    auto ov_ptr = node.shared_from_this();
    MetalNode* out = ctx.create_node(MetalOpType::MaxPool, ov_ptr, deps);

    auto src = ctx.get_input_value(node, 0);
    auto src_nhwc = ctx.to_nhwc(src);

    auto desc = [MPSGraphPooling2DOpDescriptor descriptorWithKernelWidth:(NSUInteger)kernel[1]
                                                            kernelHeight:(NSUInteger)kernel[0]
                                                               strideInX:(NSUInteger)strides[1]
                                                               strideInY:(NSUInteger)strides[0]
                                                         dilationRateInX:1
                                                         dilationRateInY:1
                                                             paddingLeft:(NSUInteger)pads_begin[1]
                                                            paddingRight:(NSUInteger)pads_end[1]
                                                              paddingTop:(NSUInteger)pads_begin[0]
                                                           paddingBottom:(NSUInteger)pads_end[0]
                                                           paddingStyle:MPSGraphPaddingStyleExplicit
                                                             dataLayout:MPSGraphTensorNamedDataLayoutNHWC];
    desc.ceilMode = node.get_rounding_type() == ov::op::RoundingType::CEIL;
    MPSGraphTensor* res = [ctx.graph() maxPooling2DWithSourceTensor:src_nhwc.tensor descriptor:desc name:nil];

    ov::Shape out_nchw = node.get_output_shape(0);
    ov::Shape out_nhwc{out_nchw[0], out_nchw[2], out_nchw[3], out_nchw[1]};
    Value pool_out{{out_nhwc, node.get_output_element_type(0), Layout::NHWC}, res};
    auto pool_nchw = ctx.to_nchw(pool_out);

    out->output_desc = pool_nchw.desc;
    out->mps_tensor = pool_nchw.tensor;
    return out;
}

MetalNode* build_avg_pool(NodeContext& ctx, const ov::op::v1::AvgPool& node) {
    const auto& kernel = node.get_kernel();
    const auto& strides = node.get_strides();
    const auto& pads_begin = node.get_pads_begin();
    const auto& pads_end = node.get_pads_end();
    if (kernel.size() != 2 || strides.size() != 2 || pads_begin.size() != 2 || pads_end.size() != 2) {
        OPENVINO_THROW("AvgPool: only 2D pooling with explicit pads/strides is supported");
    }
    auto rank = node.get_input_partial_shape(0).rank();
    if (!(rank.is_static() && rank.get_length() == 4)) {
        OPENVINO_THROW("AvgPool: only rank-4 NCHW inputs are supported");
    }
    auto* src_node = ctx.get_node(node.input_value(0).get_node());
    std::vector<MetalNode*> deps{src_node};
    auto ov_ptr = node.shared_from_this();
    MetalNode* out = ctx.create_node(MetalOpType::AvgPool, ov_ptr, deps);

    auto src = ctx.get_input_value(node, 0);
    auto src_nhwc = ctx.to_nhwc(src);

    auto desc = [MPSGraphPooling2DOpDescriptor descriptorWithKernelWidth:(NSUInteger)kernel[1]
                                                            kernelHeight:(NSUInteger)kernel[0]
                                                               strideInX:(NSUInteger)strides[1]
                                                               strideInY:(NSUInteger)strides[0]
                                                         dilationRateInX:1
                                                         dilationRateInY:1
                                                             paddingLeft:(NSUInteger)pads_begin[1]
                                                            paddingRight:(NSUInteger)pads_end[1]
                                                              paddingTop:(NSUInteger)pads_begin[0]
                                                           paddingBottom:(NSUInteger)pads_end[0]
                                                           paddingStyle:MPSGraphPaddingStyleExplicit
                                                             dataLayout:MPSGraphTensorNamedDataLayoutNHWC];
    desc.ceilMode = node.get_rounding_type() == ov::op::RoundingType::CEIL;
    desc.includeZeroPadToAverage = !node.get_exclude_pad();
    MPSGraphTensor* res = [ctx.graph() avgPooling2DWithSourceTensor:src_nhwc.tensor descriptor:desc name:nil];

    ov::Shape out_nchw = node.get_output_shape(0);
    ov::Shape out_nhwc{out_nchw[0], out_nchw[2], out_nchw[3], out_nchw[1]};
    Value pool_out{{out_nhwc, node.get_output_element_type(0), Layout::NHWC}, res};
    auto pool_nchw = ctx.to_nchw(pool_out);

    out->output_desc = pool_nchw.desc;
    out->mps_tensor = pool_nchw.tensor;
    return out;
}

}  // namespace ov::metal_plugin::ops
