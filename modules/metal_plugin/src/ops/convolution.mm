// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#import <MetalPerformanceShadersGraph/MPSGraphConvolutionOps.h>

#include "ops/convolution.hpp"

namespace ov::metal_plugin::ops {

MetalNode* build_convolution(NodeContext& ctx, const ov::op::v1::Convolution& node) {
    const auto& strides = node.get_strides();
    const auto& pads_begin = node.get_pads_begin();
    const auto& pads_end = node.get_pads_end();
    const auto& dilations = node.get_dilations();
    if (strides.size() != 2 || pads_begin.size() != 2 || pads_end.size() != 2 || dilations.size() != 2) {
        OPENVINO_THROW("Convolution: unsupported stride/pads/dilations configuration");
    }
    auto* src_node = ctx.get_node(node.input_value(0).get_node());
    auto* wts_node = ctx.get_node(node.input_value(1).get_node());
    std::vector<MetalNode*> deps{src_node, wts_node};

    auto ov_ptr = node.shared_from_this();
    MetalNode* out = ctx.create_node(MetalOpType::Convolution, ov_ptr, deps);

    auto src = ctx.get_input_value(node, 0);
    auto wts = ctx.get_input_value(node, 1);
    auto src_nhwc = ctx.to_nhwc(src);

    MPSGraphConvolution2DOpDescriptor* desc =
        [MPSGraphConvolution2DOpDescriptor descriptorWithStrideInX:(NSUInteger)strides[1]
                                                       strideInY:(NSUInteger)strides[0]
                                               dilationRateInX:(NSUInteger)dilations[1]
                                               dilationRateInY:(NSUInteger)dilations[0]
                                                        groups:1
                                                   paddingLeft:(NSUInteger)pads_begin[1]
                                                  paddingRight:(NSUInteger)pads_end[1]
                                                    paddingTop:(NSUInteger)pads_begin[0]
                                                 paddingBottom:(NSUInteger)pads_end[0]
                                                 paddingStyle:MPSGraphPaddingStyleExplicit
                                                     dataLayout:MPSGraphTensorNamedDataLayoutNHWC
                                                  weightsLayout:MPSGraphTensorNamedDataLayoutOIHW];

    MPSGraphTensor* res = [ctx.graph() convolution2DWithSourceTensor:src_nhwc.tensor
                                                       weightsTensor:wts.tensor
                                                          descriptor:desc
                                                                name:nil];

    ov::Shape out_nchw = node.get_output_shape(0);
    ov::Shape out_nhwc{out_nchw[0], out_nchw[2], out_nchw[3], out_nchw[1]};
    Value conv_out{{out_nhwc, node.get_output_element_type(0), Layout::NHWC}, res};
    auto conv_nchw = ctx.to_nchw(conv_out);

    out->output_desc = conv_nchw.desc;
    out->mps_tensor = conv_nchw.tensor;
    return out;
}

}  // namespace ov::metal_plugin::ops
