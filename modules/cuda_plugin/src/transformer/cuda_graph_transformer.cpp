// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "cuda_graph_transformer.hpp"

#include <ngraph/pass/manager.hpp>
#include <transformations/common_optimizations/common_optimizations.hpp>
#include <transformations/common_optimizations/conv_bias_fusion.hpp>
#include <transformations/common_optimizations/nop_elimination.hpp>
#include <transformations/convert_precision.hpp>
#include <transformations/init_node_info.hpp>
#include <transformations/op_conversions/bidirectional_sequences_decomposition.hpp>
#include <transformations/op_conversions/convert_sequences_to_tensor_iterator.hpp>
#include <transformations/op_conversions/convert_ti_to_sequences.hpp>
#include <transformer/convolution_asym_padding_transformation.hpp>
#include <transformer/fuse_conv_biasadd_activation.hpp>

#include "bidirectional_lstm_sequence_composition.hpp"
#include "concat_transformation.hpp"
#include "cuda/cuda_config.hpp"
#include "cuda_fullyconnected_transformation.hpp"
#include "matmul_transformations.hpp"
#include "noop_broadcast_transformation.hpp"
#include "transformations/op_conversions/convert_interpolate1_to_interpolate4.hpp"

using namespace CUDAPlugin;

std::shared_ptr<ngraph::Function> GraphTransformer::transform(const CUDA::Device& device,
                                                              const std::shared_ptr<const ngraph::Function>& function,
                                                              const Configuration& cfg) const {
    auto transformed_function = ngraph::clone_function(*function);

    auto passConfig = std::make_shared<ngraph::pass::PassConfig>();
    ngraph::pass::Manager manager{passConfig};

    passConfig->enable<ngraph::pass::ConvertInterpolate1ToInterpolate4>();

    [[maybe_unused]] const auto& originOps = function->get_ordered_ops();
    [[maybe_unused]] const auto& originOpsSize = originOps.size();

    manager.register_pass<ngraph::pass::InitNodeInfo>();
    manager.register_pass<ngraph::pass::CommonOptimizations>();
    if (!isHalfSupported(device)) {
        manager.register_pass<ngraph::pass::ConvertPrecision>(ngraph::element::f16, ngraph::element::f32);
    }
    if (!isInt8Supported(device)) {
        manager.register_pass<ngraph::pass::ConvertPrecision>(
            ngraph::element::i8, isHalfSupported(device) ? ngraph::element::f16 : ngraph::element::f32);
    }

    if (!cfg.disabled_tensoriterator_transform) {
        manager.register_pass<ngraph::pass::BidirectionalSequenceComposition>(passConfig);
    }
    manager.register_pass<ngraph::pass::ConvolutionAsymPaddingTransformation>();
    manager.register_pass<ngraph::pass::GroupConvolutionAsymPaddingTransformation>();
    manager.register_pass<ngraph::pass::CudaFuseConvBiasAddActivation>();
    // TODO: Enable when FusedGroupConvolutionOp is ready
    // manager.register_pass<ngraph::pass::CudaFuseGroupConvBiasAddActivation>();
    manager.register_pass<ngraph::pass::CudaFuseConvBackpropDataAdd>();
    manager.register_pass<ngraph::pass::TransposeMatMulTransformation>();
    manager.register_pass<ngraph::pass::FullyConnectedTransformation>();
    manager.register_pass<ngraph::pass::ConcatTransformation>();
    manager.register_pass<ngraph::pass::NoopBroadcastTransformation>();

    manager.run_passes(transformed_function);

    [[maybe_unused]] const auto& transformedOps = transformed_function->get_ordered_ops();
    [[maybe_unused]] const auto& transformedOpsSize = transformedOps.size();

    return transformed_function;
}
