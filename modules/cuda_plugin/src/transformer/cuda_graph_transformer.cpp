// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "cuda_graph_transformer.hpp"

#include <openvino/pass/manager.hpp>
#include <transformations/common_optimizations/common_optimizations.hpp>
#include <transformations/common_optimizations/nop_elimination.hpp>
#include <transformations/convert_precision.hpp>
#include <transformations/init_node_info.hpp>
#include <transformations/op_conversions/bidirectional_sequences_decomposition.hpp>
#include <transformations/op_conversions/convert_mod.hpp>
#include <transformations/op_conversions/convert_sequences_to_tensor_iterator.hpp>
#include <transformations/op_conversions/convert_ti_to_sequences.hpp>
#include <transformer/convolution_asym_padding_transformation.hpp>
#include <transformer/fuse_conv_biasadd_activation.hpp>
#include <transformer/preprocessing/preprocessing.hpp>

#include "bidirectional_lstm_sequence_composition.hpp"
#include "concat_transformation.hpp"
#include "cuda/cuda_config.hpp"
#include "cuda_fullyconnected_transformation.hpp"
#include "matmul_transformations.hpp"
#include "noop_broadcast_transformation.hpp"
#include "transformations/op_conversions/convert_divide.hpp"
#include "transformations/op_conversions/convert_interpolate1_to_interpolate4.hpp"
#include "transformations/op_conversions/convert_subtract.hpp"
#include "transformations/op_conversions/mvn6_decomposition.hpp"
#include "remove_duplicated_results_transformation.hpp"
#include "transformations/common_optimizations/convert_compression_only_to_legacy.hpp"
#include "remove_redundant_convert_transformation.hpp"
#include <fmt/format.h>

using namespace CUDAPlugin;

std::shared_ptr<ngraph::Function> GraphTransformer::export_transform(
    const CUDA::Device& device,
    const std::shared_ptr<const ngraph::Function>& function,
    const InferenceEngine::InputsDataMap& inputInfoMap,
    const InferenceEngine::OutputsDataMap& outputsInfoMap,
    const Configuration& config) const {
    auto transformed_function = ngraph::clone_function(*function);

    auto passConfig = std::make_shared<ngraph::pass::PassConfig>();
    ngraph::pass::Manager manager{passConfig};

    passConfig->enable<ngraph::pass::ConvertInterpolate1ToInterpolate4>();
    passConfig->disable<ngraph::pass::MVN6Decomposition>();
    passConfig->disable<ov::pass::ConvertCompressedOnlyToLegacy>();

    // NOTE: Elementwise decompositions are now disabled because generally their straightforward versions
    // are executed faster on CUDA/cuDNN.
    // However this is not valid for the case with broadcasting of very large shapes (e.g. {{1024, 1024, 384, 2}, {1}})
    // on CUDA, for them decomposed cuDNN versions are faster.
    // TODO: Consider as possible optimisations: enabling these decompositions for large shapes, creating cuDNN versions
    // for these operations, implementing in-place logic in CUDA plugin for these operations.
    passConfig->disable<ngraph::pass::ConvertSubtract>();
    passConfig->disable<ngraph::pass::ConvertDivide>();
    passConfig->disable<ngraph::pass::ConvertMod>();

    [[maybe_unused]] const auto& originOps = function->get_ordered_ops();
    [[maybe_unused]] const auto& originOpsSize = originOps.size();

    manager.register_pass<ngraph::pass::InitNodeInfo>();
    manager.register_pass<ngraph::pass::AddPreprocessing>(inputInfoMap);
    manager.register_pass<ngraph::pass::CommonOptimizations>();
    // NOTE: G-API supports only FP32 networks for pre-processing
    //       CUDAPlugin supports FP16 networks, but this transformation is needed for export
    bool needF16toF32 = false;
    for (const auto& param : function->get_parameters()) {
        if (param->get_element_type() == ngraph::element::f16 &&
            inputInfoMap.at(param->get_friendly_name())->getTensorDesc().getPrecision() !=
                InferenceEngine::Precision::FP16) {
            needF16toF32 = true;
            break;
        }
    }
    if (needF16toF32) {
        manager.register_pass<ngraph::pass::ConvertPrecision>(
            precisions_array{{ngraph::element::f16, ngraph::element::f32}});
    }

    manager.run_passes(transformed_function);

    [[maybe_unused]] const auto& transformedOps = transformed_function->get_ordered_ops();
    [[maybe_unused]] const auto& transformedOpsSize = transformedOps.size();

    return transformed_function;
}

std::shared_ptr<ngraph::Function> GraphTransformer::transform(const CUDA::Device& device,
                                                              const std::shared_ptr<const ngraph::Function>& function,
                                                              const InferenceEngine::InputsDataMap& inputInfoMap,
                                                              const InferenceEngine::OutputsDataMap& outputsInfoMap,
                                                              const Configuration& config) const {
    auto transformed_function = ngraph::clone_function(*function);

    auto passConfig = std::make_shared<ngraph::pass::PassConfig>();
    ngraph::pass::Manager manager{passConfig};

    passConfig->enable<ngraph::pass::ConvertInterpolate1ToInterpolate4>();
    passConfig->disable<ngraph::pass::MVN6Decomposition>();
    passConfig->disable<ov::pass::ConvertCompressedOnlyToLegacy>();

    // NOTE: Elementwise decompositions are now disabled because generally their straightforward versions
    // are executed faster on CUDA/cuDNN.
    // However this is not valid for the case with broadcasting of very large shapes (e.g. {{1024, 1024, 384, 2}, {1}})
    // on CUDA, for them decomposed cuDNN versions are faster.
    // TODO: Consider as possible optimisations: enabling these decompositions for large shapes, creating cuDNN versions
    // for these operations, implementing in-place logic in CUDA plugin for these operations.
    passConfig->disable<ngraph::pass::ConvertSubtract>();
    passConfig->disable<ngraph::pass::ConvertDivide>();
    passConfig->disable<ngraph::pass::ConvertMod>();

    [[maybe_unused]] const auto& originOps = function->get_ordered_ops();
    [[maybe_unused]] const auto& originOpsSize = originOps.size();

    manager.register_pass<ngraph::pass::InitNodeInfo>();
    manager.register_pass<ngraph::pass::AddPreprocessing>(inputInfoMap);
    manager.register_pass<ngraph::pass::CommonOptimizations>();
    manager.register_pass<ngraph::pass::RemoveDuplicatedResultsTransformation>();
    if (!isHalfSupported(device)) {
        manager.register_pass<ngraph::pass::ConvertPrecision>(ov::element::f16, ov::element::f32);
    }
    if (!isInt8Supported(device)) {
        manager.register_pass<ngraph::pass::ConvertPrecision>(
            ov::element::i8, isHalfSupported(device) ? ov::element::f16 : ov::element::f32);
    }
    manager.register_pass<ngraph::pass::RemoveRedundantConvertTransformation>();

    if (!config.disabled_tensoriterator_transform) {
        manager.register_pass<ngraph::pass::BidirectionalSequenceComposition>(passConfig);
    }
    manager.register_pass<ngraph::pass::ConvolutionAsymPaddingTransformation>();
    manager.register_pass<ngraph::pass::GroupConvolutionAsymPaddingTransformation>();
    manager.register_pass<ngraph::pass::CudaFuseConvBiasAddActivation>();
    manager.register_pass<ngraph::pass::CudaFuseGroupConvBiasAddActivation>();
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
