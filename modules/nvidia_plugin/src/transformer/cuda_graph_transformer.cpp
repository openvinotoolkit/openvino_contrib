// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/cc/pass/itt.hpp"
#include "cuda_graph_transformer.hpp"

#include <fmt/format.h>

#include "openvino/core/preprocess/pre_post_process.hpp"
#include "openvino/pass/manager.hpp"
#include "openvino/op/gru_sequence.hpp"
#include "openvino/op/rnn_sequence.hpp"
#include "openvino/op/lstm_sequence.hpp"
#include "transformations/common_optimizations/common_optimizations.hpp"
#include "transformations/disable_decompression_convert_constant_folding.hpp"
#include "transformations/common_optimizations/nop_elimination.hpp"
#include "transformations/convert_precision.hpp"
#include "transformations/init_node_info.hpp"
#include "transformations/op_conversions/bidirectional_sequences_decomposition.hpp"
#include "transformations/op_conversions/convert_mod.hpp"
#include "transformations/op_conversions/convert_sequences_to_tensor_iterator.hpp"
#include "transformations/op_conversions/convert_ti_to_sequences.hpp"
#include "transformer/convolution_asym_padding_transformation.hpp"
#include "transformer/fuse_conv_biasadd_activation.hpp"

#include "bidirectional_lstm_sequence_composition.hpp"
#include "concat_transformation.hpp"
#include "cuda_fullyconnected_transformation.hpp"
#include "matmul_transformations.hpp"
#include "noop_broadcast_transformation.hpp"
#include "remove_duplicated_results_transformation.hpp"
#include "remove_redundant_convert_transformation.hpp"
#include "transformations/common_optimizations/convert_compression_only_to_legacy.hpp"
#include "transformations/op_conversions/convert_divide.hpp"
#include "transformations/op_conversions/convert_interpolate1_to_interpolate4.hpp"
#include "transformations/op_conversions/convert_subtract.hpp"
#include "transformations/op_conversions/mvn6_decomposition.hpp"
#include "transformations/common_optimizations/reshape_prelu.hpp"

using namespace ov::nvidia_gpu;

void GraphTransformer::transform(const CUDA::Device& device,
                                 std::shared_ptr<ov::Model>& model,
                                 const Configuration& config) const {
    auto inference_precision = config.get_inference_precision();
    if (inference_precision == ov::element::f16 && !isHalfSupported(device)) {
        throw_ov_exception("Inference precision f16 is not supported by device!");
    }
    auto upscale_precision = [&]() -> bool {
        return !isHalfSupported(device) || inference_precision == ov::element::f32;
    };
    auto downscale_precision = [&]() -> bool {
        return isHalfSupported(device) && inference_precision == ov::element::f16;
    };

    // WA for ConvertPrecision which doesn't keep original precisions (CVS-111453)
    // Store original precisions for inputs and outputs
    // And restore them after transformations if ConvertPrecision was called
    std::map<size_t, ov::element::Type> input_preprocess;
    std::map<size_t, ov::element::Type> output_postprocess;
    for (size_t i = 0; i < model->inputs().size(); i++) {
        auto input = model->input(i);
        input_preprocess[i] = input.get_element_type();
    }
    for (size_t i = 0; i < model->get_output_size(); i++) {
        auto output = model->output(i);
        output_postprocess[i] = output.get_element_type();
    }
    auto pass_config = std::make_shared<ov::pass::PassConfig>();
    ov::pass::Manager common_manager{pass_config};

    pass_config->enable<ov::pass::ConvertInterpolate1ToInterpolate4>();
    pass_config->disable<ov::pass::MVN6Decomposition>();
    if (upscale_precision()) {
        // Allow FP16 Converts to be folded and FP16 constants to be upgraded to FP32 data type
        pass_config->disable<ov::pass::DisableDecompressionConvertConstantFolding>();
        pass_config->disable<ov::pass::ConvertCompressedOnlyToLegacy>();
    }

    // NOTE: Elementwise decompositions are now disabled because generally their straightforward versions
    // are executed faster on CUDA/cuDNN.
    // However this is not valid for the case with broadcasting of very large shapes (e.g. {{1024, 1024, 384, 2}, {1}})
    // on CUDA, for them decomposed cuDNN versions are faster.
    // TODO: Consider as possible optimisations: enabling these decompositions for large shapes, creating cuDNN versions
    // for these operations, implementing in-place logic in NVIDIA GPU plugin for these operations.
    pass_config->disable<ov::pass::ConvertSubtract>();
    pass_config->disable<ov::pass::ConvertDivide>();
    pass_config->disable<ov::pass::ConvertMod>();

    [[maybe_unused]] const auto& originOps = model->get_ordered_ops();
    [[maybe_unused]] const auto& originOpsSize = originOps.size();

    common_manager.register_pass<ov::pass::InitNodeInfo>();
    common_manager.register_pass<ov::pass::CommonOptimizations>();
    common_manager.register_pass<ov::pass::ReshapePRelu>();
    // Do we actually need this transformations in plugin?
    // Having duplicated results seems to be rare case in real world.
    // But currently it affects the following places:
    // 1. HETERO plugin creates separate Result operation for each subset of inputs
    //    if subset belongs to other subgraph - it should be fixed, if we
    //    want to avoid extra output processing (CVS-111877)
    // 2. CudaInferRequest implementation relies on number of outputs of original model
    // 3. WA for ConvertPrecision - will removed after fixing CVS-111453
    // common_manager.register_pass<ov::nvidia_gpu::pass::RemoveDuplicatedResultsTransformation>();
    precisions_map fp_convert_precision_map = {
        {ov::element::f64, ov::element::f32}
    };
    type_to_fuse_map empty_fuse_map = {};
    if (upscale_precision()) {
        fp_convert_precision_map.insert(std::make_pair(ov::element::f16, ov::element::f32));
    } else if (downscale_precision()) {
        fp_convert_precision_map.insert(std::make_pair(ov::element::f32, ov::element::f16));
    }
    common_manager.register_pass<ov::pass::ConvertPrecision>(fp_convert_precision_map, empty_fuse_map, true);
    common_manager.run_passes(model);

    auto preprocessor = ov::preprocess::PrePostProcessor(model);
    for (auto& item : input_preprocess) {
        auto& in = preprocessor.input(item.first);
        in.tensor().set_element_type(item.second);
    }
    for (auto& item : output_postprocess) {
        auto& out = preprocessor.output(item.first);
        out.tensor().set_element_type(item.second);
    }
    model = preprocessor.build();

    ov::pass::Manager plugin_manager{pass_config};
    plugin_manager.register_pass<ov::nvidia_gpu::pass::RemoveRedundantConvertTransformation>();
    plugin_manager.register_pass<ov::nvidia_gpu::pass::BidirectionalSequenceComposition>(pass_config);
    plugin_manager.register_pass<ov::pass::ConvertSequenceToTensorIterator>();

    // Sequences supported by the plugin shouldn't be converted to TensorIterator.
    auto is_sequence_primitive_supported = [](const std::shared_ptr<const ov::Node> &node) -> bool {
        if (std::dynamic_pointer_cast<const ov::op::v5::RNNSequence>(node)) {
            return false;
        } else if (const auto &gru_seq = std::dynamic_pointer_cast<const ov::op::v5::GRUSequence>(node)) {
            return (gru_seq->get_clip() == 0.0f &&
                    gru_seq->get_activations() == std::vector<std::string>{"sigmoid", "tanh"} &&
                    (gru_seq->get_input_size() != 1 || gru_seq->get_hidden_size() != 1) &&
                    (gru_seq->get_direction() != ov::op::RecurrentSequenceDirection::REVERSE) &&
                    (gru_seq->get_direction() != ov::op::RecurrentSequenceDirection::BIDIRECTIONAL));
        } else if (const auto &lstm_seq = std::dynamic_pointer_cast<const ov::op::v5::LSTMSequence>(node)) {
            return (lstm_seq->get_clip() == 0.0f &&
                    lstm_seq->get_activations() == std::vector<std::string>{"sigmoid", "tanh", "tanh"} &&
                    lstm_seq->get_activations_alpha() == std::vector<float>{1.0f, 1.0f, 1.0f} &&
                    lstm_seq->get_activations_beta() == std::vector<float>{0.0f, 0.0f, 0.0f} &&
                    (lstm_seq->get_input_size() != 1 || lstm_seq->get_hidden_size() != 1) &&
                    (lstm_seq->get_direction() != ov::op::RecurrentSequenceDirection::REVERSE));
        }
        return false;
    };

    pass_config->set_callback<ov::pass::ConvertRNNSequenceToTensorIterator,
                                ov::pass::ConvertGRUSequenceToTensorIterator,
                                ov::pass::ConvertLSTMSequenceToTensorIterator>(
            [is_sequence_primitive_supported](const std::shared_ptr<const ov::Node> &node) -> bool {
                return is_sequence_primitive_supported(node);
            });

    plugin_manager.register_pass<ov::nvidia_gpu::pass::ConvolutionAsymPaddingTransformation>();
    plugin_manager.register_pass<ov::nvidia_gpu::pass::GroupConvolutionAsymPaddingTransformation>();
    plugin_manager.register_pass<ov::nvidia_gpu::pass::CudaConvolutionFusion>();
    plugin_manager.register_pass<ov::nvidia_gpu::pass::ConvolutionBackpropDataAsymPaddingTransformation>();
    plugin_manager.register_pass<ov::nvidia_gpu::pass::GroupConvolutionBackpropDataAsymPaddingTransformation>();
    plugin_manager.register_pass<ov::nvidia_gpu::pass::FusedConvBackpropDataAsymPaddingTransformation>();
    plugin_manager.register_pass<ov::nvidia_gpu::pass::TransposeMatMulTransformation>();
    plugin_manager.register_pass<ov::nvidia_gpu::pass::FullyConnectedTransformation>();
    plugin_manager.register_pass<ov::nvidia_gpu::pass::ConcatTransformation>();
    plugin_manager.register_pass<ov::nvidia_gpu::pass::NoopBroadcastTransformation>();

    plugin_manager.run_passes(model);

    [[maybe_unused]] const auto& transformedOps = model->get_ordered_ops();
    [[maybe_unused]] const auto& transformedOpsSize = transformedOps.size();

    return;
}