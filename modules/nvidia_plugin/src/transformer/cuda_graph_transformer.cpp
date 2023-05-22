// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/cc/ngraph/itt.hpp"
#include "cuda_graph_transformer.hpp"

#include <fmt/format.h>

#include "openvino/openvino.hpp"
#include "openvino/pass/manager.hpp"
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
#include "nvidia/nvidia_config.hpp"
#include "remove_duplicated_results_transformation.hpp"
#include "remove_redundant_convert_transformation.hpp"
#include "transformations/common_optimizations/convert_compression_only_to_legacy.hpp"
#include "transformations/op_conversions/convert_divide.hpp"
#include "transformations/op_conversions/convert_interpolate1_to_interpolate4.hpp"
#include "transformations/op_conversions/convert_subtract.hpp"
#include "transformations/op_conversions/mvn6_decomposition.hpp"
#include "transformations/common_optimizations/reshape_prelu.hpp"
#include "ngraph/opsets/opset10.hpp"

using namespace ov::nvidia_gpu;

void GraphTransformer::transform(const CUDA::Device& device,
                                 std::shared_ptr<ov::Model>& model,
                                 const Configuration& config) const {
    auto inference_precision = config.get_inference_precision();
    if (inference_precision == ov::element::f16 && !isHalfSupported(device)) {
        throwIEException("Inference precision f16 is not supported by device!");
    }
    auto upscale_precision = [&]() -> bool {
        return !isHalfSupported(device) || inference_precision == ov::element::f32;
    };
    auto downscale_precision = [&]() -> bool {
        return isHalfSupported(device) && inference_precision == ov::element::f16;
    };

    bool convert_called = false;

    auto passConfig = std::make_shared<ov::pass::PassConfig>();
    ov::pass::Manager manager{passConfig};

    passConfig->enable<ov::pass::ConvertInterpolate1ToInterpolate4>();
    passConfig->disable<ov::pass::MVN6Decomposition>();
    if (upscale_precision()) {
        // Allow FP16 Converts to be folded and FP16 constants to be upgraded to FP32 data type
        passConfig->disable<ov::pass::DisableDecompressionConvertConstantFolding>();
        passConfig->disable<ov::pass::ConvertCompressedOnlyToLegacy>();
    }

    // NOTE: Elementwise decompositions are now disabled because generally their straightforward versions
    // are executed faster on CUDA/cuDNN.
    // However this is not valid for the case with broadcasting of very large shapes (e.g. {{1024, 1024, 384, 2}, {1}})
    // on CUDA, for them decomposed cuDNN versions are faster.
    // TODO: Consider as possible optimisations: enabling these decompositions for large shapes, creating cuDNN versions
    // for these operations, implementing in-place logic in NVIDIA GPU plugin for these operations.
    passConfig->disable<ov::pass::ConvertSubtract>();
    passConfig->disable<ov::pass::ConvertDivide>();
    passConfig->disable<ov::pass::ConvertMod>();

    [[maybe_unused]] const auto& originOps = model->get_ordered_ops();
    [[maybe_unused]] const auto& originOpsSize = originOps.size();

    manager.register_pass<ov::pass::InitNodeInfo>();
    manager.register_pass<ov::pass::CommonOptimizations>();
    manager.register_pass<ov::pass::ReshapePRelu>();
    manager.register_pass<ov::nvidia_gpu::pass::RemoveDuplicatedResultsTransformation>();
    if (upscale_precision()) {
        convert_called = true;
        manager.register_pass<ov::pass::ConvertPrecision>(ov::element::f16, ov::element::f32);
    } else {
        if (downscale_precision()) {
            convert_called = true;
            manager.register_pass<ov::pass::ConvertPrecision>(ov::element::f32, ov::element::f16, type_to_fuse_map{}, true);
        }
    }
    manager.register_pass<ov::nvidia_gpu::pass::RemoveRedundantConvertTransformation>();
    manager.register_pass<ov::nvidia_gpu::pass::BidirectionalSequenceComposition>(passConfig);
    manager.register_pass<ov::pass::ConvertSequenceToTensorIterator>();

    // Sequences supported by the plugin shouldn't be converted to TensorIterator.
    auto is_sequence_primitive_supported = [](const std::shared_ptr<const ov::Node> &node) -> bool {
        if (std::dynamic_pointer_cast<const ngraph::opset10::RNNSequence>(node)) {
            return false;
        } else if (const auto &gru_seq = std::dynamic_pointer_cast<const ngraph::opset10::GRUSequence>(node)) {
            return (gru_seq->get_clip() == 0.0f &&
                    gru_seq->get_activations() == std::vector<std::string>{"sigmoid", "tanh"} &&
                    (gru_seq->get_input_size() != 1 || gru_seq->get_hidden_size() != 1) &&
                    (gru_seq->get_direction() != ov::op::RecurrentSequenceDirection::REVERSE) &&
                    (gru_seq->get_direction() != ov::op::RecurrentSequenceDirection::BIDIRECTIONAL));
        } else if (const auto &lstm_seq = std::dynamic_pointer_cast<const ngraph::opset10::LSTMSequence>(node)) {
            return (lstm_seq->get_clip() == 0.0f &&
                    lstm_seq->get_activations() == std::vector<std::string>{"sigmoid", "tanh", "tanh"} &&
                    lstm_seq->get_activations_alpha() == std::vector<float>{1.0f, 1.0f, 1.0f} &&
                    lstm_seq->get_activations_beta() == std::vector<float>{0.0f, 0.0f, 0.0f} &&
                    (lstm_seq->get_input_size() != 1 || lstm_seq->get_hidden_size() != 1) &&
                    (lstm_seq->get_direction() != ov::op::RecurrentSequenceDirection::REVERSE));
        }
        return false;
    };

    passConfig->set_callback<ov::pass::ConvertRNNSequenceToTensorIterator,
                             ov::pass::ConvertGRUSequenceToTensorIterator,
                             ov::pass::ConvertLSTMSequenceToTensorIterator>(
            [is_sequence_primitive_supported](const std::shared_ptr<const ov::Node> &node) -> bool {
                return is_sequence_primitive_supported(node);
            });

    manager.register_pass<ov::nvidia_gpu::pass::ConvolutionAsymPaddingTransformation>();
    manager.register_pass<ov::nvidia_gpu::pass::GroupConvolutionAsymPaddingTransformation>();
    manager.register_pass<ov::nvidia_gpu::pass::CudaConvolutionFusion>();
    manager.register_pass<ov::nvidia_gpu::pass::ConvolutionBackpropDataAsymPaddingTransformation>();
    manager.register_pass<ov::nvidia_gpu::pass::GroupConvolutionBackpropDataAsymPaddingTransformation>();
    manager.register_pass<ov::nvidia_gpu::pass::FusedConvBackpropDataAsymPaddingTransformation>();
    manager.register_pass<ov::nvidia_gpu::pass::TransposeMatMulTransformation>();
    manager.register_pass<ov::nvidia_gpu::pass::FullyConnectedTransformation>();
    manager.register_pass<ov::nvidia_gpu::pass::ConcatTransformation>();
    manager.register_pass<ov::nvidia_gpu::pass::NoopBroadcastTransformation>();

    // WA for ConvertPrecision which doesn't keep original precisions (CVS-111453)
    // Store original precisions for inputs and outputs
    // And restore them after transformations if ConvertPrecision was called
    std::map<size_t, ov::element::Type> input_direct;
    std::map<size_t, ov::element::Type> input_preprocess;
    std::map<size_t, ov::element::Type> output_postprocess;
    if (convert_called) {
        for (size_t i = 0; i < model->inputs().size(); i++) {
            auto input = model->input(i);
            auto target_inputs = input.get_target_inputs();
            if (all_of(target_inputs.begin(), target_inputs.end(),
                [&] (const Input<Node>& i) {return ov::op::util::is_output(i.get_node());})) {
                input_direct[i] = input.get_element_type();
            } else {
                input_preprocess[i] = input.get_element_type();
            }
        }
        for (size_t i = 0; i < model->get_output_size(); i++) {
            auto output = model->output(i);
            if (!ov::op::util::is_parameter(output.get_node())) {
                output_postprocess[i] = output.get_element_type();
            }
        }
    }

    manager.run_passes(model);

    if (convert_called) {
        for (auto& item : input_direct) {
            auto node = model->input(item.first).get_node_shared_ptr();
            if (auto param = ov::as_type_ptr<ov::op::v0::Parameter>(node)) {
                param->set_element_type(item.second);
                param->validate_and_infer_types();
            }
        }
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
    }

    [[maybe_unused]] const auto& transformedOps = model->get_ordered_ops();
    [[maybe_unused]] const auto& transformedOpsSize = transformedOps.size();

    return;
}