// Copyright (C) 2018-2026 Intel Corporation
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
#include "transformations/common_optimizations/nop_elimination.hpp"
#include "transformations/common_optimizations/shuffle_channels_fusion.hpp"
#include "transformations/convert_precision.hpp"
#include "transformations/fp16_compression/convert_compression_only_to_legacy.hpp"
#include "transformations/fp16_compression/mark_decompression_convert_constant_folding.hpp"
#include "transformations/init_node_info.hpp"
#include "transformations/op_conversions/bidirectional_sequences_decomposition.hpp"
#include "transformations/op_conversions/convert_mod.hpp"
#include "transformations/op_conversions/convert_reduce_to_pooling.hpp"
#include "transformations/op_conversions/convert_sequences_to_tensor_iterator.hpp"
#include "transformations/op_conversions/convert_ti_to_sequences.hpp"
#include "transformer/convolution_asym_padding_transformation.hpp"
#include "transformer/fuse_conv_biasadd_activation.hpp"

#include "bidirectional_lstm_sequence_composition.hpp"
#include "concat_transformation.hpp"
#include "detection_output_fix_input_types_transformation.hpp"
#include "fuse_matmul_add.hpp"
#include "matmul_transformations.hpp"
#include "reduce_transformation.hpp"
#include "remove_duplicated_results_transformation.hpp"
#include "remove_redundant_convert_transformation.hpp"
#include "transformations/op_conversions/convert_divide.hpp"
#include "transformations/op_conversions/convert_interpolate1_to_interpolate4.hpp"
#include "transformations/op_conversions/convert_subtract.hpp"
#include "transformations/op_conversions/convert_gelu.hpp"
#include "transformations/op_conversions/gelu7_downgrade.hpp"
#include "transformations/op_conversions/mvn6_decomposition.hpp"
#include "transformations/op_conversions/hswish_decomposition.hpp"
#include "transformations/common_optimizations/reshape_prelu.hpp"
#include "transformations/op_conversions/scaled_dot_product_attention_decomposition.hpp"
#include "transformations/op_conversions/convert_convertlike.hpp"
#include "transformations/op_conversions/convert_slice_to_strided_slice.hpp"

using namespace ov::nvidia_gpu;

void GraphTransformer::transform(const CUDA::Device& device,
                                 std::shared_ptr<ov::Model>& model,
                                 const Configuration& config) const {
    auto inference_precision = config.get_inference_precision();
    if (inference_precision == ov::element::f16 && !isHalfSupported(device)) {
        inference_precision = ov::element::f32;
    }

    auto upscale_precision = [&]() -> bool {
        return !isHalfSupported(device) || inference_precision == ov::element::f32;
    };
    auto downscale_precision = [&]() -> bool {
        return isHalfSupported(device) && inference_precision == ov::element::f16;
    };

    precisions_map fp_convert_precision_map = {
        {ov::element::f64, ov::element::f32}
    };
    type_to_fuse_map empty_fuse_map = {};
    if (upscale_precision()) {
        fp_convert_precision_map.insert(std::make_pair(ov::element::f16, ov::element::f32));
    } else if (downscale_precision()) {
        fp_convert_precision_map.insert(std::make_pair(ov::element::f32, ov::element::f16));
    }

    auto pass_config = std::make_shared<ov::pass::PassConfig>();
    ov::pass::Manager pass_manager{pass_config};

    pass_config->enable<ov::pass::ConvertInterpolate1ToInterpolate4>();
    pass_config->disable<ov::pass::MVN6Decomposition>();
    // NOTE: Elementwise decompositions are now disabled because generally their straightforward versions
    // are executed faster on CUDA/cuDNN.
    // However this is not valid for the case with broadcasting of very large shapes (e.g. {{1024, 1024, 384, 2}, {1}})
    // on CUDA, for them decomposed cuDNN versions are faster.
    // TODO: Consider as possible optimisations: enabling these decompositions for large shapes, creating cuDNN versions
    // for these operations, implementing in-place logic in NVIDIA GPU plugin for these operations.
    pass_config->disable<ov::pass::ConvertSubtract>();
    pass_config->disable<ov::pass::ConvertDivide>();
    pass_config->disable<ov::pass::ConvertMod>();
    pass_config->disable<ov::pass::Gelu7Downgrade>();
    pass_config->disable<ov::pass::ConvertGELU>();
    pass_config->disable<ov::pass::HSwishDecomposition>();
    pass_config->disable<ov::pass::ConvertReduceMaxToPooling>();
    pass_config->disable<ov::pass::ConvertReduceMeanToPooling>();
    pass_config->disable<ov::pass::ConvertReduceSumToPooling>();
    pass_config->disable<ov::pass::ShuffleChannelsFusion>();

    // Skip decomposition for LSTMSequence and GRUSequence
    pass_config->disable<ov::pass::BidirectionalLSTMSequenceDecomposition>();
    pass_config->disable<ov::pass::BidirectionalGRUSequenceDecomposition>();
    // TODO: Uncomment when support for RNNSequence will be added
    //pass_config->disable<ov::pass::BidirectionalRNNSequenceDecomposition>();

    [[maybe_unused]] const auto& originOps = model->get_ordered_ops();
    [[maybe_unused]] const auto& originOpsSize = originOps.size();

    pass_manager.register_pass<ov::pass::InitNodeInfo>();
    pass_manager.register_pass<ov::pass::ConvertPrecision>(fp_convert_precision_map, empty_fuse_map, true, false);
    pass_manager.register_pass<ov::pass::CommonOptimizations>();
    pass_manager.register_pass<ov::pass::ReshapePRelu>();
    // Do we actually need this transformations in plugin?
    // Having duplicated results seems to be rare case in real world.
    // But currently it affects the CudaInferRequest which implementation
    // relies on number of outputs of original model
    // pass_manager.register_pass<ov::nvidia_gpu::pass::RemoveDuplicatedResultsTransformation>();
    pass_manager.register_pass<ov::nvidia_gpu::pass::RemoveRedundantConvertTransformation>();
    pass_manager.register_pass<ov::nvidia_gpu::pass::BidirectionalSequenceComposition>();
    pass_manager.register_pass<ov::pass::ConvertSequenceToTensorIterator>();

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

    // Decompose ScaledDotProductAttention into elementary operations (MatMul, Softmax, etc.)
    // that are already supported by the plugin
    pass_manager.register_pass<ov::pass::ScaledDotProductAttentionDecomposition>();
    // Convert ops produced by SDPA decomposition to supported equivalents
    pass_manager.register_pass<ov::pass::ConvertConvertLike>();
    pass_manager.register_pass<ov::pass::SliceToStridedSlice>(true);

    pass_manager.register_pass<ov::nvidia_gpu::pass::ConvolutionAsymPaddingTransformation>();
    pass_manager.register_pass<ov::nvidia_gpu::pass::GroupConvolutionAsymPaddingTransformation>();
    pass_manager.register_pass<ov::nvidia_gpu::pass::CudaConvolutionFusion>();
    pass_manager.register_pass<ov::nvidia_gpu::pass::ConvolutionBackpropDataAsymPaddingTransformation>();
    pass_manager.register_pass<ov::nvidia_gpu::pass::GroupConvolutionBackpropDataAsymPaddingTransformation>();
    pass_manager.register_pass<ov::nvidia_gpu::pass::FusedConvBackpropDataAsymPaddingTransformation>();
    pass_manager.register_pass<ov::nvidia_gpu::pass::TransposeMatMulTransformation>();
    pass_manager.register_pass<ov::nvidia_gpu::pass::FullyConnectedTransformation>();
    pass_manager.register_pass<ov::nvidia_gpu::pass::ConcatTransformation>();
    pass_manager.register_pass<ov::nvidia_gpu::pass::ReduceTransformation>();
    pass_manager.register_pass<ov::nvidia_gpu::pass::DetectionOutputFixInputTypesTransformation>();

    // Do we actually need to eliminate broadcast one more time at the end?
    pass_manager.register_pass<ov::pass::NopElimination>();

    pass_manager.run_passes(model);

    [[maybe_unused]] const auto& transformedOps = model->get_ordered_ops();
    [[maybe_unused]] const auto& transformedOpsSize = transformedOps.size();

    return;
}
