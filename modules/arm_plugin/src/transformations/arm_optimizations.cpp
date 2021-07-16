// Copyright (C) 2020-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0


#include "transformations/common_optimizations/algebraic_simplification.hpp"
#include "transformations/common_optimizations/nop_elimination.hpp"
#include "transformations/common_optimizations/conv_mul_fusion.hpp"
#include "transformations/convert_precision.hpp"
#include "transformations/init_node_info.hpp"
#include "transformations/decompose_variadic_split.hpp"
#include "transformations/common_optimizations/softplus_fusion.hpp"
#include "transformations/op_conversions/convert_mod.hpp"
#include "transformations/op_conversions/convert_negative.hpp"
#include "transformations/op_conversions/convert_divide.hpp"
#include "transformations/op_conversions/convert_reduce_to_pooling.hpp"
#include "transformations/op_conversions/convert_broadcast3.hpp"
#include "transformations/op_conversions/convert_broadcast_to_tiles.hpp"
#include "transformations/op_conversions/rnn_cell_decomposition.hpp"
#include "transformations/op_conversions/lstm_cell_decomposition.hpp"
#include "transformations/op_conversions/gru_cell_decomposition.hpp"
#include "transformations/common_optimizations/lin_op_sequence_fusion.hpp"
#include "transformations/op_conversions/reduce_l1_decomposition.hpp"
#include "transformations/op_conversions/reduce_l2_decomposition.hpp"
#include "transformations/op_conversions/log_softmax_decomposition.hpp"
#include "transformations/common_optimizations/remove_filtering_boxes_by_size.hpp"
#include "transformations/common_optimizations/hswish_fusion.hpp"
#include "transformations/op_conversions/convert_interpolate1_to_interpolate4.hpp"
#include "transformations/op_conversions/convert_mvn1_to_mvn6.hpp"
#include "transformations/op_conversions/convert_gelu.hpp"
#include "transformations/op_conversions/convert_ti_to_sequences.hpp"
#include "transformations/common_optimizations/weights_dequantize_to_fake_quantize.hpp"
#include "transformations/common_optimizations/convert_quantize_dequantize.hpp"
#include "transformations/op_conversions/convert_subtract.hpp"

#include "conv_bias_fusion.hpp"
#include "convert_eltwise.hpp"
#include "convert_sign.hpp"
#include "convert_round.hpp"
#include "convert_comparison.hpp"
#include "convert_logical.hpp"
#include "convert_strided_slice.hpp"
#include "convert_strided_slice_arm.hpp"
#include "convert_group_conv.hpp"
#include "convert_conv1d_to_conv2d.hpp"
#include "convert_grn_to_normalizel2.hpp"
#include "convert_mat_mul.hpp"
#include "convert_batchnorm_v0_to_v5.hpp"
#include "convert_batch_norm.hpp"
#include "convert_ceiling.hpp"
#include "convert_convert.hpp"
#include "convert_split.hpp"
#include "convert_concat.hpp"
#include "decompose_swish.hpp"
#include "convert_shuffle_channels.hpp"
#include "convert_tile_to_concats.hpp"
#include "convert_transpose_arm.hpp"
#include "convert_prelu.hpp"
#include "convert_gather_arm.hpp"
#include "convert_mvn_arm.hpp"
#include "convert_reduce_multi_axis.hpp"
#include "convert_select.hpp"
#include "normalizel2_max_fusion.hpp"
#include "decompose_normalizel2_add.hpp"
#include "decompose_mish.hpp"
#include "convert_interpolate_arm.hpp"
#include "convert_normalizel2_arm.hpp"
#include "convert_fft_arm.hpp"
#include "convert_pool1d_to_pool2d.hpp"
#include "convert_inputs_precision.hpp"
#include "finalize_trailing_nodes.hpp"
#include "transformations/convert_reorg.hpp"
#include "quantize_fusion.hpp"
#include "store_result_name.hpp"

#include <ngraph/pass/manager.hpp>
#include <ngraph/pass/constant_folding.hpp>

#include <transformations/low_precision/disable_convert_constant_folding_on_const_path.hpp>
#include <low_precision/transformer.hpp>
#include <low_precision/mat_mul.hpp>
#include <low_precision/strided_slice.hpp>
#include <low_precision/network_helper.hpp>
#include <low_precision/convolution.hpp>
#include <low_precision/group_convolution.hpp>
#include <low_precision/multiply_to_group_convolution.hpp>
#include <low_precision/fake_quantize_decomposition.hpp>
#include <low_precision/add.hpp>
#include <low_precision/multiply.hpp>
#include <low_precision/fake_quantize.hpp>
#include <low_precision/fold_convert.hpp>
#include <low_precision/fuse_convert.hpp>
#include <low_precision/fuse_multiply_to_fake_quantize.hpp>
#include <low_precision/fuse_subtract_to_fake_quantize.hpp>
#include "transformations/serialize.hpp"


#include "arm_optimizations.hpp"

void ArmPlugin::pass::ArmOptimizations::Dump(const std::shared_ptr<ngraph::Function>& f, const std::string& postfix) {
    if (_dump) {
        ngraph::pass::VisualizeTree{f->get_friendly_name() + "_" + postfix +
        (_lpt ? std::string{"_lpt"} : std::string{""}) + ".dot",
        [&] (const ngraph::Node& node, std::vector<std::string>& attributes) {
            auto& rt_info = node.get_rt_info();
            auto itInfo = rt_info.find("QuantizationInfo");
            if (itInfo != rt_info.end()) {
                std::stringstream strm;
                auto printVec = [&] (const std::string& name, auto& vec) {
                    strm << "\\n" + name + ": [";
                    int i = 0;
                    for (auto&& v : vec) {
                        if (i > 5) {
                            strm << "...";
                            break;
                        } else {
                            strm << v << ", ";
                            i++;
                        }
                    }
                    strm << "]";
                };
                const auto& quantizationInfo = std::dynamic_pointer_cast<ngraph::VariantImpl<arm_compute::QuantizationInfo>>(itInfo->second)->get();
                printVec("Scale", quantizationInfo.scale());
                printVec("Offset", quantizationInfo.offset());

                auto itLabel = std::find_if(std::begin(attributes), std::end(attributes), [] (const std::string& str) {
                    return str.find("label") != std::string::npos;
                });
                IE_ASSERT(itLabel != attributes.end());
                itLabel->pop_back();
                (*itLabel) += strm.str() + '\"';
            }
        }}.run_on_function(f);
    }
}

bool ArmPlugin::pass::ArmOptimizations::run_on_function(std::shared_ptr<ngraph::Function> f) {
    ngraph::pass::Manager manager;

    Dump(f, "initial");

    auto quantized = _lpt && ngraph::pass::low_precision::LowPrecisionTransformer::isFunctionQuantized(f);

    if (quantized) {
        manager.register_pass<ngraph::pass::DisableConvertConstantFoldingOnConstPath>(
            std::vector<ngraph::element::Type>{ ngraph::element::i8, ngraph::element::u8 });
    }

    // This pass must be called first in pipeline
    manager.register_pass<ngraph::pass::InitNodeInfo>();
    manager.register_pass<pass::StoreResultName>();
    manager.register_pass<ngraph::pass::RemoveFilteringBoxesBySize>(); // Resolves dynamism (replaces NonZero), CF needed
    manager.register_pass<ngraph::pass::ConstantFolding>();
    manager.register_pass<ngraph::pass::NopElimination>(); // may introduce fake dynamism
    manager.register_pass<ngraph::pass::AlgebraicSimplification>(); // may introduce fake dynamism
    manager.register_pass<ngraph::pass::ConstantFolding>();
    manager.register_pass<ngraph::pass::SoftPlusFusion>();
    manager.register_pass<ngraph::pass::HSwishFusion>();

    // LinOpSequenceFusion must be executed after all decompositions
    manager.register_pass<ngraph::pass::LinOpSequenceFusion>();
    manager.register_pass<ngraph::pass::RNNCellDecomposition>();
    manager.register_pass<ngraph::pass::LSTMCellDecomposition>();
    manager.register_pass<ngraph::pass::GRUCellDecomposition>();
    manager.register_pass<ngraph::pass::ConvertGELU>();
    manager.register_pass<ngraph::pass::ConstantFolding>();
    manager.register_pass<pass::ConvertConv1D>();
    manager.register_pass<pass::ConvertGroupConv1D>();
    manager.register_pass<pass::ConvertGroupConvolution>();
    manager.register_pass<ngraph::pass::ConstantFolding>();
    manager.register_pass<ngraph::pass::ConvolutionMultiplyFusion>();
    manager.register_pass<ngraph::pass::GroupConvolutionMultiplyFusion>();
    manager.register_pass<ngraph::pass::ConvolutionBackpropDataMultiplyFusion>();
    manager.register_pass<ngraph::pass::GroupConvolutionBackpropDataMultiplyFusion>();
    manager.register_pass<ngraph::pass::ConvertTensorIteratorToGRUSequence>();
    manager.register_pass<ngraph::pass::ConvertTensorIteratorToLSTMSequence>();
    manager.register_pass<ngraph::pass::ConvertTensorIteratorToRNNSequence>();
    manager.register_pass<ngraph::pass::ConstantFolding>();


    manager.register_pass<ngraph::pass::ConvertInterpolate1ToInterpolate4>();
    manager.register_pass<ngraph::pass::ConvertMVN1ToMVN6>();
    manager.register_pass<ngraph::pass::ConvertQuantizeDequantize>();
    #ifndef __ARM_FEATURE_FP16_VECTOR_ARITHMETIC
        manager.register_pass<ngraph::pass::ConvertPrecision>(ngraph::element::f16, ngraph::element::f32);
    #endif

    auto pass_config = manager.get_pass_config();

    using ConstNodePtr = const std::shared_ptr<const ngraph::Node>;

    if (quantized) {
        pass_config->set_callback<ngraph::pass::ConvertQuantizeDequantize>([](ConstNodePtr& node) -> bool {
            return ngraph::pass::low_precision::NetworkHelper::areQuantizeAndDequantizeSupportedForMultiply(node);
        });

        pass_config->set_callback<ngraph::pass::ConvertSubtract>([](ConstNodePtr& node) -> bool {
            return ngraph::pass::low_precision::NetworkHelper::areQuantizeAndDequantizeSupportedForSubtract(node);
        });
    }

    manager.run_passes(f);


    if (quantized) {
        Dump(f, "before_common");
        using namespace ngraph::pass::low_precision;
        auto params = LayerTransformation::Params(
            true,  // updatePrecisions
            LayerTransformation::QuantizedTensorAlignment::UpdateLevel,  // quantizedTensorAlignmentOnActivations
            LayerTransformation::QuantizedTensorAlignment::None,  // quantizedTensorAlignmentOnWeights
            true)
            .setPrecisionsOnActivations({ngraph::element::i8});

        LowPrecisionTransformer transformer(
            LowPrecisionTransformer::getAllTransformations(params)
            .removeStandaloneCleanup<MultiplyToGroupConvolutionTransformation, opset::Multiply>()
            .removeCleanup<FoldConvertTransformation, opset::Subtract>()
            .removeCleanup<FuseConvertTransformation, opset::Multiply>()
            .removeStandaloneCleanup<FuseSubtractToFakeQuantizeTransformation, opset::Subtract>()
            .removeStandaloneCleanup<FuseMultiplyToFakeQuantizeTransformation, opset::Multiply>());

        transformer.transform(f);
    }


    {
        Dump(f, "before_arm_specific_transformations");
        ngraph::pass::Manager manager;
        manager.register_pass<ngraph::pass::LogSoftmaxDecomposition>();
        manager.register_pass<pass::ConvertGRN>();
        manager.register_pass<pass::NormalizeL2Fusion>();
        manager.register_pass<pass::DecomposeNormalizeL2Add>();
        manager.register_pass<pass::ConvertNormalizeL2ToArm>();
        manager.register_pass<pass::ConvertReduceMultiAxis>();
        manager.register_pass<ngraph::pass::ReduceL1Decomposition>();
        manager.register_pass<ngraph::pass::ReduceL2Decomposition>();
        manager.register_pass<ngraph::pass::ConvertReduceMeanToPooling>();
        manager.register_pass<ngraph::pass::ConvertReduceMaxToPooling>();
        manager.register_pass<ngraph::pass::ConvertReduceSumToPooling>();
        manager.register_pass<ngraph::pass::ConvertMod>();
        manager.register_pass<ngraph::pass::ConstantFolding>();
        manager.register_pass<pass::DecomposeSwish>();
        manager.register_pass<pass::DecomposeMish>();
        manager.register_pass<pass::BroadcastPRelu>();
        manager.register_pass<pass::ConvertLogical>();
        manager.register_pass<pass::ConvertComparison>();
        manager.register_pass<pass::ConvertTranspose>();
        manager.register_pass<ngraph::pass::ConstantFolding>();
        manager.register_pass<pass::ConvertRound>();
        manager.register_pass<pass::ConvertSign>();
        manager.register_pass<pass::ConvertCeiling>();
        manager.register_pass<pass::DecomposeVariadicSplit>();
        manager.register_pass<pass::ConvertStridedSliceToArm>();
        manager.register_pass<pass::ConvertStridedSlice>();
        manager.register_pass<pass::ConvertBatchNormInferenceV0toV5>();
        manager.register_pass<pass::ConvertBatchNormInference>();
        manager.register_pass<pass::ConvertShuffleChannels>();
        manager.register_pass<pass::ConvertInterpolate>();
        manager.register_pass<pass::ConvertMVN>();
        manager.register_pass<pass::ConvertReorgYolo>();
        manager.register_pass<pass::ConvertMaxPool1D>();
        manager.register_pass<pass::ConvertAvgPool1D>();
        manager.register_pass<pass::BroadcastSelect>();
        manager.register_pass<pass::ConvertGather>();
        manager.register_pass<pass::ConvertDFT>();
        manager.register_pass<pass::ConvertIDFT>();
        manager.register_pass<ngraph::pass::ConstantFolding>();
        manager.register_pass<pass::ConvBiasFusion>();
        manager.register_pass<ngraph::pass::ConstantFolding>();
        manager.register_pass<pass::ConvertMatMulToFC>();
        manager.register_pass<ngraph::pass::ConstantFolding>();
        manager.register_pass<pass::ConvertArmConvert>();
        manager.register_pass<pass::ConvertArmConvertLike>();
        manager.register_pass<ngraph::pass::ConstantFolding>();
        manager.register_pass<ngraph::pass::ConvertDivide>();
        manager.register_pass<ngraph::pass::ConvertBroadcast3>();
        manager.register_pass<ngraph::pass::ConvertBroadcastToTiles>();
        manager.register_pass<pass::ConvertEltwise>();
        manager.register_pass<ngraph::pass::ConstantFolding>();
        manager.register_pass<pass::ConvertTile>();
        manager.register_pass<pass::ConvertSplit>();
        manager.register_pass<pass::ConvertConcat>();
        manager.register_pass<pass::FinalizeTrailingNodes>();
        manager.register_pass<pass::StoreResultName>();
        manager.register_pass<ngraph::pass::ConstantFolding>();
        manager.register_pass<ngraph::pass::ConvertPrecision>(ngraph::element::boolean, ngraph::element::u8);
        manager.register_pass<ngraph::pass::ConvertPrecision>(ngraph::element::i64, ngraph::element::i32);
        manager.register_pass<ngraph::pass::ConvertPrecision>(ngraph::element::u64, ngraph::element::i32);
        manager.register_pass<pass::AlignNodePrecision>();
        manager.register_pass<ngraph::pass::ConstantFolding>();
        manager.run_passes(f);
    }


    if (quantized) {
        Dump(f, "before_arm");
        ngraph::pass::Manager manager;
        manager.register_pass<pass::NodeQuantizeFusion>();
        manager.register_pass<pass::DequantizeNodeFusion>();
        manager.register_pass<pass::AddDequantizeOnInputs>();
        manager.register_pass<ngraph::pass::ConstantFolding>();
        manager.register_pass<pass::ConvertQuantize>();
        manager.register_pass<pass::ConvertBiasToI32>();
        manager.register_pass<ngraph::pass::ConstantFolding>();
        manager.register_pass<pass::MovePerChenelQuantizationInfoToWeights>();
        manager.register_pass<ngraph::pass::ConstantFolding>();
        manager.register_pass<PropogateQuantizationInfo>();
        manager.run_passes(f);
    }

    Dump(f, "final");

    return false;
}