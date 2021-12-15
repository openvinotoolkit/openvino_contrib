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
#include "transformations/op_conversions/convert_gather_downgrade.hpp"
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
#include "replace_power_by_mul.hpp"

#include <ngraph/pass/manager.hpp>
#include <ngraph/pass/constant_folding.hpp>

#include <transformations/low_precision/disable_convert_constant_folding_on_const_path.hpp>
#include <low_precision/common/operation_per_tensor_quantization_restriction.hpp>
#include <low_precision/common/operation_precision_restriction.hpp>
#include <low_precision/convolution.hpp>
#include <low_precision/fake_quantize.hpp>
#include <low_precision/fold_convert.hpp>
#include <low_precision/fuse_convert.hpp>
#include <low_precision/fuse_multiply_to_fake_quantize.hpp>
#include <low_precision/fuse_subtract_to_fake_quantize.hpp>
#include <low_precision/group_convolution.hpp>
#include <low_precision/low_precision.hpp>
#include <low_precision/multiply_to_group_convolution.hpp>
#include <low_precision/multiply.hpp>
#include <low_precision/network_helper.hpp>

#include "transformations/serialize.hpp"


#include "arm_optimizations.hpp"

#if(__ANDROID__)
#define REGISTER_PASS manager.register_pass<ov::pass::GraphRewrite>()->add_matcher
#else
#define REGISTER_PASS manager.register_pass
#endif


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
                const auto& quantizationInfo = itInfo->second.as<arm_compute::QuantizationInfo>();
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
    ov::pass::Manager manager;

    Dump(f, "initial");

    auto quantized = _lpt && ngraph::pass::low_precision::LowPrecision::isFunctionQuantized(f);

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
    REGISTER_PASS<pass::ReplacePowerByMul>();
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
    REGISTER_PASS<pass::ConvertConv1D>();
    REGISTER_PASS<pass::ConvertGroupConv1D>();
    REGISTER_PASS<pass::ConvertGroupConvolution>();
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


    using namespace ngraph::pass::low_precision;
    if (quantized) {
        Dump(f, "before_common");
        auto supportedPrecisions = std::vector<OperationPrecisionRestriction>({
            OperationPrecisionRestriction::create<ngraph::opset1::Convolution>({
                {0, {ngraph::element::u8, ngraph::element::i8}},
                {1, {ngraph::element::u8, ngraph::element::i8}},
            }),
            OperationPrecisionRestriction::create<ngraph::opset1::ConvolutionBackpropData>({
                {0, {ngraph::element::u8, ngraph::element::i8}},
                {1, {ngraph::element::u8, ngraph::element::i8}}
            }),
            OperationPrecisionRestriction::create<ngraph::opset1::GroupConvolution>({
                {0, {ngraph::element::u8, ngraph::element::i8}},
                {1, {ngraph::element::u8, ngraph::element::i8}}
            })
        });

        auto perTensorQuantization = std::vector<OperationPerTensorQuantizationRestriction>({
            OperationPerTensorQuantizationRestriction::create<ngraph::opset1::Convolution>({0}),
            OperationPerTensorQuantizationRestriction::create<ngraph::opset1::ConvolutionBackpropData>({0}),
            OperationPerTensorQuantizationRestriction::create<ngraph::opset1::GroupConvolution>({0})
        });

        ov::pass::Manager lptManager;
        lptManager.register_pass<ngraph::pass::low_precision::LowPrecision>(supportedPrecisions, perTensorQuantization);
        auto pass_config = lptManager.get_pass_config();
        pass_config->disable<ngraph::pass::low_precision::MultiplyToGroupConvolutionTransformation>();
        pass_config->disable<ngraph::pass::low_precision::FoldConvertTransformation>();
        pass_config->disable<ngraph::pass::low_precision::FuseConvertTransformation>();
        pass_config->disable<ngraph::pass::low_precision::FuseSubtractToFakeQuantizeTransformation>();
        pass_config->disable<ngraph::pass::low_precision::FuseMultiplyToFakeQuantizeTransformation>();
        lptManager.run_passes(f);
    }


    {
        Dump(f, "before_arm_specific_transformations");
        ov::pass::Manager manager;
        manager.register_pass<ngraph::pass::LogSoftmaxDecomposition>();
        REGISTER_PASS<pass::ConvertGRN>();
        REGISTER_PASS<pass::NormalizeL2Fusion>();
        REGISTER_PASS<pass::DecomposeNormalizeL2Add>();
        REGISTER_PASS<pass::ConvertNormalizeL2ToArm>();
        REGISTER_PASS<pass::ConvertReduceMultiAxis>();
        manager.register_pass<ngraph::pass::ReduceL1Decomposition>();
        manager.register_pass<ngraph::pass::ReduceL2Decomposition>();
        manager.register_pass<ngraph::pass::ConvertReduceMeanToPooling>();
        manager.register_pass<ngraph::pass::ConvertReduceMaxToPooling>();
        manager.register_pass<ngraph::pass::ConvertReduceSumToPooling>();
        manager.register_pass<ngraph::pass::ConvertMod>();
        manager.register_pass<ngraph::pass::ConstantFolding>();
        REGISTER_PASS<pass::DecomposeSwish>();
        REGISTER_PASS<pass::DecomposeMish>();
        REGISTER_PASS<pass::BroadcastPRelu>();
        REGISTER_PASS<pass::ConvertLogical>();
        REGISTER_PASS<pass::ConvertComparison>();
        REGISTER_PASS<pass::ConvertTranspose>();
        manager.register_pass<ngraph::pass::ConstantFolding>();
        REGISTER_PASS<pass::ConvertRound>();
        REGISTER_PASS<pass::ConvertSign>();
        REGISTER_PASS<pass::ConvertCeiling>();
        REGISTER_PASS<pass::DecomposeVariadicSplit>();
        REGISTER_PASS<pass::ConvertStridedSliceToArm>();
        REGISTER_PASS<pass::ConvertStridedSlice>();
        REGISTER_PASS<pass::ConvertBatchNormInferenceV0toV5>();
        REGISTER_PASS<pass::ConvertBatchNormInference>();
        REGISTER_PASS<pass::ConvertShuffleChannels>();
        REGISTER_PASS<pass::ConvertInterpolate>();
        REGISTER_PASS<pass::ConvertMVN>();
        REGISTER_PASS<pass::ConvertReorgYolo>();
        REGISTER_PASS<pass::ConvertMaxPool1D>();
        REGISTER_PASS<pass::ConvertAvgPool1D>();
        REGISTER_PASS<pass::BroadcastSelect>();
        REGISTER_PASS<pass::ConvertGather>();
        manager.register_pass<ngraph::pass::ConvertGather8ToGather7>();
        REGISTER_PASS<pass::ConvertDFT>();
        REGISTER_PASS<pass::ConvertIDFT>();
        manager.register_pass<ngraph::pass::ConstantFolding>();
        REGISTER_PASS<pass::ConvBiasFusion>();
        manager.register_pass<ngraph::pass::ConstantFolding>();
        REGISTER_PASS<pass::ConvertMatMulToFC>();
        manager.register_pass<ngraph::pass::ConstantFolding>();
        REGISTER_PASS<pass::ConvertArmConvert>();
        REGISTER_PASS<pass::ConvertArmConvertLike>();
        manager.register_pass<ngraph::pass::ConstantFolding>();
        manager.register_pass<ngraph::pass::ConvertDivide>();
        manager.register_pass<ngraph::pass::ConvertBroadcast3>();
        manager.register_pass<ngraph::pass::ConvertBroadcastToTiles>();
        REGISTER_PASS<pass::ConvertEltwise>();
        manager.register_pass<ngraph::pass::ConstantFolding>();
        REGISTER_PASS<pass::ConvertTile>();
        REGISTER_PASS<pass::ConvertSplit>();
        REGISTER_PASS<pass::ConvertConcat>();
        manager.register_pass<pass::FinalizeTrailingNodes>();
        manager.register_pass<pass::StoreResultName>();
        manager.register_pass<ngraph::pass::ConstantFolding>();
        manager.register_pass<ngraph::pass::ConvertPrecision>(ngraph::element::boolean, ngraph::element::u8);
        manager.register_pass<ngraph::pass::ConvertPrecision>(ngraph::element::i64, ngraph::element::i32);
        manager.register_pass<ngraph::pass::ConvertPrecision>(ngraph::element::u64, ngraph::element::i32);
        REGISTER_PASS<pass::AlignNodePrecision>();
        manager.register_pass<ngraph::pass::ConstantFolding>();
        manager.run_passes(f);
    }

    if (quantized) {
        Dump(f, "before_arm");
        ov::pass::Manager manager;
        REGISTER_PASS<pass::ConvolutionQuantizeFusion>();
        REGISTER_PASS<pass::MeanQuantizeFusion>();
        REGISTER_PASS<pass::DequantizeInputFusion>();
        REGISTER_PASS<pass::AddDequantizeOnInputs>();
        manager.register_pass<ngraph::pass::ConstantFolding>();
        REGISTER_PASS<pass::ConvertQuantize>();
        REGISTER_PASS<pass::ConvertBiasToI32>();
        manager.register_pass<ngraph::pass::ConstantFolding>();
        manager.run_passes(f);
    }

    Dump(f, "final");

    return false;
}