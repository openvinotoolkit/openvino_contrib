// Copyright (C) 2020-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0


#include "transformations/common_optimizations/nop_elimination.hpp"
#include "transformations/convert_precision.hpp"
#include "transformations/init_node_info.hpp"
#include "transformations/decompose_variadic_split.hpp"
#include "transformations/common_optimizations/softplus_fusion.hpp"
#include "transformations/common_optimizations/reshape_prelu.hpp"
#include "transformations/op_conversions/convert_reduce_to_pooling.hpp"
#include "transformations/op_conversions/convert_broadcast3.hpp"
#include "transformations/op_conversions/convert_broadcast_to_tiles.hpp"
#include "transformations/op_conversions/convert_gather_downgrade.hpp"
#include "transformations/op_conversions/convert_softmax_downgrade.hpp"
#include "transformations/op_conversions/rnn_cell_decomposition.hpp"
#include "transformations/op_conversions/lstm_cell_decomposition.hpp"
#include "transformations/op_conversions/gru_cell_decomposition.hpp"
#include "transformations/op_conversions/log_softmax_decomposition.hpp"
#include "transformations/common_optimizations/remove_filtering_boxes_by_size.hpp"
#include "transformations/common_optimizations/hswish_fusion.hpp"
#include "transformations/op_conversions/convert_mvn1_to_mvn6.hpp"
#include "transformations/op_conversions/convert_gelu.hpp"
#include "transformations/op_conversions/convert_ti_to_sequences.hpp"
#include "transformations/common_optimizations/convert_quantize_dequantize.hpp"
#include "transformations/op_conversions/convert_subtract.hpp"
#include "transformations/op_conversions/convert_maxpool_downgrade.hpp"
#include "transformations/op_conversions/convert_previous_nms_to_nms_9.hpp"
#include "transformations/common_optimizations/common_optimizations.hpp"
#include "transformations/common_optimizations/convert_compression_only_to_legacy.hpp"
#include "transformations/op_conversions/hswish_decomposition.hpp"
#include "transformations/op_conversions/convert_minimum_to_power_and_max.hpp"
#include "transformations/op_conversions/convert_divide.hpp"
#include "transformations/op_conversions/convert_depth_to_space.hpp"
#include "transformations/op_conversions/convert_space_to_depth.hpp"
#include "transformations/op_conversions/batch_norm_decomposition.hpp"
#include "transformations/op_conversions/mvn6_decomposition.hpp"
#include <transformations/op_conversions/normalize_l2_decomposition.hpp>
#include <transformations/op_conversions/softmax_decomposition.hpp>

#include "conv_bias_fusion.hpp"
#include "convert_eltwise.hpp"
#include "convert_sign.hpp"
#include "convert_round.hpp"
#include "convert_comparison.hpp"
#include "convert_logical.hpp"
#include "convert_strided_slice.hpp"
#include "convert_strided_slice_arm.hpp"
#include "convert_slice_arm.hpp"
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
#include "convert_shuffle_channels.hpp"
#include "convert_tile_to_concats.hpp"
#include "convert_transpose_arm.hpp"
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
#include "convert_maxpool_v8.hpp"
#include "convert_inputs_precision.hpp"
#include "finalize_trailing_nodes.hpp"
#include "transformations/convert_reorg.hpp"
#include "quantize_fusion.hpp"
#include "store_result_name.hpp"
#include "replace_power_by_mul.hpp"
#include "convert_precision_fp16_to_fp32.hpp"

#include <ngraph/pass/manager.hpp>
#include <ngraph/pass/constant_folding.hpp>

#include <transformations/low_precision/mark_dequantization_subgraph.hpp>
#include <low_precision/common/quantization_granularity_restriction.hpp>
#include <low_precision/common/precisions_restriction.hpp>
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

#include "openvino/pass/serialize.hpp"


#include "arm_optimizations.hpp"

NGRAPH_RTTI_DEFINITION(ArmPlugin::pass::ArmOptimizations, "ArmOptimizations", 0);
void ArmPlugin::pass::ArmOptimizations::Dump(const std::shared_ptr<ov::Model>& m, const std::string& postfix) {
    if (_dump) {
        ngraph::pass::VisualizeTree{m->get_friendly_name() + "_" + postfix +
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
        }}.run_on_model(m);
    }
}

static bool fuse_type_to_convert(const std::shared_ptr<ngraph::Node>& node, ov::element::Type to, size_t idx) {
    if (auto convert = ov::as_type_ptr<ArmPlugin::opset::Convert>(node)) {
        // For Convert node, converting precision from floating point to boolean will lead to mathematical
        // error, because here the output precision boolean is replaced by u8. E.g. floating point value 0.01
        // is converted to be 1 for boolean, but 0 for u8. Thus an Abs and Ceil node should be added before the
        // Convert node for this scenario.
        if (convert->input(0).get_element_type().is_real() &&
            convert->get_convert_element_type() == ngraph::element::boolean && to.is_integral_number()) {
            auto abs = std::make_shared<ArmPlugin::opset::Abs>(convert->input_value(0).get_node_shared_ptr());
            auto ceil = std::make_shared<ArmPlugin::opset::Ceiling>(abs);
            auto new_convert = std::make_shared<ArmPlugin::opset::Convert>(ceil, to);
            new_convert->set_friendly_name(convert->get_friendly_name());
            ov::copy_runtime_info(convert, {abs, ceil, new_convert});
            ov::replace_node(convert, new_convert);
            return true;
        } else {
            convert->set_convert_element_type(to);
            return true;
        }
    }
    return false;
}

bool ArmPlugin::pass::ArmOptimizations::run_on_model(const std::shared_ptr<ov::Model> &m) {
    auto quantized = _lpt && ngraph::pass::low_precision::LowPrecision::isFunctionQuantized(m);
    {
        ov::pass::Manager manager;

        Dump(m, "initial");

        if (quantized) {
            manager.register_pass<ov::pass::GraphRewrite>()->add_matcher<ov::pass::MarkDequantizationSubgraph>(
                std::vector<ngraph::element::Type>{ ngraph::element::i8, ngraph::element::u8 });
        }

        // This pass must be called first in pipeline
        manager.register_pass<ov::pass::InitNodeInfo>();
        manager.register_pass<pass::StoreResultName>();

        // Resolves dynamism (replaces NonZero), CF needed
        manager.register_pass<ov::pass::GraphRewrite>()->add_matcher<ov::pass::RemoveFilteringBoxesBySize>();
        manager.register_pass<ngraph::pass::ConstantFolding>();
        // may introduce fake dynamism
        manager.register_pass<ov::pass::GraphRewrite>()->add_matcher<ov::pass::NopElimination>();
        manager.register_pass<ov::pass::GraphRewrite>()->add_matcher<pass::ReplacePowerByMul>();
        manager.register_pass<ngraph::pass::ConstantFolding>();
        manager.register_pass<ov::pass::GraphRewrite>()->add_matcher<ov::pass::SoftPlusFusion>();
        manager.register_pass<ov::pass::GraphRewrite>()->add_matcher<ov::pass::HSwishFusion>();

        // LinOpSequenceFusion must be executed after all decompositions
        manager.register_pass<ngraph::pass::ConstantFolding>();
        manager.register_pass<ov::pass::GraphRewrite>()->add_matcher<ov::pass::ConvertTensorIteratorToGRUSequence>();
        manager.register_pass<ov::pass::GraphRewrite>()->add_matcher<ov::pass::ConvertTensorIteratorToLSTMSequence>();
        manager.register_pass<ov::pass::GraphRewrite>()->add_matcher<ov::pass::ConvertTensorIteratorToRNNSequence>();
        manager.register_pass<ngraph::pass::ConstantFolding>();
        manager.register_pass<ov::pass::GraphRewrite>()->add_matcher<ov::pass::RNNCellDecomposition>();
        manager.register_pass<ov::pass::GraphRewrite>()->add_matcher<ov::pass::LSTMCellDecomposition>();
        manager.register_pass<ov::pass::GraphRewrite>()->add_matcher<ov::pass::GRUCellDecomposition>();

        // Run common optimizations
        manager.register_pass<ov::pass::CommonOptimizations>();
        manager.register_pass<ov::pass::ReshapePRelu>();
        manager.get_pass_config()->disable<ov::pass::ConvertCompressedOnlyToLegacy>();
        manager.get_pass_config()->disable<ov::pass::HSwishDecomposition>();
        manager.get_pass_config()->disable<ov::pass::LogSoftmaxDecomposition>();
#ifdef __aarch64__
        manager.get_pass_config()->disable<ov::pass::ConvertGELU>();
#endif /* __aarch64__ */
        manager.get_pass_config()->disable<ov::pass::ConvertBroadcastToTiles>();
        manager.get_pass_config()->disable<ov::pass::ConvertMinimum>();
        manager.get_pass_config()->disable<ov::pass::ConvertSubtract>();
        manager.get_pass_config()->disable<ov::pass::ConvertDivide>();
        manager.get_pass_config()->disable<ov::pass::ConvertDepthToSpace>();
        manager.get_pass_config()->disable<ov::pass::ConvertSpaceToDepth>();
        manager.get_pass_config()->disable<ov::pass::BatchNormDecomposition>();
        // MVN6Decomposition doesn't work with ARM native ReduceMean operation
        manager.get_pass_config()->disable<ov::pass::MVN6Decomposition>();
        manager.get_pass_config()->disable<ov::pass::NormalizeL2Decomposition>();
        manager.get_pass_config()->disable<ov::pass::SoftmaxDecomposition>();

        manager.register_pass<ngraph::pass::ConstantFolding>();
        manager.register_pass<ov::pass::GraphRewrite>()->add_matcher<pass::ConvertConv1D>();
        manager.register_pass<ov::pass::GraphRewrite>()->add_matcher<pass::ConvertGroupConv1D>();
        manager.register_pass<ov::pass::GraphRewrite>()->add_matcher<pass::ConvertGroupConvolution>();
        manager.register_pass<ngraph::pass::ConstantFolding>();
        manager.register_pass<ov::pass::GraphRewrite>()->add_matcher<ov::pass::ConvertMVN1ToMVN6>();

        auto pass_config = manager.get_pass_config();

        using ConstNodePtr = const std::shared_ptr<const ngraph::Node>;

        if (quantized) {
            pass_config->set_callback<ov::pass::ConvertQuantizeDequantize>([](ConstNodePtr& node) -> bool {
                return ngraph::pass::low_precision::NetworkHelper::areQuantizeAndDequantizeSupportedForMultiply(node);
            });

            pass_config->set_callback<ov::pass::ConvertSubtract>([](ConstNodePtr& node) -> bool {
                return ngraph::pass::low_precision::NetworkHelper::areQuantizeAndDequantizeSupportedForSubtract(node);
            });
        }

        manager.run_passes(m);
    }

    using namespace ngraph::pass::low_precision;
    if (quantized) {
        Dump(m, "before_common");
        auto supportedPrecisions = std::vector<PrecisionsRestriction>({
            PrecisionsRestriction::create<ngraph::opset1::Convolution>({
                {{0}, {ngraph::element::u8, ngraph::element::i8}},
                {{1}, {ngraph::element::u8, ngraph::element::i8}},
            }),
            PrecisionsRestriction::create<ngraph::opset1::ConvolutionBackpropData>({
                {{0}, {ngraph::element::u8, ngraph::element::i8}},
                {{1}, {ngraph::element::u8, ngraph::element::i8}}
            }),
            PrecisionsRestriction::create<ngraph::opset1::GroupConvolution>({
                {{0}, {ngraph::element::u8, ngraph::element::i8}},
                {{1}, {ngraph::element::u8, ngraph::element::i8}}
            })
        });

        auto perTensorQuantization = std::vector<QuantizationGranularityRestriction>({
            QuantizationGranularityRestriction::create<ngraph::opset1::Convolution>({0}),
            QuantizationGranularityRestriction::create<ngraph::opset1::ConvolutionBackpropData>({0}),
            QuantizationGranularityRestriction::create<ngraph::opset1::GroupConvolution>({0})
        });

        ov::pass::Manager lptManager;
        lptManager.register_pass<ngraph::pass::low_precision::LowPrecision>(supportedPrecisions, perTensorQuantization);
        auto pass_config = lptManager.get_pass_config();
        pass_config->disable<ngraph::pass::low_precision::MultiplyToGroupConvolutionTransformation>();
        pass_config->disable<ngraph::pass::low_precision::FoldConvertTransformation>();
        pass_config->disable<ngraph::pass::low_precision::FuseConvertTransformation>();
        pass_config->disable<ngraph::pass::low_precision::FuseSubtractToFakeQuantizeTransformation>();
        pass_config->disable<ngraph::pass::low_precision::FuseMultiplyToFakeQuantizeTransformation>();
        lptManager.run_passes(m);
    }

    auto get_convert_precisions = []() {
        precisions_array array = {
            {ngraph::element::i64,     ngraph::element::i32},
            {ngraph::element::u64,     ngraph::element::i32},
            {ngraph::element::f64,     ngraph::element::f32},
            {ngraph::element::boolean, ngraph::element::u8},
            {ngraph::element::i4,      ngraph::element::i8},
            {ngraph::element::u4,      ngraph::element::u8}
        };
        #ifndef __ARM_FEATURE_FP16_VECTOR_ARITHMETIC
            array.push_back({ngraph::element::f16, ngraph::element::f32});
        #endif
        return array;
    };
    static const auto precisions = get_convert_precisions();
    type_to_fuse_map type_to_fuse = {{ArmPlugin::opset::Convert::get_type_info_static(), fuse_type_to_convert}};

    {
        Dump(m, "before_arm_specific_transformations");
        ov::pass::Manager manager;
        manager.register_pass<ov::pass::GraphRewrite>()->add_matcher<pass::ConvertGRN>();
        manager.register_pass<ov::pass::GraphRewrite>()->add_matcher<pass::NormalizeL2Fusion>();
        manager.register_pass<ov::pass::GraphRewrite>()->add_matcher<pass::DecomposeNormalizeL2Add>();
        manager.register_pass<ov::pass::GraphRewrite>()->add_matcher<pass::ConvertNormalizeL2ToArm>();
        manager.register_pass<ov::pass::GraphRewrite>()->add_matcher<pass::ConvertReduceMultiAxis>();
        manager.register_pass<ov::pass::GraphRewrite>()->add_matcher<ov::pass::ConvertReduceMeanToPooling>();
        manager.register_pass<ov::pass::GraphRewrite>()->add_matcher<ov::pass::ConvertReduceMaxToPooling>();
        manager.register_pass<ov::pass::GraphRewrite>()->add_matcher<ov::pass::ConvertReduceSumToPooling>();
        manager.register_pass<ngraph::pass::ConstantFolding>();
        manager.register_pass<ov::pass::GraphRewrite>()->add_matcher<pass::DecomposeMish>();
        manager.register_pass<ov::pass::GraphRewrite>()->add_matcher<pass::ConvertLogical>();
        manager.register_pass<ov::pass::GraphRewrite>()->add_matcher<pass::ConvertComparison>();
        manager.register_pass<ov::pass::GraphRewrite>()->add_matcher<pass::ConvertTranspose>();
        manager.register_pass<ngraph::pass::ConstantFolding>();
        manager.register_pass<ov::pass::GraphRewrite>()->add_matcher<pass::ConvertRound>();
        manager.register_pass<ov::pass::GraphRewrite>()->add_matcher<pass::ConvertSign>();
        manager.register_pass<ov::pass::GraphRewrite>()->add_matcher<pass::ConvertCeiling>();
        manager.register_pass<ov::pass::GraphRewrite>()->add_matcher<pass::DecomposeVariadicSplit>();
        manager.register_pass<ov::pass::GraphRewrite>()->add_matcher<pass::ConvertStridedSliceToArm>();
        manager.register_pass<ov::pass::GraphRewrite>()->add_matcher<pass::ConvertStridedSlice>();
        manager.register_pass<ov::pass::GraphRewrite>()->add_matcher<pass::ConvertSliceToArm>();
        manager.register_pass<ov::pass::GraphRewrite>()->add_matcher<pass::ConvertBatchNormInferenceV0toV5>();
        manager.register_pass<ov::pass::GraphRewrite>()->add_matcher<pass::ConvertBatchNormInference>();
        manager.register_pass<ov::pass::GraphRewrite>()->add_matcher<pass::ConvertShuffleChannels>();
        manager.register_pass<ov::pass::GraphRewrite>()->add_matcher<pass::ConvertInterpolate>();
        manager.register_pass<ov::pass::GraphRewrite>()->add_matcher<pass::ConvertMVN>();
        manager.register_pass<ov::pass::GraphRewrite>()->add_matcher<ov::pass::ConvertNMS5ToNMS9>();
        manager.register_pass<ov::pass::GraphRewrite>()->add_matcher<pass::ConvertReorgYolo>();
        manager.register_pass<ov::pass::GraphRewrite>()->add_matcher<ov::pass::ConvertMaxPool8ToMaxPool1>();
        manager.register_pass<ov::pass::GraphRewrite>()->add_matcher<pass::ConvertMaxPool1D>();
        manager.register_pass<ov::pass::GraphRewrite>()->add_matcher<pass::ConvertAvgPool1D>();
        manager.register_pass<ov::pass::GraphRewrite>()->add_matcher<pass::ConvertMaxPoolV8>();
        manager.register_pass<ov::pass::GraphRewrite>()->add_matcher<pass::BroadcastSelect>();
        manager.register_pass<ov::pass::GraphRewrite>()->add_matcher<pass::ConvertGather>();
        manager.register_pass<ov::pass::GraphRewrite>()->add_matcher<ov::pass::ConvertGather8ToGather7>();
        manager.register_pass<ov::pass::GraphRewrite>()->add_matcher<ov::pass::ConvertSoftMax8ToSoftMax1>();
        manager.register_pass<ov::pass::GraphRewrite>()->add_matcher<pass::ConvertDFT>();
        manager.register_pass<ov::pass::GraphRewrite>()->add_matcher<pass::ConvertIDFT>();
        manager.register_pass<ngraph::pass::ConstantFolding>();
        manager.register_pass<ov::pass::GraphRewrite>()->add_matcher<pass::ConvBiasFusion>();
        manager.register_pass<ngraph::pass::ConstantFolding>();
        manager.register_pass<ov::pass::GraphRewrite>()->add_matcher<pass::ConvertMatMulToFC>();
        manager.register_pass<ngraph::pass::ConstantFolding>();
        manager.register_pass<ov::pass::GraphRewrite>()->add_matcher<ov::pass::ConvertBroadcast3>();
        manager.register_pass<ov::pass::GraphRewrite>()->add_matcher<ov::pass::ConvertBroadcastToTiles>();
        manager.register_pass<ov::pass::GraphRewrite>()->add_matcher<pass::ConvertEltwise>();
        manager.register_pass<ngraph::pass::ConstantFolding>();
        manager.register_pass<ov::pass::GraphRewrite>()->add_matcher<pass::ConvertTile>();
        manager.register_pass<ov::pass::GraphRewrite>()->add_matcher<pass::ConvertSplit>();
        manager.register_pass<ov::pass::GraphRewrite>()->add_matcher<pass::ConvertConcat>();
        manager.register_pass<pass::FinalizeTrailingNodes>();
        manager.register_pass<pass::StoreResultName>();
        manager.register_pass<ngraph::pass::ConstantFolding>();
        manager.register_pass<ov::pass::ConvertPrecision>(precisions, type_to_fuse);
        manager.register_pass<ov::pass::GraphRewrite>()->add_matcher<pass::AlignNodePrecision>();
        manager.register_pass<pass::ConvertPrecisionFP16ToFP32>();
        manager.register_pass<ov::pass::GraphRewrite>()->add_matcher<pass::ConvertArmConvert>();
        manager.register_pass<ov::pass::GraphRewrite>()->add_matcher<pass::ConvertArmConvertLike>();
        manager.register_pass<ngraph::pass::ConstantFolding>();
        manager.run_passes(m);
    }

    if (quantized) {
        Dump(m, "before_arm");
        ov::pass::Manager manager;
        {
            auto pass = manager.register_pass<ov::pass::GraphRewrite>();
            pass->add_matcher<pass::ConvolutionQuantizeFusion>();
            pass->add_matcher<pass::MeanQuantizeFusion>();
        }
        {
            auto pass = manager.register_pass<ov::pass::GraphRewrite>();
            pass->add_matcher<pass::DequantizeInputFusion>();
        }
        {
            auto pass = manager.register_pass<ov::pass::GraphRewrite>();
            pass->add_matcher<pass::AddDequantizeOnInputs>();
            pass->add_matcher<pass::ConvertBiasToI32>();
            pass->add_matcher<pass::ConvertQuantize>();
        }
        manager.register_pass<ngraph::pass::ConstantFolding>();
        manager.run_passes(m);
    }

    Dump(m, "final");

    return false;
}
