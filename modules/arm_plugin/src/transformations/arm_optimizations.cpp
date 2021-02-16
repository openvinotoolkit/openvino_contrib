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

#include "conv_bias_activ_fusion.hpp"
#include "convert_eltwise.hpp"
#include "convert_sign.hpp"
#include "convert_round.hpp"
#include "convert_comparison.hpp"
#include "convert_logical.hpp"
#include "convert_strided_slice.hpp"
#include "convert_group_conv.hpp"
#include "convert_conv1d_to_conv2d.hpp"
#include "convert_grn_to_normalizel2.hpp"
#include "convert_mat_mul.hpp"
#include "convert_batchnorm_v0_to_v5.hpp"
#include "convert_batch_norm.hpp"
#include "convert_ceiling.hpp"
#include "decompose_swish.hpp"
#include "convert_shuffle_channels.hpp"
#include "convert_tile_to_concats.hpp"
#include "convert_prelu.hpp"
#include "convert_reduce_multi_axis.hpp"
#include "convert_interpolate_v0_to_v4.hpp"
#include "normalizel2_max_fusion.hpp"
#include "decompose_normalizel2_add.hpp"
#include "decompose_mish.hpp"
#include "finalize_trailing_nodes.hpp"
#include "transformations/convert_reorg.hpp"
#include "transformations/convert_prior_box_to_const.hpp"
#include "transformations/convert_prior_box_clustered_to_const.hpp"

#include <ngraph/pass/manager.hpp>
#include <ngraph/pass/constant_folding.hpp>

#include "arm_optimizations.hpp"


bool ArmPlugin::pass::ArmOptimizations::run_on_function(std::shared_ptr<ngraph::Function> f) {
    ngraph::pass::Manager manager;

    // This pass must be called first in pipeline
    manager.register_pass<ngraph::pass::InitNodeInfo>();
    manager.register_pass<ngraph::pass::RemoveFilteringBoxesBySize>(); // Resolves dynamism (replaces NonZero), CF needed
    manager.register_pass<ngraph::pass::ConstantFolding>();
    manager.register_pass<ngraph::pass::NopElimination>(); // may introduce fake dynamism
    manager.register_pass<ngraph::pass::AlgebraicSimplification>(); // may introduce fake dynamism
    manager.register_pass<ngraph::pass::ConstantFolding>();
    manager.register_pass<ngraph::pass::SoftPlusFusion>();
    manager.register_pass<ngraph::pass::HSwishFusion>();

    manager.register_pass<ngraph::pass::LogSoftmaxDecomposition>();
    manager.register_pass<pass::NormalizeL2Fusion>();
    manager.register_pass<pass::DecomposeNormalizeL2Add>();
    manager.register_pass<pass::ConvertReduceMultiAxis>();
    manager.register_pass<ngraph::pass::ReduceL1Decomposition>();
    manager.register_pass<ngraph::pass::ReduceL2Decomposition>();
    manager.register_pass<ngraph::pass::ConvertReduceMeanToPooling>();
    manager.register_pass<ngraph::pass::ConvertReduceMaxToPooling>();
    manager.register_pass<ngraph::pass::ConvertReduceSumToPooling>();
    manager.register_pass<ngraph::pass::ConvertMod>();
    manager.register_pass<ngraph::pass::ConstantFolding>();

    // LinOpSequenceFusion must be executed after all decompositions
    manager.register_pass<ngraph::pass::LinOpSequenceFusion>();

    manager.register_pass<ngraph::pass::ConvolutionMultiplyFusion>();
    manager.register_pass<ngraph::pass::GroupConvolutionMultiplyFusion>();
    manager.register_pass<ngraph::pass::ConvolutionBackpropDataMultiplyFusion>();
    manager.register_pass<ngraph::pass::GroupConvolutionBackpropDataMultiplyFusion>();
    manager.register_pass<ngraph::pass::RNNCellDecomposition>();
    manager.register_pass<ngraph::pass::LSTMCellDecomposition>();
    manager.register_pass<ngraph::pass::GRUCellDecomposition>();
    manager.register_pass<ngraph::pass::ConstantFolding>();

    manager.register_pass<pass::ConvertGRN>();
    manager.register_pass<pass::DecomposeSwish>();
    manager.register_pass<pass::DecomposeMish>();
    manager.register_pass<pass::ConvertConv1D>();
    manager.register_pass<pass::ConvertGroupConv1D>();
    manager.register_pass<pass::ConvertGroupConvolution>();
    manager.register_pass<pass::ConvBiasActivationFusion>();
    manager.register_pass<pass::ConvertMatMulToFC>();
    manager.register_pass<pass::ConvertEltwise>();
    manager.register_pass<pass::BroadcastPRelu>();
    manager.register_pass<pass::ConvertLogical>();
    manager.register_pass<pass::ConvertComparison>();
    manager.register_pass<ngraph::pass::ConstantFolding>();

    manager.register_pass<pass::ConvertRound>();
    manager.register_pass<pass::ConvertSign>();
    manager.register_pass<pass::ConvertCeiling>();
    manager.register_pass<pass::DecomposeVariadicSplit>();
    manager.register_pass<pass::ConvertStridedSlice>();
    manager.register_pass<pass::ConvertBatchNormInferenceV0toV5>();
    manager.register_pass<pass::ConvertBatchNormInference>();
    manager.register_pass<pass::ConvertShuffleChannels>();
    manager.register_pass<pass::ConvertInterpolateV0toV4>();
    manager.register_pass<pass::ConvertReorgYolo>();

    manager.register_pass<pass::ConvertPriorBox>();
    manager.register_pass<pass::ConvertPriorBoxClustered>();
    manager.register_pass<ngraph::pass::ConstantFolding>();

    manager.register_pass<ngraph::pass::ConvertDivide>();
    manager.register_pass<ngraph::pass::ConvertBroadcast3>();
    manager.register_pass<ngraph::pass::ConvertBroadcastToTiles>();
    manager.register_pass<pass::ConvertTile>();
    manager.register_pass<pass::FinalizeTrailingNodes>();
    manager.register_pass<ngraph::pass::ConstantFolding>();
    manager.register_pass<ngraph::pass::ConvertPrecision>(ngraph::element::boolean, ngraph::element::u8);
    manager.register_pass<ngraph::pass::ConvertPrecision>(ngraph::element::i64, ngraph::element::i32);
    manager.register_pass<ngraph::pass::ConvertPrecision>(ngraph::element::u64, ngraph::element::i32);
#ifndef __ARM_FEATURE_FP16_VECTOR_ARITHMETIC
    manager.register_pass<ngraph::pass::ConvertPrecision>(ngraph::element::f16, ngraph::element::f32);
#endif
    manager.run_passes(f);
    return false;
}