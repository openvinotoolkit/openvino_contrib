// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "transforms/pipeline.hpp"

#include "runtime/gfx_logger.hpp"
#include "transforms/gfx_layout_cleanup.hpp"

#include <optional>
#include <vector>

#include "openvino/pass/manager.hpp"
#include "openvino/op/constant.hpp"
#include "openvino/op/convert.hpp"
#include "openvino/op/matmul.hpp"
#include "openvino/op/multiply.hpp"
#include "openvino/op/reshape.hpp"
#include "transformations/common_optimizations/common_optimizations.hpp"
#include "transformations/common_optimizations/rms_fusion.hpp"
#include "transformations/control_flow/unroll_if.hpp"
#include "transformations/fp16_compression/convert_compression_only_to_legacy.hpp"
#include "transformations/fp16_compression/mark_decompression_convert_constant_folding.hpp"
#include "transformations/op_conversions/convert_avgpool_downgrade.hpp"
#include "transformations/op_conversions/convert_maxpool_downgrade.hpp"
#include "transformations/op_conversions/convert_mod.hpp"
#include "transformations/op_conversions/convert_reduce_to_pooling.hpp"
#include "transformations/op_conversions/scaled_dot_product_attention_decomposition.hpp"
#include "transformations/rt_info/decompression.hpp"
#include "transformations/rt_info/disable_constant_folding.hpp"

namespace ov {
namespace gfx_plugin {
namespace transforms {
namespace {

struct CompressedMatMulMatch {
    std::shared_ptr<ov::op::v0::MatMul> matmul;
    std::shared_ptr<ov::op::v0::Constant> weights;
    std::shared_ptr<ov::op::v0::Constant> scale;
    ov::Output<ov::Node> data;
    std::vector<std::shared_ptr<ov::Node>> decompression_nodes;
    ov::element::Type weight_input_type = ov::element::dynamic;
    size_t n = 0;
    size_t groups = 0;
    size_t group_size = 0;
    size_t k = 0;
};

std::shared_ptr<ov::op::v0::Constant> as_constant(const ov::Output<ov::Node>& value) {
    return ov::as_type_ptr<ov::op::v0::Constant>(value.get_node_shared_ptr());
}

std::optional<CompressedMatMulMatch> match_compressed_matmul(const std::shared_ptr<ov::Node>& node) {
    auto matmul = ov::as_type_ptr<ov::op::v0::MatMul>(node);
    if (!matmul || !matmul->get_transpose_b() || matmul->get_input_size() != 2 ||
        matmul->get_output_size() != 1) {
        return std::nullopt;
    }
    if (!matmul->get_input_partial_shape(1).is_static()) {
        return std::nullopt;
    }
    const auto b_shape = matmul->get_input_shape(1);
    if (b_shape.size() != 2) {
        return std::nullopt;
    }

    CompressedMatMulMatch match;
    auto source = matmul->input_value(1).get_node_shared_ptr();
    const auto weight_input_type = matmul->get_input_element_type(1);
    if (auto convert = ov::as_type_ptr<ov::op::v0::Convert>(source)) {
        match.decompression_nodes.push_back(convert);
        source = convert->input_value(0).get_node_shared_ptr();
    }
    auto reshape = ov::as_type_ptr<ov::op::v1::Reshape>(source);
    if (!reshape) {
        return std::nullopt;
    }
    match.decompression_nodes.push_back(reshape);
    auto mul = ov::as_type_ptr<ov::op::v1::Multiply>(reshape->input_value(0).get_node_shared_ptr());
    if (!mul) {
        return std::nullopt;
    }
    match.decompression_nodes.push_back(mul);

    std::shared_ptr<ov::op::v0::Constant> weights;
    std::shared_ptr<ov::op::v0::Constant> scale;
    for (size_t i = 0; i < mul->get_input_size(); ++i) {
        auto input = mul->input_value(i);
        if (auto convert = ov::as_type_ptr<ov::op::v0::Convert>(input.get_node_shared_ptr())) {
            match.decompression_nodes.push_back(convert);
            if (auto constant = as_constant(convert->input_value(0))) {
                const auto et = constant->get_element_type();
                if (et == ov::element::i4 || et == ov::element::u4 ||
                    et == ov::element::i8 || et == ov::element::u8) {
                    weights = constant;
                    continue;
                }
            }
        }
        if (auto constant = as_constant(input)) {
            if (constant->get_element_type() == ov::element::f16 ||
                constant->get_element_type() == ov::element::f32) {
                scale = constant;
            }
        }
    }
    if (!weights || !scale) {
        return std::nullopt;
    }

    const auto raw_shape = weights->get_shape();
    const auto scale_shape = scale->get_shape();
    if (raw_shape.size() != 3 || scale_shape.size() != 3 ||
        raw_shape[0] != b_shape[0] ||
        scale_shape[0] != raw_shape[0] ||
        scale_shape[1] != raw_shape[1] ||
        scale_shape[2] != 1) {
        return std::nullopt;
    }
    const size_t n = raw_shape[0];
    const size_t groups = raw_shape[1];
    const size_t group_size = raw_shape[2];
    const size_t k = groups * group_size;
    if (n == 0 || k == 0 || b_shape[1] != k) {
        return std::nullopt;
    }

    match.matmul = matmul;
    match.weights = weights;
    match.scale = scale;
    match.data = matmul->input_value(0);
    match.weight_input_type = weight_input_type;
    match.n = n;
    match.groups = groups;
    match.group_size = group_size;
    match.k = k;
    return match;
}

void protect_compressed_decompression(const CompressedMatMulMatch& match) {
    for (const auto& node : match.decompression_nodes) {
        ov::mark_as_decompression(node);
        ov::disable_constant_folding(node);
    }
}

void protect_compressed_matmul_decompressions(const std::shared_ptr<ov::Model>& model) {
    size_t match_count = 0;
    size_t matmul_count = 0;
    for (const auto& node : model->get_ordered_ops()) {
        if (node->get_type_name() == std::string("MatMul")) {
            ++matmul_count;
        }
        auto match = match_compressed_matmul(node);
        if (!match) {
            continue;
        }
        protect_compressed_decompression(*match);
        ++match_count;
    }

    if (gfx_log_debug_enabled()) {
        gfx_log_debug("GfxTransforms") << "compressed MatMul decompression protection: matmuls=" << matmul_count
                                       << " matches=" << match_count;
    }
}

}  // namespace

std::shared_ptr<const ov::Model> run_pipeline(const std::shared_ptr<const ov::Model>& model,
                                              GpuBackend backend) {
    OPENVINO_ASSERT(model, "run_pipeline: model is null");
    // Work on a clone to preserve the original model passed by the caller.
    auto cloned = model->clone();

    ov::pass::Manager manager("Plugin:GFX");
    // Common optimizations from OpenVINO transformations library.
    manager.register_pass<ov::pass::CommonOptimizations>();
    // LLM RMSNorm frequently arrives as Power->ReduceMean->Add->Sqrt/Divide->Multiply->Multiply.
    // Fuse both no-tail-convert and Divide variants into one backend-lowerable op.
    manager.register_pass<ov::pass::RMSFusion>(false, true);
    // Plugin-local structural cleanup before stage selection / MLIR lowering.
    manager.register_pass<ov::gfx_plugin::transforms::GfxLayoutCleanup>();

    // Align behaviour with template plugin: disable a few transformations that can be harmful for backend mapping.
    auto pass_config = manager.get_pass_config();
    pass_config->disable<ov::pass::UnrollIf>();
    pass_config->disable<ov::pass::ConvertMaxPool14ToMaxPool8>();
    pass_config->disable<ov::pass::ConvertAvgPool14ToAvgPool1>();
    pass_config->disable<ov::pass::ConvertReduceSumToPooling>();
    pass_config->disable<ov::pass::ConvertMod>();
    if (backend == GpuBackend::Metal) {
        pass_config->disable<ov::pass::ScaledDotProductAttentionDecomposition>();
    }

    // Keep compressed weight decompression subgraphs intact for backend-side weight-only kernels.
    pass_config->disable<ov::pass::ConvertCompressedOnlyToLegacy>();

    protect_compressed_matmul_decompressions(cloned);
    cloned->validate_nodes_and_infer_types();
    manager.run_passes(cloned);
    cloned->validate_nodes_and_infer_types();
    return cloned;
}

}  // namespace transforms
}  // namespace gfx_plugin
}  // namespace ov
