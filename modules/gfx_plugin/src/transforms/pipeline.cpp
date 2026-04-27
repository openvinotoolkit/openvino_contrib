// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "transforms/pipeline.hpp"

#include "runtime/gfx_logger.hpp"
#include "transforms/gfx_layout_cleanup.hpp"

#include <map>
#include <optional>
#include <tuple>
#include <vector>

#include "openvino/core/rt_info.hpp"
#include "openvino/op/add.hpp"
#include "openvino/op/concat.hpp"
#include "openvino/pass/manager.hpp"
#include "openvino/op/constant.hpp"
#include "openvino/op/convert.hpp"
#include "openvino/op/matmul.hpp"
#include "openvino/op/multiply.hpp"
#include "openvino/op/reshape.hpp"
#include "openvino/op/slice.hpp"
#include "openvino/op/strided_slice.hpp"
#include "openvino/op/variadic_split.hpp"
#include "ov_ops/rotary_positional_embeddings.hpp"
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

struct CompressedMatMulGroupKey {
    const ov::Node* data_node = nullptr;
    size_t data_index = 0;

    bool operator<(const CompressedMatMulGroupKey& other) const {
        return std::tie(data_node, data_index) < std::tie(other.data_node, other.data_index);
    }
};

bool same_value(const ov::Output<ov::Node>& lhs, const ov::Output<ov::Node>& rhs) {
    return lhs.get_node() == rhs.get_node() && lhs.get_index() == rhs.get_index();
}

std::shared_ptr<ov::op::v0::Constant> as_constant(const ov::Output<ov::Node>& value) {
    return ov::as_type_ptr<ov::op::v0::Constant>(value.get_node_shared_ptr());
}

bool is_scalar_like(const ov::Output<ov::Node>& value) {
    const auto pshape = value.get_partial_shape();
    if (pshape.rank().is_dynamic()) {
        return false;
    }
    if (pshape.rank().get_length() == 0) {
        return true;
    }
    if (pshape.is_static()) {
        return ov::shape_size(pshape.to_shape()) == 1;
    }
    return false;
}

std::optional<ov::Output<ov::Node>> slice_data_input(const ov::Output<ov::Node>& value) {
    const auto node = value.get_node_shared_ptr();
    if (auto slice = ov::as_type_ptr<ov::op::v8::Slice>(node)) {
        return slice->input_value(0);
    }
    if (auto strided = ov::as_type_ptr<ov::op::v1::StridedSlice>(node)) {
        return strided->input_value(0);
    }
    return std::nullopt;
}

std::optional<ov::Output<ov::Node>> negated_slice_data_input(const ov::Output<ov::Node>& value) {
    auto mul = ov::as_type_ptr<ov::op::v1::Multiply>(value.get_node_shared_ptr());
    if (!mul || mul->get_input_size() != 2) {
        return std::nullopt;
    }
    for (size_t idx = 0; idx < 2; ++idx) {
        auto maybe_slice = slice_data_input(mul->input_value(idx));
        if (maybe_slice && is_scalar_like(mul->input_value(1 - idx))) {
            return maybe_slice;
        }
    }
    return std::nullopt;
}

struct RotateHalfMatch {
    ov::Output<ov::Node> data;
};

std::optional<RotateHalfMatch> match_rotate_half_concat(const ov::Output<ov::Node>& value) {
    auto concat = ov::as_type_ptr<ov::op::v0::Concat>(value.get_node_shared_ptr());
    if (!concat || concat->get_input_size() != 2) {
        return std::nullopt;
    }
    const auto out_pshape = concat->get_output_partial_shape(0);
    if (out_pshape.rank().is_dynamic() || out_pshape.rank().get_length() == 0 ||
        out_pshape[out_pshape.rank().get_length() - 1].is_dynamic()) {
        return std::nullopt;
    }
    const auto rank = static_cast<size_t>(out_pshape.rank().get_length());
    const auto axis = concat->get_axis() < 0
                          ? static_cast<int64_t>(rank) + concat->get_axis()
                          : concat->get_axis();
    if (axis != static_cast<int64_t>(rank - 1)) {
        return std::nullopt;
    }
    const auto head_size = static_cast<size_t>(out_pshape[rank - 1].get_length());
    if (head_size == 0 || (head_size % 2) != 0) {
        return std::nullopt;
    }
    const auto first_pshape = concat->get_input_partial_shape(0);
    const auto second_pshape = concat->get_input_partial_shape(1);
    if (first_pshape.rank().is_dynamic() || second_pshape.rank().is_dynamic() ||
        first_pshape.rank().get_length() != static_cast<int64_t>(rank) ||
        second_pshape.rank().get_length() != static_cast<int64_t>(rank) ||
        first_pshape[rank - 1].is_dynamic() || second_pshape[rank - 1].is_dynamic() ||
        static_cast<size_t>(first_pshape[rank - 1].get_length()) != head_size / 2 ||
        static_cast<size_t>(second_pshape[rank - 1].get_length()) != head_size / 2) {
        return std::nullopt;
    }

    auto neg_src = negated_slice_data_input(concat->input_value(0));
    auto pos_src = slice_data_input(concat->input_value(1));
    if (!neg_src || !pos_src || !same_value(*neg_src, *pos_src)) {
        return std::nullopt;
    }
    return RotateHalfMatch{*neg_src};
}

std::optional<ov::Output<ov::Node>> non_matching_multiply_input(const std::shared_ptr<ov::op::v1::Multiply>& mul,
                                                                const ov::Output<ov::Node>& match) {
    if (!mul || mul->get_input_size() != 2) {
        return std::nullopt;
    }
    if (same_value(mul->input_value(0), match)) {
        return mul->input_value(1);
    }
    if (same_value(mul->input_value(1), match)) {
        return mul->input_value(0);
    }
    return std::nullopt;
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

bool compatible_for_horizontal_fusion(const CompressedMatMulMatch& base,
                                      const CompressedMatMulMatch& candidate,
                                      size_t output_rank) {
    if (candidate.data.get_node() != base.data.get_node() ||
        candidate.data.get_index() != base.data.get_index() ||
        candidate.matmul->get_transpose_a() != base.matmul->get_transpose_a() ||
        candidate.weight_input_type != base.weight_input_type ||
        candidate.weights->get_element_type() != base.weights->get_element_type() ||
        candidate.scale->get_element_type() != base.scale->get_element_type() ||
        candidate.groups != base.groups ||
        candidate.group_size != base.group_size ||
        candidate.k != base.k) {
        return false;
    }
    const auto output_pshape = candidate.matmul->get_output_partial_shape(0);
    if (output_pshape.rank().is_dynamic() ||
        static_cast<size_t>(output_pshape.rank().get_length()) != output_rank ||
        output_pshape[output_rank - 1].is_dynamic() ||
        static_cast<size_t>(output_pshape[output_rank - 1].get_length()) != candidate.n) {
        return false;
    }
    return true;
}

size_t fuse_compressed_matmul_horizontal(const std::shared_ptr<ov::Model>& model) {
    std::map<CompressedMatMulGroupKey, std::vector<CompressedMatMulMatch>> groups;
    for (const auto& node : model->get_ordered_ops()) {
        auto match = match_compressed_matmul(node);
        if (!match) {
            continue;
        }
        const CompressedMatMulGroupKey key{match->data.get_node(), match->data.get_index()};
        groups[key].push_back(std::move(*match));
    }

    size_t fused_groups = 0;
    size_t fused_matmuls = 0;
    for (auto& entry : groups) {
        auto& matches = entry.second;
        if (matches.size() < 2) {
            continue;
        }

        const auto output_pshape = matches.front().matmul->get_output_partial_shape(0);
        if (output_pshape.rank().is_dynamic()) {
            continue;
        }
        const size_t output_rank = static_cast<size_t>(output_pshape.rank().get_length());
        if (output_rank == 0 ||
            output_pshape[output_rank - 1].is_dynamic() ||
            static_cast<size_t>(output_pshape[output_rank - 1].get_length()) != matches.front().n) {
            continue;
        }

        bool compatible = true;
        size_t total_n = 0;
        std::vector<ov::Output<ov::Node>> weight_inputs;
        std::vector<int64_t> split_lengths;
        ov::NodeVector rt_sources;
        weight_inputs.reserve(matches.size());
        split_lengths.reserve(matches.size());
        for (const auto& match : matches) {
            if (!compatible_for_horizontal_fusion(matches.front(), match, output_rank)) {
                compatible = false;
                break;
            }
            total_n += match.n;
            weight_inputs.push_back(match.matmul->input_value(1));
            split_lengths.push_back(static_cast<int64_t>(match.n));
            rt_sources.push_back(match.matmul);
            for (const auto& node : match.decompression_nodes) {
                rt_sources.push_back(node);
            }
        }
        if (!compatible || total_n == 0) {
            continue;
        }

        auto concat = std::make_shared<ov::op::v0::Concat>(weight_inputs, 0);
        concat->set_friendly_name(matches.front().matmul->get_friendly_name() + "/gfx_fused_weight_concat");
        ov::mark_as_decompression(concat);
        ov::disable_constant_folding(concat);

        auto fused_matmul = std::make_shared<ov::op::v0::MatMul>(matches.front().data,
                                                                 concat,
                                                                 matches.front().matmul->get_transpose_a(),
                                                                 true);
        fused_matmul->set_friendly_name(matches.front().matmul->get_friendly_name() + "/gfx_fused_horizontal");

        auto axis = ov::op::v0::Constant::create(ov::element::i64, ov::Shape{}, {static_cast<int64_t>(output_rank - 1)});
        auto lengths = ov::op::v0::Constant::create(ov::element::i64,
                                                    ov::Shape{split_lengths.size()},
                                                    split_lengths);
        auto split = std::make_shared<ov::op::v1::VariadicSplit>(fused_matmul, axis, lengths);
        split->set_friendly_name(matches.front().matmul->get_friendly_name() + "/gfx_fused_horizontal_split");
        ov::copy_runtime_info(rt_sources, ov::NodeVector{concat, fused_matmul, split});

        split->validate_and_infer_types();
        for (size_t i = 0; i < matches.size(); ++i) {
            matches[i].matmul->output(0).replace(split->output(i));
        }
        ++fused_groups;
        fused_matmuls += matches.size();
    }

    if (gfx_log_debug_enabled()) {
        gfx_log_debug("GfxTransforms") << "compressed MatMul horizontal fusion: groups=" << fused_groups
                                       << " matmuls=" << fused_matmuls;
    }
    return fused_groups;
}

size_t fuse_llama_rotate_half_rope(const std::shared_ptr<ov::Model>& model) {
    size_t fused = 0;
    for (const auto& node : model->get_ordered_ops()) {
        auto add = ov::as_type_ptr<ov::op::v1::Add>(node);
        if (!add || add->get_input_size() != 2 || add->get_output_size() != 1) {
            continue;
        }
        auto mul0 = ov::as_type_ptr<ov::op::v1::Multiply>(add->input_value(0).get_node_shared_ptr());
        auto mul1 = ov::as_type_ptr<ov::op::v1::Multiply>(add->input_value(1).get_node_shared_ptr());
        if (!mul0 || !mul1) {
            continue;
        }

        std::shared_ptr<ov::op::v1::Multiply> direct_mul;
        std::shared_ptr<ov::op::v1::Multiply> rotated_mul;
        std::optional<RotateHalfMatch> rotate;
        std::optional<ov::Output<ov::Node>> rotate_input;
        for (size_t idx = 0; idx < 2; ++idx) {
            auto candidate_rotated = idx == 0 ? mul0 : mul1;
            for (size_t input_idx = 0; input_idx < 2; ++input_idx) {
                auto candidate = match_rotate_half_concat(candidate_rotated->input_value(input_idx));
                if (!candidate) {
                    continue;
                }
                direct_mul = idx == 0 ? mul1 : mul0;
                rotated_mul = candidate_rotated;
                rotate = candidate;
                rotate_input = candidate_rotated->input_value(input_idx);
                break;
            }
            if (rotate) {
                break;
            }
        }
        if (!rotate || !rotate_input) {
            continue;
        }

        auto cos = non_matching_multiply_input(direct_mul, rotate->data);
        auto sin = non_matching_multiply_input(rotated_mul, *rotate_input);
        if (!cos || !sin) {
            continue;
        }
        const auto data_pshape = rotate->data.get_partial_shape();
        if (data_pshape.rank().is_dynamic() || data_pshape.rank().get_length() < 2 ||
            data_pshape[data_pshape.rank().get_length() - 1].is_dynamic()) {
            continue;
        }
        const auto head_size = static_cast<size_t>(data_pshape[data_pshape.rank().get_length() - 1].get_length());
        if (head_size == 0 || (head_size % 2) != 0) {
            continue;
        }

        ov::op::internal::RoPE::Config cfg;
        cfg.rotary_ndims = head_size;
        cfg.cos_sin_ndims = head_size;
        auto rope = std::make_shared<ov::op::internal::RoPE>(ov::OutputVector{rotate->data, *cos, *sin}, cfg);
        rope->set_friendly_name(add->get_friendly_name() + "/gfx_rope");
        ov::copy_runtime_info({add, direct_mul, rotated_mul, rotate_input->get_node_shared_ptr()}, rope);
        rope->validate_and_infer_types();
        add->output(0).replace(rope->output(0));
        ++fused;
    }

    if (gfx_log_debug_enabled()) {
        gfx_log_debug("GfxTransforms") << "LLaMA rotate-half RoPE fusion: fused=" << fused;
    }
    return fused;
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
    if (backend == GpuBackend::Metal) {
        fuse_llama_rotate_half_rope(cloned);
    }
    fuse_compressed_matmul_horizontal(cloned);
    cloned->validate_nodes_and_infer_types();
    return cloned;
}

}  // namespace transforms
}  // namespace gfx_plugin
}  // namespace ov
