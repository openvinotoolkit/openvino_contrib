// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "plugin/infer_submission.hpp"

#include <algorithm>
#include <chrono>
#include <limits>
#include <sstream>
#include <string_view>
#include <unordered_map>

#include "openvino/core/shape_util.hpp"
#include "openvino/op/convolution.hpp"
#include "openvino/op/group_conv.hpp"
#include "openvino/op/matmul.hpp"
#include "runtime/gfx_logger.hpp"

namespace ov {
namespace gfx_plugin {

namespace {

struct StageProfileEstimate {
    uint64_t bytes_in = 0;
    uint64_t bytes_out = 0;
    uint64_t macs = 0;
    uint64_t flops = 0;
};

constexpr size_t MiB(size_t value) {
    return value * 1024u * 1024u;
}

constexpr uint64_t GMacs(uint64_t value) {
    return value * 1000ull * 1000ull * 1000ull;
}

enum class SubmissionDeviceClass {
    Mobile,
    Constrained,
    Wide,
};

enum class SubmissionDepthClass {
    Shallow,
    Deep,
    VeryDeep,
    ExtremelyDeep,
};

struct SubmissionWorkloadProfile {
    size_t pipeline_stages = 1;
    uint32_t simd_width = 1;
    uint32_t max_threads_per_group = 1;
    SubmissionDeviceClass device_class = SubmissionDeviceClass::Wide;
    SubmissionDepthClass depth_class = SubmissionDepthClass::Shallow;
};

struct SubmissionDeviceProfile {
    size_t max_slots = 4;
    size_t stages_per_slot = 96;
    uint64_t mac_budget_scale_num = 1;
    uint64_t mac_budget_scale_den = 1;
    size_t extremely_deep_vulkan_stage_floor = 0;
    size_t extremely_deep_vulkan_output_floor = 0;
    uint64_t extremely_deep_vulkan_mac_floor = 0;
};

struct SubmissionBudget {
    size_t weighted_stages = 1;
    size_t output_bytes = 1;
    uint64_t macs = 1;
};

struct SubmissionDependencyGraph {
    std::vector<std::vector<size_t>> input_producers;
};

struct SubmissionWindowTotals {
    size_t weighted_stages = 0;
    size_t output_bytes = 0;
    uint64_t macs = 0;
};

bool is_deep(SubmissionDepthClass depth_class) {
    return depth_class == SubmissionDepthClass::Deep ||
           depth_class == SubmissionDepthClass::VeryDeep ||
           depth_class == SubmissionDepthClass::ExtremelyDeep;
}

bool is_very_deep(SubmissionDepthClass depth_class) {
    return depth_class == SubmissionDepthClass::VeryDeep ||
           depth_class == SubmissionDepthClass::ExtremelyDeep;
}

SubmissionDeviceClass classify_submission_device(uint32_t max_threads_per_group) {
    if (max_threads_per_group <= 128u) {
        return SubmissionDeviceClass::Mobile;
    }
    if (max_threads_per_group <= 256u) {
        return SubmissionDeviceClass::Constrained;
    }
    return SubmissionDeviceClass::Wide;
}

SubmissionDepthClass classify_submission_depth(size_t pipeline_stages) {
    if (pipeline_stages >= 256u) {
        return SubmissionDepthClass::ExtremelyDeep;
    }
    if (pipeline_stages >= 128u) {
        return SubmissionDepthClass::VeryDeep;
    }
    if (pipeline_stages >= 64u) {
        return SubmissionDepthClass::Deep;
    }
    return SubmissionDepthClass::Shallow;
}

SubmissionWorkloadProfile make_submission_workload_profile(const InferSubmissionTuningCaps& caps,
                                                           size_t stage_count) {
    SubmissionWorkloadProfile workload{};
    workload.pipeline_stages = std::max<size_t>(stage_count, 1u);
    workload.simd_width = std::max<uint32_t>(1u, std::max(caps.subgroup_size, caps.preferred_simd_width));
    workload.max_threads_per_group = std::max<uint32_t>(caps.max_total_threads_per_group, 1u);
    workload.device_class = classify_submission_device(workload.max_threads_per_group);
    workload.depth_class = classify_submission_depth(workload.pipeline_stages);
    return workload;
}

SubmissionDeviceProfile make_submission_device_profile(const InferSubmissionTuningCaps& caps,
                                                       const SubmissionWorkloadProfile& workload) {
    SubmissionDeviceProfile profile{};
    switch (workload.device_class) {
    case SubmissionDeviceClass::Mobile:
        profile.max_slots = 8u;
        profile.stages_per_slot = workload.simd_width >= 64u ? 48u : 32u;
        break;
    case SubmissionDeviceClass::Constrained:
        profile.max_slots = 6u;
        profile.stages_per_slot = workload.simd_width >= 64u ? 64u : 48u;
        profile.extremely_deep_vulkan_stage_floor =
            profile.stages_per_slot + profile.stages_per_slot / 2u;
        profile.extremely_deep_vulkan_output_floor = workload.simd_width >= 64u ? MiB(96u) : MiB(64u);
        profile.extremely_deep_vulkan_mac_floor = workload.simd_width >= 64u ? GMacs(8u) : GMacs(4u);
        break;
    case SubmissionDeviceClass::Wide:
        profile.max_slots = caps.backend == GpuBackend::Vulkan ? 6u : 4u;
        profile.stages_per_slot = workload.simd_width >= 64u
                                      ? (caps.backend == GpuBackend::Vulkan ? 72u : 96u)
                                      : (caps.backend == GpuBackend::Vulkan ? 56u : 64u);
        break;
    }

    // Family-specific knobs refine the common workload class. The hierarchy stays:
    // workload shape -> generic device class -> concrete family coefficient.
    switch (caps.device_family) {
    case GpuDeviceFamily::BroadcomV3D:
        profile.mac_budget_scale_num = workload.simd_width >= 64u ? 3u : 2u;
        profile.mac_budget_scale_den = workload.simd_width >= 64u ? 4u : 3u;
        break;
    case GpuDeviceFamily::Apple:
        profile.mac_budget_scale_num = 3u;
        profile.mac_budget_scale_den = 2u;
        break;
    case GpuDeviceFamily::QualcommAdreno:
    case GpuDeviceFamily::Generic:
    default:
        break;
    }
    return profile;
}

SubmissionBudget make_base_submission_budget(const SubmissionWorkloadProfile& workload) {
    SubmissionBudget budget{};
    switch (workload.device_class) {
    case SubmissionDeviceClass::Mobile:
        budget.weighted_stages = workload.simd_width >= 64u ? 12u : 8u;
        budget.output_bytes = MiB(24u);
        budget.macs = workload.simd_width >= 64u ? GMacs(2u) : GMacs(1u);
        if (is_deep(workload.depth_class)) {
            budget.weighted_stages = workload.simd_width >= 64u ? 10u : 6u;
            budget.output_bytes = MiB(16u);
            budget.macs = workload.simd_width >= 64u ? 1500ull * 1000ull * 1000ull
                                                     : 750ull * 1000ull * 1000ull;
        }
        if (is_very_deep(workload.depth_class)) {
            budget.weighted_stages = workload.simd_width >= 64u ? 8u : 4u;
            budget.output_bytes = MiB(8u);
            budget.macs = workload.simd_width >= 64u ? GMacs(1u) : 500ull * 1000ull * 1000ull;
        }
        break;
    case SubmissionDeviceClass::Constrained:
        budget.weighted_stages = workload.simd_width >= 64u ? 20u : 16u;
        budget.output_bytes = MiB(48u);
        budget.macs = workload.simd_width >= 64u ? GMacs(8u) : GMacs(6u);
        if (is_very_deep(workload.depth_class)) {
            budget.weighted_stages = workload.simd_width >= 64u ? 16u : 12u;
            budget.output_bytes = MiB(32u);
            budget.macs = workload.simd_width >= 64u ? GMacs(6u) : GMacs(4u);
        } else if (is_deep(workload.depth_class)) {
            budget.weighted_stages = workload.simd_width >= 64u ? 18u : 14u;
            budget.output_bytes = MiB(40u);
            budget.macs = workload.simd_width >= 64u ? GMacs(7u) : GMacs(5u);
        }
        break;
    case SubmissionDeviceClass::Wide:
        budget.weighted_stages = workload.simd_width >= 64u ? 24u : 20u;
        budget.output_bytes = MiB(64u);
        budget.macs = workload.simd_width >= 64u ? GMacs(16u) : GMacs(12u);
        if (is_very_deep(workload.depth_class)) {
            budget.weighted_stages = workload.simd_width >= 64u ? 20u : 16u;
            budget.output_bytes = MiB(48u);
            budget.macs = workload.simd_width >= 64u ? GMacs(12u) : GMacs(10u);
        } else if (is_deep(workload.depth_class)) {
            budget.weighted_stages = workload.simd_width >= 64u ? 22u : 18u;
            budget.output_bytes = MiB(56u);
            budget.macs = workload.simd_width >= 64u ? GMacs(14u) : GMacs(11u);
        }
        break;
    }
    return budget;
}

uint64_t scale_macs(uint64_t macs, uint64_t numerator, uint64_t denominator) {
    if (denominator == 0u || numerator == denominator) {
        return macs;
    }
    if (numerator > 1u && macs > std::numeric_limits<uint64_t>::max() / numerator) {
        return std::numeric_limits<uint64_t>::max();
    }
    return std::max<uint64_t>((macs * numerator) / denominator, 1u);
}

void apply_submission_profile_budget(const InferSubmissionTuningCaps& caps,
                                     const SubmissionWorkloadProfile& workload,
                                     const SubmissionDeviceProfile& profile,
                                     SubmissionBudget& budget) {
    if (caps.backend == GpuBackend::Vulkan &&
        workload.depth_class == SubmissionDepthClass::ExtremelyDeep &&
        profile.extremely_deep_vulkan_stage_floor != 0u) {
        budget.weighted_stages =
            std::max<size_t>(budget.weighted_stages, profile.extremely_deep_vulkan_stage_floor);
        budget.output_bytes = std::max<size_t>(budget.output_bytes, profile.extremely_deep_vulkan_output_floor);
        budget.macs = std::max<uint64_t>(budget.macs, profile.extremely_deep_vulkan_mac_floor);
    }
    budget.macs = scale_macs(budget.macs, profile.mac_budget_scale_num, profile.mac_budget_scale_den);
}

size_t select_submission_slot_count(const SubmissionWorkloadProfile& workload,
                                    const SubmissionDeviceProfile& profile) {
    size_t slot_count = 1u;
    if (is_deep(workload.depth_class)) {
        slot_count = std::min(profile.max_slots,
                              1u + ((workload.pipeline_stages - 1u) /
                                    std::max<size_t>(profile.stages_per_slot, 1u)));
    } else if (workload.pipeline_stages >= std::max<size_t>(profile.stages_per_slot / 2u, 2u)) {
        slot_count = std::min<size_t>(2u, profile.max_slots);
    }

    if (is_very_deep(workload.depth_class)) {
        slot_count = std::max<size_t>(slot_count, std::min<size_t>(profile.max_slots, 3u));
    }
    if (workload.depth_class == SubmissionDepthClass::ExtremelyDeep) {
        slot_count = std::max<size_t>(slot_count, std::min<size_t>(profile.max_slots, 4u));
    }
    return slot_count;
}

uint64_t safe_shape_size_u64(const ov::Shape& shape) {
    return static_cast<uint64_t>(ov::shape_size(shape));
}

void add_saturating(uint64_t& dst, uint64_t value) {
    if (std::numeric_limits<uint64_t>::max() - dst < value) {
        dst = std::numeric_limits<uint64_t>::max();
        return;
    }
    dst += value;
}

size_t add_saturating_size(size_t lhs, size_t rhs) {
    if (std::numeric_limits<size_t>::max() - lhs < rhs) {
        return std::numeric_limits<size_t>::max();
    }
    return lhs + rhs;
}

uint64_t add_saturating_u64(uint64_t lhs, uint64_t rhs) {
    if (std::numeric_limits<uint64_t>::max() - lhs < rhs) {
        return std::numeric_limits<uint64_t>::max();
    }
    return lhs + rhs;
}

size_t scale_size_budget(size_t value, uint32_t numerator, uint32_t denominator) {
    if (denominator == 0u || numerator == denominator) {
        return value;
    }
    if (numerator > denominator && value > std::numeric_limits<size_t>::max() / numerator) {
        return std::numeric_limits<size_t>::max();
    }
    return std::max<size_t>((value * numerator) / denominator, 1u);
}

uint64_t scale_u64_budget(uint64_t value, uint32_t numerator, uint32_t denominator) {
    if (denominator == 0u || numerator == denominator) {
        return value;
    }
    if (numerator > denominator && value > std::numeric_limits<uint64_t>::max() / numerator) {
        return std::numeric_limits<uint64_t>::max();
    }
    return std::max<uint64_t>((value * numerator) / denominator, 1u);
}

uint64_t mul_saturating(uint64_t lhs, uint64_t rhs) {
    if (lhs == 0 || rhs == 0) {
        return 0;
    }
    if (lhs > std::numeric_limits<uint64_t>::max() / rhs) {
        return std::numeric_limits<uint64_t>::max();
    }
    return lhs * rhs;
}

uint64_t tensor_bytes_estimate(const GpuTensor* tensor) {
    if (!tensor) {
        return 0;
    }
    if (tensor->buf.valid() && tensor->buf.size != 0) {
        return static_cast<uint64_t>(tensor->buf.size);
    }
    if (tensor->shape.empty() || tensor->expected_type == ov::element::dynamic ||
        tensor->expected_type.bitwidth() == 0) {
        return 0;
    }
    return mul_saturating(safe_shape_size_u64(tensor->shape),
                          static_cast<uint64_t>(tensor->expected_type.size()));
}

uint64_t output_bytes_estimate(const InferStage& stage) {
    uint64_t bytes = 0;
    for (const auto& output : stage.outputs) {
        add_saturating(bytes, tensor_bytes_estimate(output.get()));
    }
    return bytes;
}

uint64_t input_bytes_estimate(const std::vector<GpuTensor*>& inputs) {
    uint64_t bytes = 0;
    for (const auto* input : inputs) {
        add_saturating(bytes, tensor_bytes_estimate(input));
    }
    return bytes;
}

uint64_t conv_macs_estimate(const ov::Node& node) {
    try {
        if (auto conv = dynamic_cast<const ov::op::v1::Convolution*>(&node)) {
            const auto weights = conv->get_input_shape(1);
            const auto output = conv->get_output_shape(0);
            if (weights.size() != 4 || output.size() != 4) {
                return 0;
            }
            const uint64_t output_elems = safe_shape_size_u64(output);
            const uint64_t reduction =
                mul_saturating(static_cast<uint64_t>(weights[1]),
                               mul_saturating(static_cast<uint64_t>(weights[2]),
                                              static_cast<uint64_t>(weights[3])));
            return mul_saturating(output_elems, reduction);
        }
        if (auto group_conv = dynamic_cast<const ov::op::v1::GroupConvolution*>(&node)) {
            const auto weights = group_conv->get_input_shape(1);
            const auto output = group_conv->get_output_shape(0);
            if (weights.size() != 5 || output.size() != 4) {
                return 0;
            }
            const uint64_t output_elems = safe_shape_size_u64(output);
            const uint64_t reduction =
                mul_saturating(static_cast<uint64_t>(weights[2]),
                               mul_saturating(static_cast<uint64_t>(weights[3]),
                                              static_cast<uint64_t>(weights[4])));
            return mul_saturating(output_elems, reduction);
        }
    } catch (const std::exception&) {
        return 0;
    }
    return 0;
}

uint64_t matmul_macs_estimate(const ov::Node& node) {
    try {
        auto matmul = dynamic_cast<const ov::op::v0::MatMul*>(&node);
        if (!matmul) {
            return 0;
        }
        const auto a = matmul->get_input_shape(0);
        const auto b = matmul->get_input_shape(1);
        const auto output = matmul->get_output_shape(0);
        if (a.size() < 2 || b.size() < 2 || output.size() < 2) {
            return 0;
        }
        const auto k = matmul->get_transpose_a() ? a[a.size() - 2] : a[a.size() - 1];
        return mul_saturating(safe_shape_size_u64(output), static_cast<uint64_t>(k));
    } catch (const std::exception&) {
        return 0;
    }
}

StageProfileEstimate estimate_stage_profile(const InferStage& stage,
                                            const std::vector<GpuTensor*>& resolved_inputs) {
    StageProfileEstimate estimate{};
    estimate.bytes_in = input_bytes_estimate(resolved_inputs);
    estimate.bytes_out = output_bytes_estimate(stage);
    if (stage.node) {
        estimate.macs = conv_macs_estimate(*stage.node);
        if (estimate.macs == 0) {
            estimate.macs = matmul_macs_estimate(*stage.node);
        }
    }
    estimate.flops = mul_saturating(estimate.macs, 2);
    return estimate;
}

size_t stage_output_bytes_for_budget(uint64_t bytes_out) {
    return static_cast<size_t>(std::min<uint64_t>(bytes_out, std::numeric_limits<size_t>::max()));
}

void register_submission_stage_output(
    std::unordered_map<NodePortKey, StageOutputRef, NodePortKeyHash>& output_by_source,
    const std::shared_ptr<const ov::Node>& node,
    size_t source_port,
    size_t stage_index,
    size_t output_index,
    const InferStage& stage) {
    if (!node || output_index >= stage.outputs.size()) {
        return;
    }
    output_by_source[NodePortKey{node.get(), source_port}] = StageOutputRef{stage_index, output_index};
}

SubmissionDependencyGraph build_submission_dependency_graph(const std::vector<InferStage>& pipeline) {
    SubmissionDependencyGraph graph{};
    graph.input_producers.resize(pipeline.size());
    std::unordered_map<NodePortKey, StageOutputRef, NodePortKeyHash> output_by_source;

    for (size_t stage_index = 0; stage_index < pipeline.size(); ++stage_index) {
        const auto& stage = pipeline[stage_index];
        if (stage.node) {
            for (size_t output_index = 0; output_index < stage.outputs.size(); ++output_index) {
                register_submission_stage_output(output_by_source,
                                                 stage.node,
                                                 output_index,
                                                 stage_index,
                                                 output_index,
                                                 stage);
            }
        }
        for (size_t output_index = 0; output_index < stage.output_sources.size(); ++output_index) {
            const auto& source = stage.output_sources[output_index];
            register_submission_stage_output(output_by_source,
                                             source.node,
                                             source.port,
                                             stage_index,
                                             output_index,
                                             stage);
        }
        for (const auto& alias : stage.output_aliases) {
            register_submission_stage_output(output_by_source,
                                             alias.node,
                                             alias.source_port,
                                             stage_index,
                                             alias.output_port,
                                             stage);
        }
    }

    for (size_t stage_index = 0; stage_index < pipeline.size(); ++stage_index) {
        auto& producers = graph.input_producers[stage_index];
        const auto& stage = pipeline[stage_index];
        for (const auto& input : stage.inputs) {
            if (!input.node) {
                continue;
            }
            const auto it = output_by_source.find(NodePortKey{input.node.get(), input.port});
            if (it == output_by_source.end() || it->second.stage == stage_index) {
                continue;
            }
            producers.push_back(it->second.stage);
        }
        std::sort(producers.begin(), producers.end());
        producers.erase(std::unique(producers.begin(), producers.end()), producers.end());
    }

    return graph;
}

bool stage_depends_on_recording_window(size_t stage_index,
                                       const SubmissionDependencyGraph& graph,
                                       const std::vector<bool>& recorded_stage_mask) {
    if (stage_index >= graph.input_producers.size()) {
        return false;
    }
    for (const auto producer_index : graph.input_producers[stage_index]) {
        if (producer_index < recorded_stage_mask.size() && recorded_stage_mask[producer_index]) {
            return true;
        }
    }
    return false;
}

bool is_dependency_extension_boundary(std::string_view stage_type) {
    return stage_type == "Concat" ||
           stage_type == "Split" ||
           stage_type == "VariadicSplit" ||
           stage_type == "Transpose" ||
           stage_type == "Reshape" ||
           stage_type == "Squeeze" ||
           stage_type == "Unsqueeze" ||
           stage_type == "Softmax" ||
           stage_type == "LogSoftmax" ||
           stage_type == "FusedAttention";
}

bool dependency_extension_crosses_boundary(size_t stage_index,
                                           const SubmissionDependencyGraph& graph,
                                           const std::vector<bool>& recorded_stage_mask,
                                           const std::vector<InferStage>& pipeline) {
    if (stage_index >= pipeline.size()) {
        return true;
    }
    const auto& stage = pipeline[stage_index];
    if (stage.stage && is_dependency_extension_boundary(stage.stage->type())) {
        return true;
    }
    if (stage_index >= graph.input_producers.size()) {
        return false;
    }
    for (const auto producer_index : graph.input_producers[stage_index]) {
        if (producer_index >= recorded_stage_mask.size() || !recorded_stage_mask[producer_index] ||
            producer_index >= pipeline.size()) {
            continue;
        }
        const auto& producer = pipeline[producer_index];
        if (producer.stage && is_dependency_extension_boundary(producer.stage->type())) {
            return true;
        }
    }
    return false;
}

size_t next_executable_stage_index(size_t stage_index, const std::vector<InferStage>& pipeline) {
    for (size_t next_index = stage_index + 1u; next_index < pipeline.size(); ++next_index) {
        if (pipeline[next_index].stage) {
            return next_index;
        }
    }
    return std::numeric_limits<size_t>::max();
}

SubmissionWindowTotals extended_window_totals(size_t recorded_stage_count,
                                              size_t recorded_output_bytes,
                                              uint64_t recorded_macs,
                                              size_t stage_weight,
                                              const StageProfileEstimate& estimate) {
    SubmissionWindowTotals totals{};
    totals.weighted_stages = add_saturating_size(recorded_stage_count, stage_weight);
    totals.output_bytes =
        add_saturating_size(recorded_output_bytes, stage_output_bytes_for_budget(estimate.bytes_out));
    totals.macs = add_saturating_u64(recorded_macs, estimate.macs);
    return totals;
}

bool within_dependency_extension_cap(const SubmissionWindowTotals& totals,
                                     const InferSubmissionConfig& config) {
    // Direct producer -> consumer chains may extend past the soft window budget,
    // but the extension cap is still a scheduler budget derived from the common
    // workload/device-class tuning, not a device-name shortcut.
    const uint32_t numerator = std::max<uint32_t>(config.dependency_extension_budget_num, 1u);
    const uint32_t denominator = std::max<uint32_t>(config.dependency_extension_budget_den, 1u);
    const size_t stage_cap = scale_size_budget(config.max_stages_per_submit,
                                               numerator,
                                               denominator);
    const size_t output_cap = scale_size_budget(config.max_output_bytes_per_submit,
                                                numerator,
                                                denominator);
    const uint64_t mac_cap = scale_u64_budget(config.max_macs_per_submit,
                                              numerator,
                                              denominator);
    return totals.weighted_stages <= stage_cap &&
           totals.output_bytes <= output_cap &&
           totals.macs <= mac_cap;
}

}  // namespace

InferSubmissionTuning select_infer_submission_tuning(const InferSubmissionTuningCaps& caps, size_t stage_count) {
    InferSubmissionTuning tuning{};
    tuning.slot_count = 1u;

    const SubmissionWorkloadProfile workload = make_submission_workload_profile(caps, stage_count);
    if (!caps.supports_incremental_submit) {
        tuning.config.max_stages_per_submit = workload.pipeline_stages;
        tuning.config.max_output_bytes_per_submit = std::numeric_limits<size_t>::max() / 4u;
        tuning.config.max_macs_per_submit = std::numeric_limits<uint64_t>::max() / 4u;
        tuning.config.allow_incremental_submit = false;
        return tuning;
    }

    const SubmissionDeviceProfile device_profile = make_submission_device_profile(caps, workload);
    SubmissionBudget budget = make_base_submission_budget(workload);
    apply_submission_profile_budget(caps, workload, device_profile, budget);
    tuning.slot_count = select_submission_slot_count(workload, device_profile);
    size_t slot_parallelism = 1u;

    if (caps.supports_incremental_submit && tuning.slot_count > 1u) {
        slot_parallelism = std::min<size_t>(tuning.slot_count, 4u);
        const size_t extra_stage_budget =
            (slot_parallelism - 1u) * std::max<size_t>(budget.weighted_stages / 3u, 2u);
        const size_t extra_output_budget =
            (slot_parallelism - 1u) * std::max<size_t>(budget.output_bytes / 4u, MiB(4u));
        uint64_t extra_mac_budget = 0;
        for (size_t slot = 1u; slot < slot_parallelism; ++slot) {
            add_saturating(extra_mac_budget, std::max<uint64_t>(budget.macs / 3u, GMacs(1u)));
        }
        constexpr uint64_t kMaxWindowMacBudget = std::numeric_limits<uint64_t>::max() / 4u;
        budget.weighted_stages = std::min(workload.pipeline_stages, budget.weighted_stages + extra_stage_budget);
        budget.output_bytes =
            std::min(std::numeric_limits<size_t>::max() / 4u, budget.output_bytes + extra_output_budget);
        if (budget.macs < kMaxWindowMacBudget) {
            budget.macs = std::min(kMaxWindowMacBudget,
                                   budget.macs + std::min(extra_mac_budget, kMaxWindowMacBudget - budget.macs));
        } else {
            budget.macs = kMaxWindowMacBudget;
        }
    }

    tuning.config.max_stages_per_submit = std::max<size_t>(budget.weighted_stages, 1u);
    tuning.config.max_output_bytes_per_submit = std::max<size_t>(budget.output_bytes, 1u);
    tuning.config.max_macs_per_submit = std::max<uint64_t>(budget.macs, 1u);
    if (caps.backend == GpuBackend::Vulkan &&
        workload.depth_class == SubmissionDepthClass::ExtremelyDeep &&
        workload.device_class != SubmissionDeviceClass::Wide &&
        slot_parallelism > 1u) {
        tuning.config.dependency_extension_budget_num = static_cast<uint32_t>(slot_parallelism + 1u);
        tuning.config.dependency_extension_budget_den = static_cast<uint32_t>(slot_parallelism);
    }
    tuning.config.allow_incremental_submit = true;
    return tuning;
}

void record_infer_submission_tuning_counters(const InferSubmissionTuning& tuning,
                                             const InferSubmissionTuningCaps& caps,
                                             GfxProfiler* profiler) {
    if (!profiler) {
        return;
    }
    profiler->set_counter("submission_slot_count", tuning.slot_count);
    profiler->set_counter("submission_max_stages_per_window", tuning.config.max_stages_per_submit);
    profiler->set_counter("submission_max_output_bytes_per_window", tuning.config.max_output_bytes_per_submit);
    profiler->set_counter("submission_max_macs_per_window", tuning.config.max_macs_per_submit);
    profiler->set_counter("submission_dependency_extension_budget_num",
                          tuning.config.dependency_extension_budget_num);
    profiler->set_counter("submission_dependency_extension_budget_den",
                          tuning.config.dependency_extension_budget_den);
    profiler->set_counter("submission_preferred_simd_width", std::max<uint32_t>(caps.preferred_simd_width, 1u));
    profiler->set_counter("submission_subgroup_size", std::max<uint32_t>(caps.subgroup_size, 1u));
    profiler->set_counter("submission_max_total_threads_per_group",
                          std::max<uint32_t>(caps.max_total_threads_per_group, 1u));
    profiler->set_counter("submission_incremental_submit_supported", caps.supports_incremental_submit ? 1u : 0u);
    profiler->set_counter("submission_incremental_submit_enabled", tuning.config.allow_incremental_submit ? 1u : 0u);
}

void TrackedInferSubmissionSession::begin_recording() {
    m_completed_command_buffer = nullptr;
    m_current_command_buffer = begin_recording_impl();
}

GpuCommandBufferHandle TrackedInferSubmissionSession::current_command_buffer() const {
    OPENVINO_ASSERT(m_current_command_buffer, "GFX: submission command buffer is not initialized");
    return m_current_command_buffer;
}

void TrackedInferSubmissionSession::submit_recorded(bool continue_recording) {
    OPENVINO_ASSERT(m_current_command_buffer, "GFX: submission command buffer is not initialized");
    submit_recorded_impl(m_current_command_buffer, continue_recording);
    m_completed_command_buffer = m_current_command_buffer;
    m_current_command_buffer = continue_recording ? begin_recording_impl() : nullptr;
}

void TrackedInferSubmissionSession::finish() {
    finish_impl();
}

GpuCommandBufferHandle TrackedInferSubmissionSession::completed_command_buffer() const {
    return m_completed_command_buffer;
}

void execute_pipeline_with_submission(
    std::vector<InferStage>& pipeline,
    const std::unordered_map<const ov::Node*, size_t>& node_map,
    const std::unordered_map<const ov::Node*, size_t>& param_map,
    const InferInputLookup& input_lookup,
    InferSubmissionSession& submission,
    const InferSubmissionConfig& config,
    GfxProfiler* profiler,
    PreparedInferExecutionPlan* prepared_plan,
    const InferStageHook& on_stage,
    const InferSubmissionFinalRecordHook& on_before_final_submit,
    bool recording_started) {
    if (!recording_started) {
        submission.begin_recording();
    }

    size_t recorded_stage_count = 0;
    size_t recorded_output_bytes = 0;
    uint64_t recorded_macs = 0;
    uint64_t recorded_flops = 0;
    std::vector<InferStage*> recorded_stages;
    std::vector<bool> recorded_stage_mask(pipeline.size(), false);
    const auto dependency_graph = build_submission_dependency_graph(pipeline);
    size_t submission_window_index = 0;
    auto would_exceed_size_budget = [](size_t current, size_t addition, size_t limit) {
        return current >= limit || addition > limit - current;
    };
    auto would_exceed_u64_budget = [](uint64_t current, uint64_t addition, uint64_t limit) {
        return current >= limit || addition > limit - current;
    };
    auto describe_recorded_stages = [&]() {
        std::ostringstream oss;
        for (size_t index = 0; index < recorded_stages.size(); ++index) {
            if (index != 0) {
                oss << " | ";
            }
            const auto* infer_stage = recorded_stages[index];
            const auto* gpu_stage = infer_stage ? infer_stage->stage.get() : nullptr;
            oss << (gpu_stage ? gpu_stage->name() : "<null>")
                << " ["
                << (gpu_stage ? gpu_stage->type() : "unknown")
                << "]";
        }
        return oss.str();
    };
    auto flush_submission = [&](bool continue_recording) {
        if (recorded_stage_count == 0) {
            return;
        }
        const bool needs_incremental_submit = continue_recording;
        if (needs_incremental_submit &&
            (!config.allow_incremental_submit || !submission.supports_incremental_submit())) {
            return;
        }
        if (gfx_log_debug_enabled()) {
            gfx_log_debug("InferSubmit") << "Submitting window #" << submission_window_index
                                         << " continue=" << (continue_recording ? "yes" : "no")
                                         << " weighted_stages=" << recorded_stage_count
                                         << " output_bytes=" << recorded_output_bytes
                                         << " macs=" << recorded_macs
                                         << " stages=" << describe_recorded_stages();
        }
        const bool profiling = (profiler != nullptr);
        const auto submit_start = profiling ? std::chrono::steady_clock::now()
                                            : std::chrono::steady_clock::time_point{};
        submission.submit_recorded(continue_recording);
        if (profiling) {
            const auto submit_cpu_us =
                std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::steady_clock::now() - submit_start);
            profiler->increment_counter("submit_count");
            profiler->set_counter("submission_window_count", submission_window_index + 1u);
            profiler->record_segment("submit",
                                     "window#" + std::to_string(submission_window_index),
                                     submit_cpu_us,
                                     0,
                                     0,
                                     0,
                                     recorded_output_bytes,
                                     recorded_macs,
                                     recorded_flops);
            if (continue_recording) {
                profiler->increment_counter("incremental_submit_count");
            }
        }
        notify_pipeline_submission_complete(recorded_stages);
        ++submission_window_index;
        recorded_stage_count = 0;
        recorded_output_bytes = 0;
        recorded_macs = 0;
        recorded_flops = 0;
        recorded_stages.clear();
        std::fill(recorded_stage_mask.begin(), recorded_stage_mask.end(), false);
    };

    execute_pipeline(
        pipeline,
        node_map,
        param_map,
        input_lookup,
        [&](InferStage& stage, const std::vector<GpuTensor*>& resolved) {
            const size_t stage_index = static_cast<size_t>(&stage - pipeline.data());
            const auto policy = stage.stage->submit_policy();
            const size_t stage_weight = std::max<size_t>(policy.weight, 1);
            if (policy.isolate && recorded_stage_count > 0) {
                flush_submission(true);
            }

            const std::string& stage_type = stage.stage->type();
            const auto estimate = estimate_stage_profile(stage, resolved);
            const bool exceeds_soft_budget =
                recorded_stage_count > 0 &&
                (would_exceed_size_budget(recorded_stage_count,
                                          stage_weight,
                                          config.max_stages_per_submit) ||
                 would_exceed_size_budget(recorded_output_bytes,
                                          stage_output_bytes_for_budget(estimate.bytes_out),
                                          config.max_output_bytes_per_submit) ||
                 would_exceed_u64_budget(recorded_macs, estimate.macs, config.max_macs_per_submit));
            if (exceeds_soft_budget) {
                const auto totals = extended_window_totals(recorded_stage_count,
                                                           recorded_output_bytes,
                                                           recorded_macs,
                                                           stage_weight,
                                                           estimate);
                const bool keep_direct_dependency =
                    stage_depends_on_recording_window(stage_index, dependency_graph, recorded_stage_mask) &&
                    !dependency_extension_crosses_boundary(stage_index, dependency_graph, recorded_stage_mask, pipeline) &&
                    within_dependency_extension_cap(totals, config);
                if (keep_direct_dependency) {
                    if (profiler) {
                        profiler->increment_counter("submission_dependency_window_extension_count");
                    }
                } else {
                    flush_submission(true);
                }
            }

            const auto command_buffer = submission.current_command_buffer();
            if (gfx_log_debug_enabled()) {
                gfx_log_debug("InferSubmit") << "Record stage " << stage.stage->name()
                                             << " [" << stage_type << "]"
                                             << " weight=" << stage_weight
                                             << " isolate=" << (policy.isolate ? "yes" : "no")
                                             << " current_window=" << submission_window_index;
            }
            if (on_stage) {
                const auto stage_start = profiler ? std::chrono::steady_clock::now()
                                                  : std::chrono::steady_clock::time_point{};
                on_stage(stage, resolved, command_buffer);
                if (profiler) {
                    const auto stage_cpu_us = std::chrono::duration_cast<std::chrono::microseconds>(
                        std::chrono::steady_clock::now() - stage_start);
                    const auto* gpu_stage = stage.stage.get();
                    std::string segment_name = gpu_stage ? gpu_stage->type() : std::string{"unknown"};
                    segment_name += ":";
                    segment_name += gpu_stage ? gpu_stage->name() : std::string{"<null>"};
                    profiler->record_segment("stage_execute",
                                             segment_name,
                                             stage_cpu_us,
                                             0,
                                             0,
                                             estimate.bytes_in,
                                             estimate.bytes_out,
                                             estimate.macs,
                                             estimate.flops,
                                             -1,
                                             0,
                                             reinterpret_cast<uint64_t>(command_buffer));
                }
            } else {
                const auto stage_start = profiler ? std::chrono::steady_clock::now()
                                                  : std::chrono::steady_clock::time_point{};
                stage.stage->execute(command_buffer);
                if (profiler) {
                    const auto stage_cpu_us = std::chrono::duration_cast<std::chrono::microseconds>(
                        std::chrono::steady_clock::now() - stage_start);
                    const auto* gpu_stage = stage.stage.get();
                    std::string segment_name = gpu_stage ? gpu_stage->type() : std::string{"unknown"};
                    segment_name += ":";
                    segment_name += gpu_stage ? gpu_stage->name() : std::string{"<null>"};
                    profiler->record_segment("stage_execute",
                                             segment_name,
                                             stage_cpu_us,
                                             0,
                                             0,
                                             estimate.bytes_in,
                                             estimate.bytes_out,
                                             estimate.macs,
                                             estimate.flops,
                                             -1,
                                             0,
                                             reinterpret_cast<uint64_t>(command_buffer));
                }
            }
            recorded_stage_count = add_saturating_size(recorded_stage_count, stage_weight);
            add_saturating(recorded_macs, estimate.macs);
            add_saturating(recorded_flops, estimate.flops);
            recorded_stages.push_back(&stage);
            if (stage_index < recorded_stage_mask.size()) {
                recorded_stage_mask[stage_index] = true;
            }
            for (const auto& out : stage.outputs) {
                if (out && out->buf.valid()) {
                    recorded_output_bytes = add_saturating_size(recorded_output_bytes, out->buf.size);
                }
            }

            // "isolate" means "start this stage in a fresh submission window", not
            // "force this stage to be the only stage in that window". Keeping the
            // stage and its immediate epilogue/consumer chain in the same command
            // buffer avoids pathological submit-wait-submit fragmentation on deep
            // mobile Vulkan graphs while still respecting the weighted window budget.
            const bool window_reached_soft_budget =
                recorded_stage_count >= config.max_stages_per_submit ||
                recorded_output_bytes >= config.max_output_bytes_per_submit ||
                recorded_macs >= config.max_macs_per_submit;
            if (window_reached_soft_budget) {
                bool hold_for_direct_consumer = false;
                const size_t next_index = next_executable_stage_index(stage_index, pipeline);
                if (next_index != std::numeric_limits<size_t>::max()) {
                    if (stage_depends_on_recording_window(next_index, dependency_graph, recorded_stage_mask)) {
                        const auto& next_stage = pipeline[next_index];
                        const auto next_policy = next_stage.stage->submit_policy();
                        if (!next_policy.isolate &&
                            !dependency_extension_crosses_boundary(next_index,
                                                                   dependency_graph,
                                                                   recorded_stage_mask,
                                                                   pipeline)) {
                            const auto next_estimate = estimate_stage_profile(next_stage, {});
                            const auto totals = extended_window_totals(recorded_stage_count,
                                                                       recorded_output_bytes,
                                                                       recorded_macs,
                                                                       std::max<size_t>(next_policy.weight, 1),
                                                                       next_estimate);
                            hold_for_direct_consumer = within_dependency_extension_cap(totals, config);
                        }
                    }
                    if (hold_for_direct_consumer) {
                        if (profiler) {
                            profiler->increment_counter("submission_dependency_window_hold_count");
                        }
                    } else {
                        flush_submission(true);
                    }
                }
            }
        },
        prepared_plan);

    if (on_before_final_submit) {
        const auto extra_work = on_before_final_submit(submission.current_command_buffer());
        recorded_stage_count = add_saturating_size(recorded_stage_count, extra_work.weight);
        recorded_output_bytes = add_saturating_size(recorded_output_bytes, extra_work.output_bytes);
    }

    flush_submission(false);
    submission.finish();
}

void notify_pipeline_submission_complete(std::vector<InferStage*>& stages) {
    for (auto* stage : stages) {
        if (stage && stage->stage) {
            stage->stage->on_command_buffer_complete();
        }
    }
}

}  // namespace gfx_plugin
}  // namespace ov
