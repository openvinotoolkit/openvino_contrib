// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//
#pragma once

#include <cstdint>
#include <string>
#include <string_view>
#include <vector>

#include "compiler/lowering_planner.hpp"

namespace ov {
namespace gfx_plugin {
namespace compiler {

enum class TensorContractRole {
    TensorInput,
    TensorOutput,
};

enum class RuntimeParamKind {
    Scalar,
    Shape,
};

struct TensorContract {
    std::string logical_name;
    TensorContractRole role = TensorContractRole::TensorInput;
    std::string element_type;
    std::string partial_shape;
    std::string layout = "logical";
    std::string storage_kind = "device_buffer";
    std::string lifetime_class;
};

struct RuntimeParamDescriptor {
    std::string logical_name;
    RuntimeParamKind kind = RuntimeParamKind::Shape;
    std::string abi_type;
    std::string source_tensor;
};

struct RuntimeParamContract {
    size_t scalar_param_count = 0;
    size_t shape_param_count = 0;
    std::vector<RuntimeParamDescriptor> params;
    std::vector<std::string> runtime_param_names;
};

struct DispatchContract {
    LoweringRouteKind execution_kind = LoweringRouteKind::Unsupported;
    std::string backend_domain;
    std::string kernel_unit_id;
    std::string kernel_unit_kind;
    std::string dispatch_source = "lowering_plan";
};

struct MemoryContract {
    bool hidden_host_copy_allowed = false;
    std::string input_lifetime = "producer_or_external";
    std::string output_lifetime = "stage_output";
    std::string alias_group;
};

struct StageRecord {
    size_t stage_id = 0;
    uint64_t stable_record_key = 0;
    std::string source_node_name;
    std::string normalized_op_family;
    LoweringRouteKind execution_kind = LoweringRouteKind::Unsupported;
    std::string backend_domain;
    std::string kernel_unit_id;
    std::string kernel_unit_kind;
    std::vector<TensorContract> inputs;
    std::vector<TensorContract> outputs;
    RuntimeParamContract runtime_params;
    DispatchContract dispatch;
    MemoryContract memory;
    double profitability_score = 0.0;
};

struct ManifestVerificationResult {
    std::vector<std::string> diagnostics;

    bool valid() const noexcept {
        return diagnostics.empty();
    }
};

struct ManifestBundle {
    uint32_t schema_version = 2;
    std::string target_fingerprint;
    std::vector<StageRecord> stages;

    ManifestVerificationResult verify() const;
    bool valid() const;
    size_t route_count(LoweringRouteKind route_kind) const;
};

class ManifestBuilder final {
public:
    ManifestBundle build(const LoweringPlan& plan) const;
};

std::string_view tensor_contract_role_to_string(TensorContractRole role) noexcept;
std::string_view runtime_param_kind_to_string(RuntimeParamKind kind) noexcept;

}  // namespace compiler
}  // namespace gfx_plugin
}  // namespace ov
