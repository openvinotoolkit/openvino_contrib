// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "mlir/gfx_mlir_kernel_builder.hpp"

#include <vector>

#include "mlir/mlir_builder.hpp"
#include "runtime/gfx_op_utils.hpp"

#include "mlir/Dialect/GPU/IR/GPUDialect.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "openvino/core/except.hpp"
#include "openvino/op/clamp.hpp"
#include "openvino/op/elu.hpp"

namespace ov {
namespace gfx_plugin {

namespace {

using BuilderFn = mlir::ModuleOp (*)(const std::shared_ptr<const ov::Model>&, mlir::MLIRContext&);

struct BuilderEntry {
    const char* type_name;
    BuilderFn builder;
    bool multi_output;
};

const std::vector<BuilderEntry>& builder_registry() {
    static const std::vector<BuilderEntry> entries = {
        {"Add", build_mlir_add_from_model, false},
        {"Subtract", build_mlir_sub_from_model, false},
        {"Multiply", build_mlir_mul_from_model, false},
        {"Divide", build_mlir_div_from_model, false},
        {"Power", build_mlir_pow_from_model, false},
        {"Mod", build_mlir_mod_from_model, false},
        {"FloorMod", build_mlir_floor_mod_from_model, false},
        {"PRelu", build_mlir_prelu_from_model, false},
        {"Minimum", build_mlir_min_from_model, false},
        {"Maximum", build_mlir_max_from_model, false},
        {"LogicalAnd", build_mlir_logical_and_from_model, false},
        {"LogicalOr", build_mlir_logical_or_from_model, false},
        {"LogicalXor", build_mlir_logical_xor_from_model, false},
        {"Equal", build_mlir_equal_from_model, false},
        {"NotEqual", build_mlir_not_equal_from_model, false},
        {"Less", build_mlir_less_from_model, false},
        {"Greater", build_mlir_greater_from_model, false},
        {"LessEqual", build_mlir_less_equal_from_model, false},
        {"GreaterEqual", build_mlir_greater_equal_from_model, false},
        {"SquaredDifference", build_mlir_squared_difference_from_model, false},
        {"Softmax", build_mlir_softmax_from_model, false},
        {"LogSoftmax", build_mlir_logsoftmax_from_model, false},
        {"MaxPool", build_mlir_maxpool_from_model, false},
        {"AvgPool", build_mlir_avgpool_from_model, false},
        {"Convolution", build_mlir_conv2d_from_model, false},
        {"GroupConvolution", build_mlir_group_conv2d_from_model, false},
        {"BatchNormInference", build_mlir_batchnorm_from_model, false},
        {"Convert", build_mlir_convert_from_model, false},
        {"Transpose", build_mlir_transpose_from_model, false},
        {"Slice", build_mlir_slice_from_model, false},
        {"StridedSlice", build_mlir_slice_from_model, false},
        {"Concat", build_mlir_concat_from_model, false},
        {"Split", build_mlir_split_from_model, true},
        {"VariadicSplit", build_mlir_split_from_model, true},
        {"Interpolate", build_mlir_interpolate_from_model, false},
        {"Gather", build_mlir_gather_from_model, false},
        {"GatherND", build_mlir_gathernd_from_model, false},
        {"GatherElements", build_mlir_gather_elements_from_model, false},
        {"DepthToSpace", build_mlir_depth_to_space_from_model, false},
        {"SpaceToDepth", build_mlir_space_to_depth_from_model, false},
        {"ScatterElementsUpdate", build_mlir_scatter_elements_update_from_model, false},
        {"ScatterNDUpdate", build_mlir_scatter_nd_update_from_model, false},
        {"ShapeOf", build_mlir_shapeof_from_model, false},
        {"Select", build_mlir_select_from_model, false},
        {"ReduceSum", build_mlir_reducesum_from_model, false},
        {"ReduceMean", build_mlir_reducemean_from_model, false},
        {"ReduceMax", build_mlir_reducemax_from_model, false},
        {"ReduceMin", build_mlir_reducemin_from_model, false},
        {"ReduceProd", build_mlir_reduceprod_from_model, false},
        {"ReduceL1", build_mlir_reducel1_from_model, false},
        {"ReduceL2", build_mlir_reducel2_from_model, false},
        {"Pad", build_mlir_pad_from_model, false},
        {"Tile", build_mlir_tile_from_model, false},
        {"Broadcast", build_mlir_broadcast_from_model, false},
        {"Range", build_mlir_range_from_model, false},
        {"TopK", build_mlir_topk_from_model, true},
        {"Reverse", build_mlir_reverse_from_model, false},
        {"MatMul", build_mlir_module_from_model, false},
    };
    return entries;
}

}  // namespace

mlir::ModuleOp build_mlir_for_node(const std::shared_ptr<const ov::Node>& node,
                                   mlir::MLIRContext& ctx) {
    const std::string type = node->get_type_name();
    if (type == "Convolution") {
        const auto& pshape = node->get_input_partial_shape(0);
        if (pshape.rank().is_static() && pshape.rank().get_length() == 5) {
            return build_mlir_conv3d_from_model(make_single_op_model(node), ctx);
        }
        return build_mlir_conv2d_from_model(make_single_op_model(node), ctx);
    }
    if (type == "GroupConvolution") {
        const auto& pshape = node->get_input_partial_shape(0);
        if (pshape.rank().is_static() && pshape.rank().get_length() == 5) {
            OPENVINO_THROW("GFX MLIR: GroupConvolution 3D is not supported yet");
        }
    }
    for (const auto& entry : builder_registry()) {
        if (type == entry.type_name) {
            auto model = entry.multi_output ? make_single_op_model_all_outputs(node)
                                            : make_single_op_model(node);
            return entry.builder(model, ctx);
        }
    }
    struct UnaryEntry {
        const char* name;
        ActivationKind kind;
    };
    static const std::vector<UnaryEntry> unary = {
        {"Relu", ActivationKind::Relu},
        {"Sigmoid", ActivationKind::Sigmoid},
        {"Tanh", ActivationKind::Tanh},
        {"Elu", ActivationKind::Elu},
        {"Gelu", ActivationKind::Gelu},
        {"Swish", ActivationKind::Swish},
        {"HSwish", ActivationKind::HSwish},
        {"HSigmoid", ActivationKind::HSigmoid},
        {"SoftPlus", ActivationKind::SoftPlus},
        {"Mish", ActivationKind::Mish},
        {"SoftSign", ActivationKind::SoftSign},
        {"Abs", ActivationKind::Abs},
        {"Sign", ActivationKind::Sign},
        {"Clamp", ActivationKind::Clamp},
        {"LogicalNot", ActivationKind::LogicalNot},
        {"Exp", ActivationKind::Exp},
        {"Log", ActivationKind::Log},
        {"Sqrt", ActivationKind::Sqrt},
        {"Floor", ActivationKind::Floor},
        {"Ceiling", ActivationKind::Ceil},
        {"Negative", ActivationKind::Negative},
        {"Sin", ActivationKind::Sin},
        {"Cos", ActivationKind::Cos},
        {"Tan", ActivationKind::Tan},
        {"Erf", ActivationKind::Erf},
        {"Asin", ActivationKind::Asin},
        {"Acos", ActivationKind::Acos},
        {"Atan", ActivationKind::Atan},
        {"Asinh", ActivationKind::Asinh},
        {"Acosh", ActivationKind::Acosh},
        {"Atanh", ActivationKind::Atanh},
        {"Sinh", ActivationKind::Sinh},
        {"Cosh", ActivationKind::Cosh},
        {"Round", ActivationKind::RoundAway},
    };
    for (const auto& entry : unary) {
        if (type == entry.name) {
            std::optional<std::pair<double, double>> clamp_range;
            float alpha = 1.0f;
            if (entry.kind == ActivationKind::Clamp) {
                if (auto clamp = ov::as_type_ptr<const ov::op::v0::Clamp>(node)) {
                    clamp_range = std::make_pair(clamp->get_min(), clamp->get_max());
                }
            }
            if (entry.kind == ActivationKind::Elu) {
                if (auto elu = ov::as_type_ptr<const ov::op::v0::Elu>(node)) {
                    alpha = static_cast<float>(elu->get_alpha());
                }
            }
            return build_mlir_unary_from_node(node, ctx, entry.kind, alpha, clamp_range);
        }
    }
    OPENVINO_THROW("GFX MLIR: unsupported op for MLIR lowering: ", type);
}

std::string find_entry_point(mlir::ModuleOp module) {
    std::string name;
    module.walk([&](mlir::gpu::GPUFuncOp func) {
        if (name.empty()) {
            name = func.getName().str();
        }
    });
    module.walk([&](mlir::func::FuncOp func) {
        if (name.empty()) {
            name = func.getName().str();
        }
    });
    return name;
}

std::string resolve_entry_point(mlir::ModuleOp module,
                                const std::string& hint,
                                std::string_view fallback) {
    if (!hint.empty()) {
        return hint;
    }
    if (module) {
        std::string entry = find_entry_point(module);
        if (!entry.empty()) {
            return entry;
        }
    }
    return std::string(fallback);
}

KernelPlan MlirKernelPlanBuilder::build_plan(const std::shared_ptr<const ov::Node>& node,
                                             mlir::MLIRContext& ctx,
                                             uint32_t arg_count) const {
    auto module = build_mlir_for_node(node, ctx);
    std::string entry = resolve_entry_point(module, {}, "gfx_kernel");
    return KernelPlan(module, std::move(entry), arg_count);
}

KernelPlan MlirKernelPlanBuilder::build_plan(const KernelSpec& spec, mlir::MLIRContext& ctx) const {
    return build_plan(spec.node(), ctx, spec.arg_count());
}

}  // namespace gfx_plugin
}  // namespace ov
