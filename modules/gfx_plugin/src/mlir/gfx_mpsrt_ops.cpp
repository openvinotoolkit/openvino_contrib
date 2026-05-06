// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "mlir/gfx_mpsrt_ops.hpp"

#include "mlir/gfx_mpsrt_dialect.hpp"
#include "mlir/gfx_mpsrt_metadata.hpp"

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"

#include <string>
#include <utility>
#include <vector>

namespace ov {
namespace gfx_plugin {
namespace {

constexpr const char* kGfxMpsrtReturnOpName = "gfx.mpsrt.return";

std::string mpsrt_op_name_for_stage(GfxMpsrtStageKind kind) {
    switch (kind) {
        case GfxMpsrtStageKind::MPSConv2D:
            return "gfx.mpsrt.conv2d";
        case GfxMpsrtStageKind::MPSGroupConv2D:
            return "gfx.mpsrt.group_conv2d";
        case GfxMpsrtStageKind::MPSPool2D:
            return "gfx.mpsrt.pool2d";
        case GfxMpsrtStageKind::MPSGemm:
            return "gfx.mpsrt.gemm";
        case GfxMpsrtStageKind::MPSSoftmax:
            return "gfx.mpsrt.softmax";
        case GfxMpsrtStageKind::MPSTopK:
            return "gfx.mpsrt.topk";
        case GfxMpsrtStageKind::MSLDispatch:
            return "gfx.mpsrt.dispatch";
        case GfxMpsrtStageKind::Unknown:
        default:
            return {};
    }
}

std::string mpsrt_op_name_for_storage_bridge(GfxMpsrtStorageBridgeDirection direction) {
    switch (direction) {
        case GfxMpsrtStorageBridgeDirection::BufferToImage:
            return "gfx.mpsrt.to_image";
        case GfxMpsrtStorageBridgeDirection::BufferToMatrix:
            return "gfx.mpsrt.to_matrix";
        case GfxMpsrtStorageBridgeDirection::BufferToNDArray:
            return "gfx.mpsrt.to_ndarray";
        case GfxMpsrtStorageBridgeDirection::ImageToBuffer:
        case GfxMpsrtStorageBridgeDirection::MatrixToBuffer:
        case GfxMpsrtStorageBridgeDirection::NDArrayToBuffer:
            return "gfx.mpsrt.to_buffer";
        case GfxMpsrtStorageBridgeDirection::Alias:
            return "gfx.mpsrt.alias";
        case GfxMpsrtStorageBridgeDirection::Unknown:
        default:
            return {};
    }
}

std::string mpsrt_stage_op_name(const GfxMpsrtStageDesc& stage) {
    auto name = mpsrt_op_name_for_stage(stage.kind);
    if (name.empty() || stage.builder_symbol.empty()) {
        return {};
    }
    return name;
}

mlir::func::FuncOp lookup_generated_ops_func(mlir::ModuleOp module) {
    if (!module) {
        return {};
    }
    std::string symbol = kGfxMpsrtOpsSymbol;
    if (auto attr = module->getAttrOfType<mlir::StringAttr>("gfx.mpsrt.ops.symbol")) {
        symbol = attr.str();
    }
    auto ops_func = module.lookupSymbol<mlir::func::FuncOp>(symbol);
    if (!ops_func ||
        !ops_func->getAttrOfType<mlir::BoolAttr>("gfx.mpsrt.ops.generated")) {
        return {};
    }
    return ops_func;
}

void set_op_tensor_outputs(mlir::Operation* op,
                           mlir::Builder& builder,
                           const std::vector<GfxMpsrtTensorDesc>& outputs) {
    op->setAttr("gfx.mpsrt.op.output_count",
                builder.getI32IntegerAttr(static_cast<int32_t>(outputs.size())));
    for (size_t i = 0; i < outputs.size(); ++i) {
        detail::gfx_mpsrt_set_tensor_desc_attrs(op,
                                                "gfx.mpsrt.op.output" + std::to_string(i),
                                                outputs[i]);
    }
}

bool read_op_tensor_outputs(mlir::Operation* op,
                            std::vector<GfxMpsrtTensorDesc>& outputs) {
    outputs.clear();
    uint32_t output_count = 0;
    if (!detail::gfx_mpsrt_read_i32_attr(op, "gfx.mpsrt.op.output_count", output_count)) {
        return false;
    }
    outputs.reserve(output_count);
    for (uint32_t i = 0; i < output_count; ++i) {
        GfxMpsrtTensorDesc desc{};
        if (!detail::gfx_mpsrt_read_tensor_desc_attrs(op,
                                                      "gfx.mpsrt.op.output" + std::to_string(i),
                                                      desc)) {
            outputs.clear();
            return false;
        }
        outputs.push_back(desc);
    }
    return true;
}

void set_program_common_attrs(mlir::Operation* op,
                              mlir::Builder& builder,
                              const GfxMpsrtProgram& program) {
    op->setAttr("gfx.mpsrt.ops.version", builder.getI32IntegerAttr(1));
    op->setAttr("gfx.mpsrt.ops.kind",
                builder.getStringAttr(program.multi_stage ? "multi_stage" : "single_stage"));
    op->setAttr("gfx.mpsrt.ops.record_key", builder.getStringAttr(program.record_key));
    op->setAttr("gfx.mpsrt.ops.input_count",
                builder.getI32IntegerAttr(static_cast<int32_t>(program.inputs.size())));
    op->setAttr("gfx.mpsrt.ops.stage_count",
                builder.getI32IntegerAttr(static_cast<int32_t>(program.stages.size())));
    op->setAttr("gfx.mpsrt.ops.output_values",
                detail::gfx_mpsrt_u32_vector_attr(builder, program.output_values));
    for (size_t i = 0; i < program.inputs.size(); ++i) {
        detail::gfx_mpsrt_set_tensor_desc_attrs(op,
                                                "gfx.mpsrt.ops.input" + std::to_string(i),
                                                program.inputs[i]);
    }
    if (program.external_buffer_abi.valid) {
        if (program.external_buffer_abi.has_buffer_count) {
            op->setAttr("gfx.mpsrt.ops.external_buffer_count",
                        builder.getI32IntegerAttr(
                            static_cast<int32_t>(program.external_buffer_abi.buffer_count)));
        }
        if (program.external_buffer_abi.has_output_buffer_count) {
            op->setAttr("gfx.mpsrt.ops.external_output_buffer_count",
                        builder.getI32IntegerAttr(
                            static_cast<int32_t>(program.external_buffer_abi.output_buffer_count)));
        }
        if (program.external_buffer_abi.has_buffer_roles) {
            detail::gfx_mpsrt_set_external_buffer_roles_attrs(op,
                                                              "gfx.mpsrt.ops",
                                                              program.external_buffer_abi.buffer_roles);
        }
    }
    if (program.has_storage_bridges) {
        detail::gfx_mpsrt_set_storage_bridges_attrs(op,
                                                    "gfx.mpsrt.ops",
                                                    program.storage_bridges);
    }
}

bool read_program_common_attrs(mlir::Operation* op,
                               GfxMpsrtProgram& out,
                               uint32_t& stage_count) {
    std::string kind;
    uint32_t input_count = 0;
    if (!detail::gfx_mpsrt_read_string_attr(op, "gfx.mpsrt.ops.kind", kind) ||
        !detail::gfx_mpsrt_read_string_attr(op, "gfx.mpsrt.ops.record_key", out.record_key) ||
        !detail::gfx_mpsrt_read_i32_attr(op, "gfx.mpsrt.ops.input_count", input_count) ||
        !detail::gfx_mpsrt_read_i32_attr(op, "gfx.mpsrt.ops.stage_count", stage_count)) {
        return false;
    }
    out.multi_stage = kind == "multi_stage";
    out.inputs.reserve(input_count);
    for (uint32_t i = 0; i < input_count; ++i) {
        GfxMpsrtTensorDesc desc{};
        if (!detail::gfx_mpsrt_read_tensor_desc_attrs(op,
                                                      "gfx.mpsrt.ops.input" + std::to_string(i),
                                                      desc)) {
            return false;
        }
        out.inputs.push_back(desc);
    }
    out.output_values = detail::gfx_mpsrt_read_u32_vector_attr(op, "gfx.mpsrt.ops.output_values");

    GfxMpsrtExternalBufferAbiPlan external_buffer_abi{};
    external_buffer_abi.has_buffer_count =
        detail::gfx_mpsrt_read_i32_attr(op,
                                        "gfx.mpsrt.ops.external_buffer_count",
                                        external_buffer_abi.buffer_count);
    external_buffer_abi.has_output_buffer_count =
        detail::gfx_mpsrt_read_i32_attr(op,
                                        "gfx.mpsrt.ops.external_output_buffer_count",
                                        external_buffer_abi.output_buffer_count);
    const auto role_values =
        detail::gfx_mpsrt_read_u32_vector_attr(op, "gfx.mpsrt.ops.external_buffer_roles");
    if (!role_values.empty()) {
        external_buffer_abi.has_buffer_roles = true;
        external_buffer_abi.buffer_roles.reserve(role_values.size());
        for (const auto role_value : role_values) {
            external_buffer_abi.buffer_roles.push_back(static_cast<GfxMpsrtExternalBufferRole>(role_value));
        }
    }
    if (external_buffer_abi.has_buffer_count ||
        external_buffer_abi.has_output_buffer_count ||
        external_buffer_abi.has_buffer_roles) {
        if (!gfx_mpsrt_finalize_external_buffer_abi(external_buffer_abi)) {
            return false;
        }
        out.external_buffer_abi = std::move(external_buffer_abi);
    }
    if (detail::gfx_mpsrt_read_storage_bridges_attrs(op,
                                                     "gfx.mpsrt.ops",
                                                     out.storage_bridges)) {
        out.has_storage_bridges = true;
    }
    return true;
}

bool read_u32_op_attr(mlir::Operation* op,
                      const std::string& name,
                      uint32_t& value) {
    return detail::gfx_mpsrt_read_i32_attr(op, name, value);
}

mlir::Operation* create_mpsrt_op(mlir::OpBuilder& builder,
                                 mlir::Location loc,
                                 llvm::StringRef name) {
    mlir::OperationState state(loc, name);
    return builder.create(state);
}

bool create_mpsrt_storage_bridge_op(mlir::OpBuilder& builder,
                                    mlir::Location loc,
                                    const GfxMpsrtStorageBridgeDesc& bridge) {
    const auto op_name = mpsrt_op_name_for_storage_bridge(bridge.direction);
    if (op_name.empty()) {
        return false;
    }
    auto* op = create_mpsrt_op(builder, loc, op_name);
    op->setAttr("gfx.mpsrt.storage_bridge.generated", builder.getBoolAttr(true));
    detail::gfx_mpsrt_set_storage_bridge_attrs(op, "gfx.mpsrt.storage_bridge", bridge);
    return true;
}

bool create_mpsrt_storage_bridge_ops(mlir::OpBuilder& builder,
                                     mlir::Location loc,
                                     const std::vector<GfxMpsrtStorageBridgeDesc>& bridges,
    bool external_inputs) {
    for (const auto& bridge : bridges) {
        const bool is_input_bridge =
            gfx_mpsrt_storage_bridge_reads_external_buffer(bridge.direction);
        if (is_input_bridge != external_inputs) {
            continue;
        }
        if (!create_mpsrt_storage_bridge_op(builder, loc, bridge)) {
            return false;
        }
    }
    return true;
}

void erase_generated_legacy_program_facade(mlir::ModuleOp module) {
    if (!module) {
        return;
    }
    if (auto facade = module.lookupSymbol<mlir::func::FuncOp>("gfx_mpsrt_program")) {
        if (facade->getAttrOfType<mlir::BoolAttr>("gfx.mpsrt.program.generated")) {
            facade.erase();
        }
    }
    module->removeAttr("gfx.mpsrt.program.symbol");
}

}  // namespace

void erase_module_mpsrt_ops(mlir::ModuleOp module) {
    if (!module) {
        return;
    }
    if (auto ops_func = lookup_generated_ops_func(module)) {
        ops_func.erase();
    }
    module->removeAttr("gfx.mpsrt.ops.symbol");
}

void erase_module_mpsrt_legacy_attrs(mlir::ModuleOp module) {
    if (!module) {
        return;
    }

    // These module-level placement/MSL fields are a transitional adapter surface.
    // Once a generated gfx.mpsrt program exists, the canonical contract lives
    // on the generated stage op via gfx.stage_manifest and typed stage attrs.
    for (const char* name : {
             "gfx.backend",
             "gfx.storage",
             "gfx.stage_type",
             "gfx.uses_vendor_primitive",
             "gfx.uses_custom_kernel",
             "gfx.specialization_key",
         }) {
        module->removeAttr(name);
    }

    std::vector<std::string> attrs_to_remove;
    for (const auto& named_attr : module->getAttrs()) {
        const auto name = named_attr.getName().strref();
        if (!name.starts_with("gfx.mpsrt.")) {
            if (!name.starts_with("gfx.msl.")) {
                continue;
            }
        }
        if (name.starts_with("gfx.mpsrt.ops.") ||
            name.starts_with("gfx.mpsrt.runtime_abi.")) {
            continue;
        }
        attrs_to_remove.push_back(name.str());
    }
    for (const auto& name : attrs_to_remove) {
        module->removeAttr(name);
    }
    erase_generated_legacy_program_facade(module);
}

bool materialize_module_mpsrt_ops(mlir::ModuleOp module,
                                  const GfxMpsrtProgram& program) {
    if (!module || !gfx_mpsrt_validate_program(program, nullptr)) {
        return false;
    }
    auto* context = module->getContext();
    mpsrt::register_gfx_mpsrt_dialect(*context);
    context->loadDialect<mlir::func::FuncDialect>();

    erase_module_mpsrt_ops(module);
    if (module.lookupSymbol<mlir::func::FuncOp>(kGfxMpsrtOpsSymbol)) {
        return false;
    }

    mlir::OpBuilder builder(context);
    const auto loc = mlir::UnknownLoc::get(context);
    const auto empty_func_type = builder.getFunctionType({}, {});

    builder.setInsertionPointToEnd(module.getBody());
    auto ops_func = mlir::func::FuncOp::create(builder,
                                               loc,
                                               kGfxMpsrtOpsSymbol,
                                               empty_func_type);
    ops_func.setSymVisibility("private");
    ops_func->setAttr("gfx.mpsrt.ops.generated", builder.getBoolAttr(true));
    set_program_common_attrs(ops_func.getOperation(), builder, program);
    ops_func.addEntryBlock();

    mlir::OpBuilder body_builder(ops_func.getBody());
    body_builder.setInsertionPointToEnd(&ops_func.getBody().front());
    if (program.has_storage_bridges &&
        !create_mpsrt_storage_bridge_ops(body_builder,
                                         loc,
                                         program.storage_bridges,
                                         /*external_inputs=*/true)) {
        return false;
    }
    for (size_t i = 0; i < program.stages.size(); ++i) {
        const auto& spec = program.stages[i];
        const auto op_name = mpsrt_op_name_for_stage(spec.stage.kind);
        if (op_name.empty() || mpsrt_stage_op_name(spec.stage).empty()) {
            return false;
        }
        auto* op = create_mpsrt_op(body_builder, loc, op_name);
        op->setAttr("gfx.mpsrt.op.generated", body_builder.getBoolAttr(true));
        op->setAttr("gfx.mpsrt.op.stage_index",
                    body_builder.getI32IntegerAttr(static_cast<int32_t>(i)));
        detail::gfx_mpsrt_set_stage_desc_attrs(op,
                                               "gfx.mpsrt.op.stage",
                                               spec.stage,
                                               spec.stage_record_key);
        detail::gfx_mpsrt_set_stage_manifest_attrs(op, spec.stage.stage_manifest);
        op->setAttr("gfx.mpsrt.op.input_values",
                    detail::gfx_mpsrt_u32_vector_attr(body_builder, spec.inputs));
        op->setAttr("gfx.mpsrt.op.output_values",
                    detail::gfx_mpsrt_u32_vector_attr(body_builder, spec.outputs));
        set_op_tensor_outputs(op, body_builder, spec.output_descs);
    }
    if (program.has_storage_bridges &&
        !create_mpsrt_storage_bridge_ops(body_builder,
                                         loc,
                                         program.storage_bridges,
                                         /*external_inputs=*/false)) {
        return false;
    }

    auto* return_op = create_mpsrt_op(body_builder, loc, kGfxMpsrtReturnOpName);
    return_op->setAttr("gfx.mpsrt.op.generated", body_builder.getBoolAttr(true));
    return_op->setAttr("gfx.mpsrt.op.output_values",
                       detail::gfx_mpsrt_u32_vector_attr(body_builder, program.output_values));
    mlir::func::ReturnOp::create(body_builder, loc);

    module->setAttr("gfx.mpsrt.ops.symbol", builder.getStringAttr(kGfxMpsrtOpsSymbol));
    return true;
}

bool read_module_mpsrt_ops_program(mlir::ModuleOp module,
                                   GfxMpsrtProgram& out) {
    out = {};
    auto ops_func = lookup_generated_ops_func(module);
    if (!ops_func) {
        return false;
    }

    uint32_t expected_stage_count = 0;
    if (!read_program_common_attrs(ops_func.getOperation(), out, expected_stage_count)) {
        out = {};
        return false;
    }

    bool saw_return = false;
    bool invalid = false;
    ops_func.walk([&](mlir::Operation* op) {
        if (saw_return || invalid) {
            return;
        }
        if (op == ops_func.getOperation() || mlir::isa<mlir::func::ReturnOp>(op)) {
            return;
        }
        auto generated_attr = op->getAttrOfType<mlir::BoolAttr>("gfx.mpsrt.op.generated");
        if (!generated_attr || !generated_attr.getValue()) {
            return;
        }
        const auto op_name = op->getName().getStringRef();
        if (op_name == kGfxMpsrtReturnOpName) {
            const auto return_values =
                detail::gfx_mpsrt_read_u32_vector_attr(op, "gfx.mpsrt.op.output_values");
            if (!return_values.empty()) {
                out.output_values = return_values;
            }
            saw_return = true;
            return;
        }

        GfxMpsrtBuilderStageSpec spec{};
        if (!detail::gfx_mpsrt_read_stage_desc_attrs(op,
                                                     "gfx.mpsrt.op.stage",
                                                     spec.stage,
                                                     spec.stage_record_key)) {
            invalid = true;
            return;
        }
        GfxKernelStageManifest canonical_manifest{};
        if (detail::gfx_mpsrt_read_stage_manifest_attrs(op, "gfx.stage_manifest", canonical_manifest)) {
            spec.stage.stage_manifest = std::move(canonical_manifest);
            detail::gfx_mpsrt_apply_stage_manifest_to_stage_desc(spec.stage);
            spec.stage_record_key = gfx_mpsrt_stage_record_key(spec.stage);
        }
        const auto expected_name = mpsrt_op_name_for_stage(spec.stage.kind);
        uint32_t stage_index = 0;
        if (expected_name.empty() ||
            mpsrt_stage_op_name(spec.stage).empty() ||
            op_name != expected_name ||
            !read_u32_op_attr(op, "gfx.mpsrt.op.stage_index", stage_index) ||
            stage_index != out.stages.size()) {
            invalid = true;
            return;
        }
        spec.inputs = detail::gfx_mpsrt_read_u32_vector_attr(op, "gfx.mpsrt.op.input_values");
        spec.outputs = detail::gfx_mpsrt_read_u32_vector_attr(op, "gfx.mpsrt.op.output_values");
        if (!read_op_tensor_outputs(op, spec.output_descs)) {
            invalid = true;
            return;
        }
        if (spec.stage.input_storage == GfxMpsrtStorage::Unknown && !spec.inputs.empty()) {
            const auto input_value = spec.inputs.front();
            if (input_value < out.inputs.size()) {
                spec.stage.input_storage = out.inputs[input_value].storage;
            }
        }
        if (spec.stage.output_storage == GfxMpsrtStorage::Unknown && !spec.output_descs.empty()) {
            spec.stage.output_storage = spec.output_descs.front().storage;
        }
        if (spec.stage.layout == GfxMpsrtLayout::Unknown &&
            spec.stage.output_storage != GfxMpsrtStorage::Unknown) {
            spec.stage.layout = gfx_mpsrt_stage_layout_for_storage(spec.stage.output_storage);
        }
        out.stages.push_back(std::move(spec));
    });

    if (invalid || !saw_return || out.stages.size() != expected_stage_count) {
        out = {};
        return false;
    }
    out.valid = gfx_mpsrt_validate_program(out, nullptr);
    if (!out.valid) {
        out = {};
    }
    return out.valid;
}

}  // namespace gfx_plugin
}  // namespace ov
