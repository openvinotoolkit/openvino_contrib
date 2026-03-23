// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "mlir/spirv_codegen.hpp"

#include "mlir/gfx_mlir_kernel_metadata.hpp"
#include "mlir/Conversion/ArithToSPIRV/ArithToSPIRV.h"
#include "mlir/Conversion/ControlFlowToSPIRV/ControlFlowToSPIRV.h"
#include "mlir/Conversion/FuncToSPIRV/FuncToSPIRV.h"
#include "mlir/Conversion/FuncToSPIRV/FuncToSPIRVPass.h"
#include "mlir/Conversion/GPUToSPIRV/GPUToSPIRV.h"
#include "mlir/Conversion/GPUToSPIRV/GPUToSPIRVPass.h"
#include "mlir/Conversion/IndexToSPIRV/IndexToSPIRV.h"
#include "mlir/Conversion/MathToSPIRV/MathToSPIRV.h"
#include "mlir/Conversion/MemRefToSPIRV/MemRefToSPIRV.h"
#include "mlir/Conversion/MemRefToSPIRV/MemRefToSPIRVPass.h"
#include "mlir/Conversion/Passes.h"
#include "mlir/Conversion/SCFToGPU/SCFToGPUPass.h"
#include "mlir/Conversion/SCFToSPIRV/SCFToSPIRV.h"
#include "mlir/Conversion/TensorToSPIRV/TensorToSPIRV.h"
#include "mlir/Conversion/UBToSPIRV/UBToSPIRV.h"
#include "mlir/Conversion/VectorToSPIRV/VectorToSPIRV.h"
#include "mlir/Conversion/VectorToSPIRV/VectorToSPIRVPass.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Math/IR/Math.h"
#include "mlir/Dialect/ControlFlow/IR/ControlFlow.h"
#include "mlir/Dialect/Arith/Transforms/Passes.h"
#include "mlir/Dialect/Arith/Transforms/BufferizableOpInterfaceImpl.h"
#include "mlir/Dialect/Bufferization/IR/Bufferization.h"
#include "mlir/Dialect/Bufferization/Transforms/FuncBufferizableOpInterfaceImpl.h"
#include "mlir/Dialect/Bufferization/Transforms/Passes.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/GPU/IR/GPUDialect.h"
#include "mlir/Dialect/GPU/Transforms/ParallelLoopMapper.h"
#include "mlir/Dialect/GPU/Transforms/Passes.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Linalg/Transforms/BufferizableOpInterfaceImpl.h"
#include "mlir/Dialect/Linalg/Passes.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/MemRef/Transforms/Passes.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/SCF/Transforms/BufferizableOpInterfaceImpl.h"
#include "mlir/Dialect/SPIRV/IR/SPIRVDialect.h"
#include "mlir/Dialect/SPIRV/IR/SPIRVOps.h"
#include "mlir/Dialect/SPIRV/IR/TargetAndABI.h"
#include "mlir/Dialect/SPIRV/Transforms/SPIRVConversion.h"
#include "mlir/Dialect/SPIRV/Transforms/Passes.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Dialect/Tensor/Transforms/BufferizableOpInterfaceImpl.h"
#include "mlir/Dialect/Vector/IR/VectorOps.h"
#include "mlir/IR/Verifier.h"
#include "mlir/IR/Diagnostics.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Target/SPIRV/Serialization.h"
#include "mlir/Parser/Parser.h"
#include "mlir/Transforms/DialectConversion.h"
#include "mlir/Transforms/Passes.h"

#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/APFloat.h"

#include "mlir/mlir_passes.hpp"
#include "mlir/mlir_utils.hpp"
#include "mlir/gfx_mlir_debug.hpp"
#include "runtime/gfx_logger.hpp"

#include "llvm/Support/FileSystem.h"
#include "llvm/Support/raw_ostream.h"

#include <optional>
#include <cstdlib>
#include <cstring>
#include <sstream>
#include <algorithm>
namespace ov {
namespace gfx_plugin {

namespace {

void populate_spirv_patterns(const mlir::SPIRVTypeConverter& type_converter,
                             mlir::ScfToSPIRVContext& scf_to_spirv_ctx,
                             mlir::RewritePatternSet& patterns) {
    mlir::arith::populateCeilFloorDivExpandOpsPatterns(patterns);
    mlir::arith::populateArithToSPIRVPatterns(type_converter, patterns);
    mlir::populateBuiltinFuncToSPIRVPatterns(type_converter, patterns);
    mlir::populateFuncToSPIRVPatterns(type_converter, patterns);
    mlir::index::populateIndexToSPIRVPatterns(type_converter, patterns);
    mlir::populateMemRefToSPIRVPatterns(type_converter, patterns);
    mlir::populateVectorToSPIRVPatterns(type_converter, patterns);
    mlir::populateSCFToSPIRVPatterns(type_converter, scf_to_spirv_ctx, patterns);
    mlir::cf::populateControlFlowToSPIRVPatterns(type_converter, patterns);
    mlir::populateMathToSPIRVPatterns(type_converter, patterns);
    mlir::ub::populateUBToSPIRVConversionPatterns(type_converter, patterns);
}

bool has_parallel_loops(mlir::ModuleOp module) {
    bool found = false;
    module.walk([&](mlir::scf::ParallelOp) { found = true; });
    module.walk([&](mlir::gpu::LaunchOp) { found = true; });
    module.walk([&](mlir::gpu::LaunchFuncOp) { found = true; });
    return found;
}

std::array<int32_t, 3> resolve_spirv_local_size(mlir::ModuleOp module) {
    constexpr int32_t kDefaultLinearLocalSizeX = 64;
    int32_t local_y = 1;
    int32_t local_x = 1;
    bool has_explicit_local_size = false;
    if (module) {
        if (auto th = module->getAttrOfType<mlir::IntegerAttr>("gfx.dispatch_threads_h")) {
            local_y = static_cast<int32_t>(th.getInt());
            has_explicit_local_size = true;
        }
        if (auto tw = module->getAttrOfType<mlir::IntegerAttr>("gfx.dispatch_threads_w")) {
            local_x = static_cast<int32_t>(tw.getInt());
            has_explicit_local_size = true;
        }
        const bool parallel_dispatch =
            module->getAttrOfType<mlir::BoolAttr>("gfx.parallel_dispatch") &&
            module->getAttrOfType<mlir::BoolAttr>("gfx.parallel_dispatch").getValue();
        if (!parallel_dispatch && !has_explicit_local_size && local_x <= 1 && local_y <= 1) {
            local_x = kDefaultLinearLocalSizeX;
        }
    }
    local_x = std::max<int32_t>(local_x, 1);
    local_y = std::max<int32_t>(local_y, 1);
    return {local_x, local_y, 1};
}

void map_parallel_loops_to_blocks(mlir::ModuleOp module) {
    module.walk([&](mlir::scf::ParallelOp op) {
        if (op->getAttr(mlir::gpu::getMappingAttrName())) {
            return;
        }
        const int64_t num_loops = op.getNumLoops();
        if (num_loops == 0) {
            return;
        }
        bool wants_threads = false;
        if (auto mod = op->getParentOfType<mlir::ModuleOp>()) {
            auto th = mod->getAttrOfType<mlir::IntegerAttr>("gfx.dispatch_threads_h");
            auto tw = mod->getAttrOfType<mlir::IntegerAttr>("gfx.dispatch_threads_w");
            const int64_t th_val = th ? th.getInt() : 1;
            const int64_t tw_val = tw ? tw.getInt() : 1;
            wants_threads = (th_val > 1 || tw_val > 1);
        }
        mlir::OpBuilder b(op);
        llvm::SmallVector<mlir::gpu::ParallelLoopDimMappingAttr, 4> attrs;
        attrs.reserve(static_cast<size_t>(num_loops));
        for (int64_t i = 0; i < num_loops; ++i) {
            mlir::gpu::Processor proc = mlir::gpu::Processor::Sequential;
            const int64_t thread_dims = (wants_threads && num_loops >= 3) ? 2 : 0;
            const int64_t block_dims = std::min<int64_t>(3, num_loops - thread_dims);
            const int64_t block_start = num_loops - thread_dims - block_dims;
            if (thread_dims != 0) {
                if (i == num_loops - 1) {
                    proc = mlir::gpu::Processor::ThreadX;
                } else if (i == num_loops - 2) {
                    proc = mlir::gpu::Processor::ThreadY;
                }
            }
            if (proc == mlir::gpu::Processor::Sequential && i >= block_start) {
                switch (i - block_start) {
                    case 0: proc = mlir::gpu::Processor::BlockX; break;
                    case 1: proc = mlir::gpu::Processor::BlockY; break;
                    case 2: proc = mlir::gpu::Processor::BlockZ; break;
                    default: break;
                }
            }
            attrs.push_back(b.getAttr<mlir::gpu::ParallelLoopDimMappingAttr>(
                proc, b.getDimIdentityMap(), b.getDimIdentityMap()));
        }
        (void)mlir::gpu::setMappingAttr(op, attrs);
    });
}

bool convert_gpu_modules_to_spirv_with_math(mlir::ModuleOp module) {
    if (!module) {
        return false;
    }
    auto* ctx = module.getContext();
    mlir::OpBuilder builder(ctx);
    llvm::SmallVector<mlir::Operation*, 1> gpu_modules;

    auto target_env_supports_kernel_cap = [](mlir::gpu::GPUModuleOp module_op) {
        auto target_attr = mlir::spirv::lookupTargetEnvOrDefault(module_op.getOperation());
        mlir::spirv::TargetEnv target_env(target_attr);
        return target_env.allows(mlir::spirv::Capability::Kernel);
    };

    module.walk([&](mlir::gpu::GPUModuleOp module_op) {
        if (target_env_supports_kernel_cap(module_op)) {
            builder.setInsertionPointToStart(module_op.getBody());
        } else {
            builder.setInsertionPoint(module_op.getOperation());
        }
        gpu_modules.push_back(builder.clone(*module_op.getOperation()));
    });

    for (auto* gpu_module : gpu_modules) {
        auto target_attr = mlir::spirv::lookupTargetEnvOrDefault(gpu_module);
        // Ensure memref memory spaces are mapped for the cloned GPU module.
        {
            auto mem_space_map =
                target_env_supports_kernel_cap(mlir::dyn_cast<mlir::gpu::GPUModuleOp>(gpu_module))
                    ? mlir::spirv::mapMemorySpaceToOpenCLStorageClass
                    : mlir::spirv::mapMemorySpaceToVulkanStorageClass;
            mlir::spirv::MemorySpaceToStorageClassConverter converter(mem_space_map);
            mlir::spirv::convertMemRefTypesAndAttrs(gpu_module, converter);

            auto target = mlir::spirv::getMemorySpaceToStorageClassTarget(*ctx);
            bool illegal = false;
            gpu_module->walk([&](mlir::Operation* child) {
                if (target->isIllegal(child)) {
                    illegal = true;
                }
            });
            if (illegal) {
                return false;
            }
        }

        auto target = mlir::SPIRVConversionTarget::get(target_attr);

        mlir::SPIRVConversionOptions options;
        mlir::SPIRVTypeConverter type_converter(target_attr, options);
        mlir::populateMMAToSPIRVCoopMatrixTypeConversion(type_converter);

        mlir::RewritePatternSet patterns(ctx);
        mlir::arith::populateCeilFloorDivExpandOpsPatterns(patterns);
        mlir::populateGPUToSPIRVPatterns(type_converter, patterns);
        mlir::populateGpuWMMAToSPIRVCoopMatrixKHRConversionPatterns(type_converter, patterns);

        mlir::ScfToSPIRVContext scf_ctx;
        mlir::populateSCFToSPIRVPatterns(type_converter, scf_ctx, patterns);
        mlir::arith::populateArithToSPIRVPatterns(type_converter, patterns);
        mlir::populateBuiltinFuncToSPIRVPatterns(type_converter, patterns);
        mlir::index::populateIndexToSPIRVPatterns(type_converter, patterns);
        mlir::populateMathToSPIRVPatterns(type_converter, patterns);
        mlir::populateMemRefToSPIRVPatterns(type_converter, patterns);
        mlir::populateFuncToSPIRVPatterns(type_converter, patterns);
        mlir::populateVectorToSPIRVPatterns(type_converter, patterns);
        mlir::cf::populateControlFlowToSPIRVPatterns(type_converter, patterns);
        mlir::ub::populateUBToSPIRVConversionPatterns(type_converter, patterns);

        if (mlir::failed(mlir::applyFullConversion(gpu_module, *target, std::move(patterns)))) {
            return false;
        }
    }

    module.walk([&](mlir::gpu::GPUModuleOp module_op) {
        if (!target_env_supports_kernel_cap(module_op)) {
            return;
        }
        module_op.walk([&](mlir::gpu::GPUFuncOp func_op) {
            builder.setInsertionPoint(func_op);
            auto new_func = builder.create<mlir::func::FuncOp>(
                func_op.getLoc(), func_op.getName(), func_op.getFunctionType());
            auto entry = new_func.addEntryBlock();
            builder.setInsertionPointToEnd(entry);
            builder.create<mlir::func::ReturnOp>(func_op.getLoc());
            new_func->setAttr(mlir::gpu::GPUDialect::getKernelFuncAttrName(),
                              builder.getUnitAttr());
            func_op.erase();
        });
    });

    return true;
}

void ensure_spirv_entry_point_interface(mlir::spirv::ModuleOp module) {
    llvm::DenseSet<mlir::StringRef> seen;
    llvm::SmallVector<mlir::Attribute, 16> iface;
    module.walk([&](mlir::spirv::GlobalVariableOp var) {
        const auto sc = var.storageClass();
        if (sc == mlir::spirv::StorageClass::Input ||
            sc == mlir::spirv::StorageClass::Output ||
            sc == mlir::spirv::StorageClass::Uniform ||
            sc == mlir::spirv::StorageClass::UniformConstant ||
            sc == mlir::spirv::StorageClass::StorageBuffer ||
            sc == mlir::spirv::StorageClass::PushConstant) {
            auto name = var.getSymName();
            if (seen.insert(name).second) {
                iface.push_back(mlir::SymbolRefAttr::get(var));
            }
        }
    });
    module.walk([&](mlir::spirv::EntryPointOp entry) {
        if (auto existing = entry.getInterface()) {
            for (auto attr : existing) {
                if (auto sym = mlir::dyn_cast<mlir::SymbolRefAttr>(attr)) {
                    if (seen.insert(sym.getRootReference()).second) {
                        iface.push_back(sym);
                    }
                }
            }
        }
        mlir::OpBuilder b(entry);
        entry->setAttr(mlir::spirv::EntryPointOp::getInterfaceAttrName(entry->getName()),
                       b.getArrayAttr(iface));
    });
}

void apply_spirv_local_size(mlir::ModuleOp module,
                            mlir::spirv::ModuleOp spirv_module,
                            const std::string& entry_point) {
    if (!module || !spirv_module || entry_point.empty()) {
        return;
    }
    const auto local_size = resolve_spirv_local_size(module);
    const int32_t local_x = local_size[0];
    const int32_t local_y = local_size[1];
    const bool has_explicit_local_size =
        module->getAttr("gfx.dispatch_threads_h") ||
        module->getAttr("gfx.dispatch_threads_w") ||
        module->getAttr("gfx.parallel_dispatch");
    if (!has_explicit_local_size &&
        local_x <= 1 && local_y <= 1 && local_size[2] <= 1) {
        return;
    }

    auto* ctx = spirv_module.getContext();
    auto exec_mode = mlir::spirv::ExecutionModeAttr::get(
        ctx, mlir::spirv::ExecutionMode::LocalSize);
    auto values = mlir::ArrayAttr::get(
        ctx,
        {mlir::IntegerAttr::get(mlir::IntegerType::get(ctx, 32), static_cast<int32_t>(local_x)),
         mlir::IntegerAttr::get(mlir::IntegerType::get(ctx, 32), static_cast<int32_t>(local_y)),
         mlir::IntegerAttr::get(mlir::IntegerType::get(ctx, 32), static_cast<int32_t>(local_size[2]))});

    bool updated = false;
    spirv_module.walk([&](mlir::spirv::ExecutionModeOp op) {
        if (op.getExecutionMode() != mlir::spirv::ExecutionMode::LocalSize) {
            return;
        }
        auto fn = op.getFnAttr();
        if (!fn || fn.getRootReference().str() != entry_point) {
            return;
        }
        op.setValuesAttr(values);
        updated = true;
    });
    if (updated) {
        return;
    }

    mlir::OpBuilder b(spirv_module.getBody(), spirv_module.getBody()->begin());
    b.create<mlir::spirv::ExecutionModeOp>(
        spirv_module.getLoc(),
        mlir::SymbolRefAttr::get(ctx, entry_point),
        exec_mode,
        values);
}

bool validate_spirv_module(mlir::spirv::ModuleOp module, std::string* log) {
    if (!module) {
        return true;
    }
    const auto binding_attr_name =
        mlir::spirv::SPIRVDialect::getAttributeName(mlir::spirv::Decoration::Binding);
    const auto set_attr_name =
        mlir::spirv::SPIRVDialect::getAttributeName(mlir::spirv::Decoration::DescriptorSet);
    const auto builtin_attr_name =
        mlir::spirv::SPIRVDialect::getAttributeName(mlir::spirv::Decoration::BuiltIn);

    llvm::SmallVector<int32_t, 16> bindings;
    llvm::SmallDenseSet<int32_t, 16> seen;
    bool ok = true;

    module.walk([&](mlir::spirv::GlobalVariableOp var) {
        const auto sc = var.storageClass();
        if (sc != mlir::spirv::StorageClass::StorageBuffer &&
            sc != mlir::spirv::StorageClass::Uniform) {
            return;
        }
        if (var->hasAttr(builtin_attr_name)) {
            return;
        }
        auto binding_attr = var->getAttrOfType<mlir::IntegerAttr>(binding_attr_name);
        auto set_attr = var->getAttrOfType<mlir::IntegerAttr>(set_attr_name);
        if (!binding_attr || !set_attr) {
            ok = false;
            if (log) {
                *log += "SPIR-V validation: missing binding/descriptor_set on ";
                *log += var.getSymName().str();
                *log += "\n";
            }
            return;
        }
        const int32_t binding = static_cast<int32_t>(binding_attr.getInt());
        const int32_t set = static_cast<int32_t>(set_attr.getInt());
        if (set != 0) {
            ok = false;
            if (log) {
                *log += "SPIR-V validation: descriptor_set != 0 for ";
                *log += var.getSymName().str();
                *log += "\n";
            }
        }
        if (seen.contains(binding)) {
            ok = false;
            if (log) {
                *log += "SPIR-V validation: duplicate binding ";
                *log += std::to_string(binding);
                *log += " for ";
                *log += var.getSymName().str();
                *log += "\n";
            }
        } else {
            seen.insert(binding);
        }
        bindings.push_back(binding);
    });

    if (!ok) {
        return false;
    }
    if (!bindings.empty()) {
        std::sort(bindings.begin(), bindings.end());
        for (size_t i = 0; i < bindings.size(); ++i) {
            if (bindings[i] != static_cast<int32_t>(i)) {
                ok = false;
                if (log) {
                    *log += "SPIR-V validation: non-contiguous binding ";
                    *log += std::to_string(bindings[i]);
                    *log += " (expected ";
                    *log += std::to_string(i);
                    *log += ")\n";
                }
                break;
            }
        }
    }
    return ok;
}

std::optional<int64_t> eval_scalar_value(mlir::Value value);

bool get_static_strides_and_offset(mlir::MemRefType memref,
                                   llvm::SmallVectorImpl<int64_t>& strides,
                                   int64_t& offset) {
    if (!memref) {
        return false;
    }
    const auto shape = memref.getShape();
    if (llvm::any_of(shape, [](int64_t dim) { return dim == mlir::ShapedType::kDynamic; })) {
        return false;
    }
    if (memref.getLayout().isIdentity()) {
        strides.assign(shape.size(), 1);
        for (int64_t i = static_cast<int64_t>(shape.size()) - 2; i >= 0; --i) {
            strides[static_cast<size_t>(i)] =
                strides[static_cast<size_t>(i + 1)] * shape[static_cast<size_t>(i + 1)];
        }
        offset = 0;
        return true;
    }
    if (auto layout = mlir::dyn_cast<mlir::StridedLayoutAttr>(memref.getLayout())) {
        auto maybe_offset = layout.getOffset();
        if (maybe_offset == mlir::ShapedType::kDynamic) {
            return false;
        }
        offset = static_cast<int64_t>(maybe_offset);
        auto layout_strides = layout.getStrides();
        if (layout_strides.size() != shape.size()) {
            return false;
        }
        strides.clear();
        strides.reserve(layout_strides.size());
        for (auto s : layout_strides) {
            if (s == mlir::ShapedType::kDynamic) {
                return false;
            }
            strides.push_back(static_cast<int64_t>(s));
        }
        return true;
    }
    return false;
}

std::optional<int64_t> eval_from_extract_strided_metadata(mlir::Value value) {
    auto meta = value.getDefiningOp<mlir::memref::ExtractStridedMetadataOp>();
    if (!meta) {
        return std::nullopt;
    }
    auto memref = mlir::dyn_cast<mlir::MemRefType>(meta.getSource().getType());
    if (!memref) {
        return std::nullopt;
    }
    int64_t offset = 0;
    llvm::SmallVector<int64_t, 4> strides;
    if (!get_static_strides_and_offset(memref, strides, offset)) {
        return std::nullopt;
    }
    if (value == meta.getOffset()) {
        return offset;
    }
    auto sizes = meta.getSizes();
    auto shape = memref.getShape();
    for (auto it : llvm::enumerate(sizes)) {
        if (it.value() == value) {
            int64_t size = shape[it.index()];
            if (size == mlir::ShapedType::kDynamic) {
                return std::nullopt;
            }
            return size;
        }
    }
    auto strides_vals = meta.getStrides();
    for (auto it : llvm::enumerate(strides_vals)) {
        if (it.value() == value) {
            int64_t stride = strides[it.index()];
            if (stride == mlir::ShapedType::kDynamic) {
                return std::nullopt;
            }
            return stride;
        }
    }
    return std::nullopt;
}

std::optional<int64_t> eval_scalar_value(mlir::Value value) {
    if (!value) {
        return std::nullopt;
    }
    if (auto cst = value.getDefiningOp<mlir::arith::ConstantOp>()) {
        auto attr = cst.getValue();
        if (auto iattr = llvm::dyn_cast<mlir::IntegerAttr>(attr)) {
            return static_cast<int64_t>(iattr.getInt());
        }
        if (auto fattr = llvm::dyn_cast<mlir::FloatAttr>(attr)) {
            const auto type = fattr.getType();
            if (type.isF16()) {
                const auto bits = fattr.getValue().bitcastToAPInt().getZExtValue();
                return static_cast<int64_t>(bits);
            }
            if (type.isF32()) {
                float f = static_cast<float>(fattr.getValueAsDouble());
                uint32_t bits = 0;
                static_assert(sizeof(bits) == sizeof(f), "f32 size mismatch");
                std::memcpy(&bits, &f, sizeof(bits));
                return static_cast<int64_t>(bits);
            }
        }
    }
    if (auto cidx = value.getDefiningOp<mlir::arith::ConstantIndexOp>()) {
        return static_cast<int64_t>(cidx.value());
    }
    if (auto cint = value.getDefiningOp<mlir::arith::ConstantIntOp>()) {
        return static_cast<int64_t>(cint.value());
    }
    if (auto meta = eval_from_extract_strided_metadata(value)) {
        return meta;
    }
    if (auto cast = value.getDefiningOp<mlir::arith::IndexCastOp>()) {
        return eval_scalar_value(cast.getIn());
    }
    if (auto ext = value.getDefiningOp<mlir::arith::ExtSIOp>()) {
        return eval_scalar_value(ext.getIn());
    }
    if (auto extu = value.getDefiningOp<mlir::arith::ExtUIOp>()) {
        return eval_scalar_value(extu.getIn());
    }
    if (auto trunc = value.getDefiningOp<mlir::arith::TruncIOp>()) {
        return eval_scalar_value(trunc.getIn());
    }
    if (auto dim = value.getDefiningOp<mlir::memref::DimOp>()) {
        auto memref = mlir::dyn_cast<mlir::MemRefType>(dim.getSource().getType());
        if (memref) {
            auto maybe_index = dim.getConstantIndex();
            if (maybe_index && *maybe_index < static_cast<int64_t>(memref.getRank())) {
                auto shape = memref.getShape();
                const int64_t size = shape[static_cast<size_t>(*maybe_index)];
                if (size != mlir::ShapedType::kDynamic) {
                    return size;
                }
            }
        }
    }
    if (auto add = value.getDefiningOp<mlir::arith::AddIOp>()) {
        auto lhs = eval_scalar_value(add.getLhs());
        auto rhs = eval_scalar_value(add.getRhs());
        if (lhs && rhs) {
            return *lhs + *rhs;
        }
    }
    if (auto sub = value.getDefiningOp<mlir::arith::SubIOp>()) {
        auto lhs = eval_scalar_value(sub.getLhs());
        auto rhs = eval_scalar_value(sub.getRhs());
        if (lhs && rhs) {
            return *lhs - *rhs;
        }
    }
    if (auto mul = value.getDefiningOp<mlir::arith::MulIOp>()) {
        auto lhs = eval_scalar_value(mul.getLhs());
        auto rhs = eval_scalar_value(mul.getRhs());
        if (lhs && rhs) {
            return *lhs * *rhs;
        }
    }
    if (auto div = value.getDefiningOp<mlir::arith::DivSIOp>()) {
        auto lhs = eval_scalar_value(div.getLhs());
        auto rhs = eval_scalar_value(div.getRhs());
        if (lhs && rhs && *rhs != 0) {
            return *lhs / *rhs;
        }
    }
    if (auto divu = value.getDefiningOp<mlir::arith::DivUIOp>()) {
        auto lhs = eval_scalar_value(divu.getLhs());
        auto rhs = eval_scalar_value(divu.getRhs());
        if (lhs && rhs && *rhs != 0) {
            return static_cast<int64_t>(static_cast<uint64_t>(*lhs) / static_cast<uint64_t>(*rhs));
        }
    }
    if (auto div = value.getDefiningOp<mlir::arith::CeilDivSIOp>()) {
        auto lhs = eval_scalar_value(div.getLhs());
        auto rhs = eval_scalar_value(div.getRhs());
        if (lhs && rhs && *rhs != 0) {
            const int64_t a = *lhs;
            const int64_t b = *rhs;
            return (a + b - 1) / b;
        }
    }
    if (auto div = value.getDefiningOp<mlir::arith::CeilDivUIOp>()) {
        auto lhs = eval_scalar_value(div.getLhs());
        auto rhs = eval_scalar_value(div.getRhs());
        if (lhs && rhs && *rhs != 0) {
            const uint64_t a = static_cast<uint64_t>(*lhs);
            const uint64_t b = static_cast<uint64_t>(*rhs);
            return static_cast<int64_t>((a + b - 1) / b);
        }
    }
    if (auto div = value.getDefiningOp<mlir::arith::FloorDivSIOp>()) {
        auto lhs = eval_scalar_value(div.getLhs());
        auto rhs = eval_scalar_value(div.getRhs());
        if (lhs && rhs && *rhs != 0) {
            const int64_t a = *lhs;
            const int64_t b = *rhs;
            return a / b;
        }
    }
    if (auto rem = value.getDefiningOp<mlir::arith::RemSIOp>()) {
        auto lhs = eval_scalar_value(rem.getLhs());
        auto rhs = eval_scalar_value(rem.getRhs());
        if (lhs && rhs && *rhs != 0) {
            return *lhs % *rhs;
        }
    }
    if (auto remu = value.getDefiningOp<mlir::arith::RemUIOp>()) {
        auto lhs = eval_scalar_value(remu.getLhs());
        auto rhs = eval_scalar_value(remu.getRhs());
        if (lhs && rhs && *rhs != 0) {
            return static_cast<int64_t>(static_cast<uint64_t>(*lhs) % static_cast<uint64_t>(*rhs));
        }
    }
    if (auto shl = value.getDefiningOp<mlir::arith::ShLIOp>()) {
        auto lhs = eval_scalar_value(shl.getLhs());
        auto rhs = eval_scalar_value(shl.getRhs());
        if (lhs && rhs) {
            return *lhs << *rhs;
        }
    }
    if (auto shr = value.getDefiningOp<mlir::arith::ShRSIOp>()) {
        auto lhs = eval_scalar_value(shr.getLhs());
        auto rhs = eval_scalar_value(shr.getRhs());
        if (lhs && rhs) {
            return *lhs >> *rhs;
        }
    }
    if (auto shr = value.getDefiningOp<mlir::arith::ShRUIOp>()) {
        auto lhs = eval_scalar_value(shr.getLhs());
        auto rhs = eval_scalar_value(shr.getRhs());
        if (lhs && rhs) {
            return static_cast<int64_t>(static_cast<uint64_t>(*lhs) >> static_cast<uint64_t>(*rhs));
        }
    }
    if (auto andi = value.getDefiningOp<mlir::arith::AndIOp>()) {
        auto lhs = eval_scalar_value(andi.getLhs());
        auto rhs = eval_scalar_value(andi.getRhs());
        if (lhs && rhs) {
            return *lhs & *rhs;
        }
    }
    if (auto ori = value.getDefiningOp<mlir::arith::OrIOp>()) {
        auto lhs = eval_scalar_value(ori.getLhs());
        auto rhs = eval_scalar_value(ori.getRhs());
        if (lhs && rhs) {
            return *lhs | *rhs;
        }
    }
    if (auto xori = value.getDefiningOp<mlir::arith::XOrIOp>()) {
        auto lhs = eval_scalar_value(xori.getLhs());
        auto rhs = eval_scalar_value(xori.getRhs());
        if (lhs && rhs) {
            return *lhs ^ *rhs;
        }
    }
    if (auto min = value.getDefiningOp<mlir::arith::MinSIOp>()) {
        auto lhs = eval_scalar_value(min.getLhs());
        auto rhs = eval_scalar_value(min.getRhs());
        if (lhs && rhs) {
            return std::min(*lhs, *rhs);
        }
    }
    if (auto max = value.getDefiningOp<mlir::arith::MaxSIOp>()) {
        auto lhs = eval_scalar_value(max.getLhs());
        auto rhs = eval_scalar_value(max.getRhs());
        if (lhs && rhs) {
            return std::max(*lhs, *rhs);
        }
    }
    if (auto min = value.getDefiningOp<mlir::arith::MinUIOp>()) {
        auto lhs = eval_scalar_value(min.getLhs());
        auto rhs = eval_scalar_value(min.getRhs());
        if (lhs && rhs) {
            const uint64_t a = static_cast<uint64_t>(*lhs);
            const uint64_t b = static_cast<uint64_t>(*rhs);
            return static_cast<int64_t>(std::min(a, b));
        }
    }
    if (auto max = value.getDefiningOp<mlir::arith::MaxUIOp>()) {
        auto lhs = eval_scalar_value(max.getLhs());
        auto rhs = eval_scalar_value(max.getRhs());
        if (lhs && rhs) {
            const uint64_t a = static_cast<uint64_t>(*lhs);
            const uint64_t b = static_cast<uint64_t>(*rhs);
            return static_cast<int64_t>(std::max(a, b));
        }
    }
    if (auto sel = value.getDefiningOp<mlir::arith::SelectOp>()) {
        auto cond = eval_scalar_value(sel.getCondition());
        if (!cond) {
            return std::nullopt;
        }
        return *cond ? eval_scalar_value(sel.getTrueValue())
                     : eval_scalar_value(sel.getFalseValue());
    }
    if (auto cmp = value.getDefiningOp<mlir::arith::CmpIOp>()) {
        auto lhs = eval_scalar_value(cmp.getLhs());
        auto rhs = eval_scalar_value(cmp.getRhs());
        if (lhs && rhs) {
            bool result = false;
            switch (cmp.getPredicate()) {
            case mlir::arith::CmpIPredicate::eq: result = (*lhs == *rhs); break;
            case mlir::arith::CmpIPredicate::ne: result = (*lhs != *rhs); break;
            case mlir::arith::CmpIPredicate::slt: result = (*lhs < *rhs); break;
            case mlir::arith::CmpIPredicate::sle: result = (*lhs <= *rhs); break;
            case mlir::arith::CmpIPredicate::sgt: result = (*lhs > *rhs); break;
            case mlir::arith::CmpIPredicate::sge: result = (*lhs >= *rhs); break;
            case mlir::arith::CmpIPredicate::ult:
                result = (static_cast<uint64_t>(*lhs) < static_cast<uint64_t>(*rhs));
                break;
            case mlir::arith::CmpIPredicate::ule:
                result = (static_cast<uint64_t>(*lhs) <= static_cast<uint64_t>(*rhs));
                break;
            case mlir::arith::CmpIPredicate::ugt:
                result = (static_cast<uint64_t>(*lhs) > static_cast<uint64_t>(*rhs));
                break;
            case mlir::arith::CmpIPredicate::uge:
                result = (static_cast<uint64_t>(*lhs) >= static_cast<uint64_t>(*rhs));
                break;
            }
            return result ? 1 : 0;
        }
    }
    return std::nullopt;
}

void dump_spirv_mlir_if_requested(mlir::spirv::ModuleOp module, const std::string& entry_point) {
    if (!module || !gfx_log_debug_enabled()) {
        return;
    }
    const char* dump_root = std::getenv("OV_GFX_DUMP_SPIRV_MLIR");
    if (!dump_root || !*dump_root) {
        return;
    }
    if (const char* filter = std::getenv("OV_GFX_DUMP_SPIRV_MLIR_FILTER")) {
        if (*filter && entry_point.find(filter) == std::string::npos) {
            return;
        }
    }
    std::string path(dump_root);
    if (!path.empty() && path.back() == '/') {
        const std::string name = entry_point.empty() ? "gfx_kernel" : entry_point;
        path += name;
        path += ".mlir";
    }
    std::error_code ec;
    llvm::raw_fd_ostream os(path, ec, llvm::sys::fs::OF_Text);
    if (ec) {
        gfx_log_warn("MLIR") << "Failed to open SPIR-V MLIR dump path: " << path;
        return;
    }
    module.print(os);
    os << "\n";
    gfx_log_debug("MLIR") << "Wrote SPIR-V MLIR dump: " << path;
}

void dump_mlir_if_requested(mlir::ModuleOp module,
                            const char* env_name,
                            const std::string& entry_point,
                            const char* suffix) {
    if (!module || !gfx_log_debug_enabled()) {
        return;
    }
    const char* dump_root = std::getenv(env_name);
    if (!dump_root || !*dump_root) {
        return;
    }
    std::string path(dump_root);
    if (!path.empty() && path.back() == '/') {
        const std::string name = entry_point.empty() ? "gfx_kernel" : entry_point;
        path += name;
        if (suffix) {
            path += suffix;
        }
        path += ".mlir";
    }
    std::error_code ec;
    llvm::raw_fd_ostream os(path, ec, llvm::sys::fs::OF_Text);
    if (ec) {
        gfx_log_warn("MLIR") << "Failed to open MLIR dump path: " << path;
        return;
    }
    module.print(os);
    os << "\n";
    gfx_log_debug("MLIR") << "Wrote MLIR dump: " << path;
}

void annotate_kernel_scalar_args(mlir::ModuleOp module) {
    const bool has_fixed_arg_count = module->hasAttr("gfx.fixed_arg_count");
    if (module->hasAttr("gfx.kernel_operand_kinds") && !has_fixed_arg_count) {
        return;
    }
    auto strip_memref_casts = [](mlir::Value value) -> mlir::Value {
        mlir::Value current = value;
        while (auto op = current.getDefiningOp()) {
            if (auto cast = mlir::dyn_cast<mlir::memref::CastOp>(op)) {
                current = cast.getSource();
                continue;
            }
            if (auto cast = mlir::dyn_cast<mlir::memref::ReinterpretCastOp>(op)) {
                current = cast.getSource();
                continue;
            }
            if (auto subview = mlir::dyn_cast<mlir::memref::SubViewOp>(op)) {
                current = subview.getSource();
                continue;
            }
            break;
        }
        return current;
    };
    bool recorded = false;
    bool saw_launch = false;
    module.walk([&](mlir::gpu::LaunchFuncOp launch) {
        saw_launch = true;
        if (recorded) {
            return;
        }
        llvm::SmallVector<int32_t, 16> operand_kinds;
        llvm::SmallVector<int32_t, 16> operand_arg_indices;
        llvm::SmallVector<int32_t, 8> scalar_values;
        const bool preserve_fixed_buffers = has_fixed_arg_count && module->hasAttr("gfx.kernel_operand_kinds");
        if (preserve_fixed_buffers) {
            auto existing_kinds = extract_kernel_operand_kinds(module);
            auto existing_indices = extract_kernel_operand_arg_indices(module);
            operand_kinds.assign(existing_kinds.begin(), existing_kinds.end());
            operand_arg_indices.assign(existing_indices.begin(), existing_indices.end());
        } else {
            operand_kinds.reserve(launch.getKernelOperands().size());
            operand_arg_indices.reserve(launch.getKernelOperands().size());
        }
        bool any_scalar = false;
        bool all_scalars_known = true;
        for (auto operand : launch.getKernelOperands()) {
            auto type = operand.getType();
            if (mlir::isa<mlir::MemRefType>(type)) {
                if (!preserve_fixed_buffers) {
                    operand_kinds.push_back(1);
                    int32_t arg_idx = -1;
                    auto base = strip_memref_casts(operand);
                    if (auto barg = mlir::dyn_cast<mlir::BlockArgument>(base)) {
                        arg_idx = static_cast<int32_t>(barg.getArgNumber());
                    }
                    operand_arg_indices.push_back(arg_idx);
                }
                continue;
            }
            auto maybe_value = eval_scalar_value(operand);
            int64_t value = 0;
            if (maybe_value) {
                value = *maybe_value;
            } else {
                all_scalars_known = false;
            }
            operand_kinds.push_back(0);
            operand_arg_indices.push_back(-1);
            scalar_values.push_back(static_cast<int32_t>(value));
            any_scalar = true;
        }
        if (gfx_mlir_debug_enabled() && preserve_fixed_buffers) {
            std::ostringstream oss;
            oss << "Fixed-ABI kernel launch operands:";
            size_t scalar_idx = 0;
            for (auto operand : launch.getKernelOperands()) {
                oss << " [";
                if (mlir::isa<mlir::MemRefType>(operand.getType())) {
                    oss << "memref";
                } else {
                    oss << "scalar";
                    if (scalar_idx < scalar_values.size()) {
                        oss << "=" << scalar_values[scalar_idx++];
                    }
                }
                oss << "]";
            }
            gfx_log_debug("MLIR") << oss.str();
        }
        if (!operand_kinds.empty()) {
            mlir::OpBuilder b(launch);
            auto make_i32_array_attr = [&](llvm::ArrayRef<int32_t> vals) {
                llvm::SmallVector<mlir::Attribute, 16> attrs;
                attrs.reserve(vals.size());
                for (auto v : vals) {
                    attrs.push_back(b.getI32IntegerAttr(v));
                }
                return b.getArrayAttr(attrs);
            };
            module->setAttr("gfx.kernel_operand_kinds", make_i32_array_attr(operand_kinds));
            module->setAttr("gfx.kernel_operand_arg_indices", make_i32_array_attr(operand_arg_indices));
            if (any_scalar) {
                module->setAttr("gfx.kernel_scalar_values", make_i32_array_attr(scalar_values));
            }

            bool prefix = true;
            bool seen_memref = false;
            for (auto kind : operand_kinds) {
                if (kind == 1) {
                    seen_memref = true;
                    continue;
                }
                if (seen_memref) {
                    prefix = false;
                    break;
                }
            }
            if (prefix && all_scalars_known && !scalar_values.empty()) {
                module->setAttr("gfx.kernel_scalar_args", make_i32_array_attr(scalar_values));
            }
            if (gfx_log_debug_enabled()) {
                std::ostringstream oss;
                oss << "Kernel operand kinds=" << operand_kinds.size()
                    << " scalars=" << scalar_values.size()
                    << " prefix=" << (prefix ? "true" : "false")
                    << " all_scalars_known=" << (all_scalars_known ? "true" : "false");
                gfx_log_debug("MLIR") << oss.str();
            }
        } else if (gfx_log_debug_enabled()) {
            gfx_log_debug("MLIR") << "Kernel scalar annotation skipped (no operands)";
        }
        recorded = true;
    });
    if (!saw_launch && gfx_log_debug_enabled()) {
        gfx_log_debug("MLIR") << "No gpu.launch_func found for kernel scalar annotation";
    }
}

bool erase_noop_unrealized_casts(mlir::ModuleOp module, std::string* log) {
    llvm::SmallVector<mlir::UnrealizedConversionCastOp, 8> to_erase;
    bool unresolved = false;
    module.walk([&](mlir::UnrealizedConversionCastOp op) {
        if (op->getNumOperands() != 1 || op->getNumResults() != 1) {
            unresolved = true;
            return;
        }
        auto src = op.getOperand(0).getType();
        auto dst = op.getResult(0).getType();
        bool can_drop = (src == dst);
        if (!can_drop) {
            auto src_ptr = mlir::dyn_cast<mlir::spirv::PointerType>(src);
            auto dst_ptr = mlir::dyn_cast<mlir::spirv::PointerType>(dst);
            if (src_ptr && dst_ptr &&
                src_ptr.getStorageClass() == dst_ptr.getStorageClass() &&
                src_ptr.getPointeeType() == dst_ptr.getPointeeType()) {
                can_drop = true;
            }
        }
        if (can_drop) {
            op.getResult(0).replaceAllUsesWith(op.getOperand(0));
            to_erase.push_back(op);
            return;
        }
        unresolved = true;
        if (log) {
            std::string op_dump;
            {
                llvm::raw_string_ostream os(op_dump);
                op->print(os);
            }
            *log += "Unresolved unrealized_conversion_cast: " + op_dump + "\n";
        }
    });
    for (auto op : to_erase) {
        op.erase();
    }
    return !unresolved;
}

bool eliminate_memref_cast_chains(mlir::ModuleOp module) {
    llvm::SmallVector<mlir::Operation*, 16> to_erase;
    bool changed = false;
    module.walk([&](mlir::UnrealizedConversionCastOp cast_in) {
        if (cast_in->getNumOperands() != 1 || cast_in->getNumResults() != 1) {
            return;
        }
        auto ptr_ty = mlir::dyn_cast<mlir::spirv::PointerType>(cast_in.getOperand(0).getType());
        auto memref_ty = mlir::dyn_cast<mlir::MemRefType>(cast_in.getResult(0).getType());
        if (!ptr_ty || !memref_ty) {
            return;
        }
        if (!cast_in.getResult(0).hasOneUse()) {
            return;
        }
        mlir::Operation* user = *cast_in.getResult(0).getUsers().begin();
        if (!user) {
            return;
        }
        mlir::Value shaped_result;
        if (auto collapse = mlir::dyn_cast<mlir::memref::CollapseShapeOp>(user)) {
            shaped_result = collapse.getResult();
            to_erase.push_back(user);
        } else if (auto expand = mlir::dyn_cast<mlir::memref::ExpandShapeOp>(user)) {
            shaped_result = expand.getResult();
            to_erase.push_back(user);
        } else {
            return;
        }
        if (!shaped_result || !shaped_result.hasOneUse()) {
            return;
        }
        auto* cast_out_op = *shaped_result.getUsers().begin();
        auto cast_out = mlir::dyn_cast_or_null<mlir::UnrealizedConversionCastOp>(cast_out_op);
        if (!cast_out || cast_out->getNumResults() != 1) {
            return;
        }
        auto out_ptr_ty =
            mlir::dyn_cast<mlir::spirv::PointerType>(cast_out.getResult(0).getType());
        if (!out_ptr_ty || out_ptr_ty != ptr_ty) {
            return;
        }
        cast_out.getResult(0).replaceAllUsesWith(cast_in.getOperand(0));
        to_erase.push_back(cast_out.getOperation());
        to_erase.push_back(cast_in.getOperation());
        changed = true;
    });

    if (changed) {
        for (auto* op : to_erase) {
            if (op) {
                op->erase();
            }
        }
    }
    return changed;
}

mlir::spirv::ModuleOp find_spirv_module(mlir::ModuleOp module) {
    mlir::spirv::ModuleOp spirv_module;
    module.walk([&](mlir::spirv::ModuleOp op) {
        if (!spirv_module) {
            spirv_module = op;
        }
    });
    return spirv_module;
}

std::vector<uint32_t> serialize_spirv(mlir::spirv::ModuleOp spirv_module, std::string* log) {
    llvm::SmallVector<uint32_t, 0> binary;
    mlir::spirv::SerializationOptions opts;
    opts.emitDebugInfo = false;
    if (mlir::failed(mlir::spirv::serialize(spirv_module, binary, opts))) {
        if (log) {
            *log = "SPIR-V serialization failed";
        }
        return {};
    }
    return std::vector<uint32_t>(binary.begin(), binary.end());
}

}  // namespace

std::vector<uint32_t> build_stub_spirv(const std::string& entry_point, std::string* log) {
    mlir::MLIRContext stub_ctx;
    stub_ctx.loadDialect<mlir::spirv::SPIRVDialect, mlir::func::FuncDialect>();
    const std::string entry = entry_point.empty() ? "gfx_stub" : entry_point;
    const std::string text =
        "module { "
        "spirv.module Logical GLSL450 requires #spirv.vce<v1.0, [Shader], []> { "
        "spirv.func @" + entry + "() -> () \"None\" { "
        "spirv.Return "
        "} "
        "spirv.EntryPoint \"GLCompute\" @" + entry + " "
        "spirv.ExecutionMode @" + entry + " \"LocalSize\", 1, 1, 1 "
        "} "
        "}";
    auto parsed = mlir::parseSourceString<mlir::ModuleOp>(text, &stub_ctx);
    if (!parsed) {
        if (log) {
            *log = "Failed to parse stub SPIR-V module";
        }
        return {};
    }
    auto spirv_module = find_spirv_module(*parsed);
    return spirv_module ? serialize_spirv(spirv_module, log) : std::vector<uint32_t>{};
}

static std::vector<uint32_t> lower_to_spirv_impl(mlir::ModuleOp module,
                                                 const std::string& entry_point,
                                                 std::string* log,
                                                 bool use_alloca,
                                                 bool use_parallel_loops) {
    if (!module) {
        if (log) {
            *log = "MLIR module is null";
        }
        return {};
    }

    auto* ctx = module.getContext();
    auto preserved_operand_kinds = module->getAttr("gfx.kernel_operand_kinds");
    auto preserved_operand_arg_indices = module->getAttr("gfx.kernel_operand_arg_indices");
    auto preserved_scalar_values = module->getAttr("gfx.kernel_scalar_values");
    auto preserved_scalar_args = module->getAttr("gfx.kernel_scalar_args");
    auto preserved_fixed_arg_count = module->getAttr("gfx.fixed_arg_count");
    const bool preserve_compact_memref_abi = preserved_fixed_arg_count != nullptr;
    mlir::DialectRegistry registry;
    mlir::arith::registerBufferizableOpInterfaceExternalModels(registry);
    mlir::tensor::registerBufferizableOpInterfaceExternalModels(registry);
    mlir::scf::registerBufferizableOpInterfaceExternalModels(registry);
    mlir::linalg::registerBufferizableOpInterfaceExternalModels(registry);
    mlir::bufferization::func_ext::registerBufferizableOpInterfaceExternalModels(registry);
    ctx->appendDialectRegistry(registry);

    ctx->loadDialect<mlir::bufferization::BufferizationDialect,
                     mlir::memref::MemRefDialect,
                     mlir::scf::SCFDialect,
                     mlir::func::FuncDialect,
                     mlir::linalg::LinalgDialect,
                     mlir::arith::ArithDialect,
                     mlir::math::MathDialect,
                     mlir::tensor::TensorDialect,
                     mlir::vector::VectorDialect,
                     mlir::spirv::SPIRVDialect>();

    if (gfx_mlir_debug_enabled()) {
        llvm::errs() << "[GFX][MLIR] Pre-verify module:\n";
        module.dump();
    }
    if (mlir::failed(mlir::verify(module))) {
        llvm::errs() << "[GFX][MLIR] Verification failed before SPIR-V lowering "
                        "(operandSegmentSizes attr bug), continuing\n";
    }

    try {
        run_mlir_pipeline(module, /*use_alloca=*/use_alloca, /*use_parallel_loops=*/use_parallel_loops);
    } catch (const std::exception& e) {
        if (log) {
            *log = std::string("MLIR preprocessing failed: ") + e.what();
        }
        return {};
    }

    strip_strided_func_layouts(module, gfx_mlir_debug_enabled());
    if (preserved_operand_kinds && !module->getAttr("gfx.kernel_operand_kinds")) {
        module->setAttr("gfx.kernel_operand_kinds", preserved_operand_kinds);
    }
    if (preserved_operand_arg_indices && !module->getAttr("gfx.kernel_operand_arg_indices")) {
        module->setAttr("gfx.kernel_operand_arg_indices", preserved_operand_arg_indices);
    }
    if (preserved_scalar_values && !module->getAttr("gfx.kernel_scalar_values")) {
        module->setAttr("gfx.kernel_scalar_values", preserved_scalar_values);
    }
    if (preserved_scalar_args && !module->getAttr("gfx.kernel_scalar_args")) {
        module->setAttr("gfx.kernel_scalar_args", preserved_scalar_args);
    }
    if (preserved_fixed_arg_count && !module->getAttr("gfx.fixed_arg_count")) {
        module->setAttr("gfx.fixed_arg_count", preserved_fixed_arg_count);
    }
    dump_mlir_if_requested(module, "OV_GFX_DUMP_MLIR_PRE_SPIRV", entry_point, "_pre_spirv");

    const bool spirv_debug = gfx_mlir_debug_enabled();

    if (spirv_debug) {
        ctx->disableMultithreading();
        llvm::errs() << "[GFX][MLIR] SPIR-V lowering input module:\n";
        module.dump();
    }

    std::string diag_log;
    mlir::ScopedDiagnosticHandler diag_handler(ctx, [&](mlir::Diagnostic& diag) {
        llvm::raw_string_ostream os(diag_log);
        diag.print(os);
        os << "\n";
        return mlir::success();
    });

    bool needs_fp16 = false;
    bool needs_int8 = false;
    bool needs_int64 = false;
    auto check_type = [&](mlir::Type t) {
        if (auto shaped = mlir::dyn_cast<mlir::ShapedType>(t)) {
            t = shaped.getElementType();
        }
        if (mlir::isa<mlir::Float16Type>(t)) {
            needs_fp16 = true;
        }
        if (auto int_ty = mlir::dyn_cast<mlir::IntegerType>(t)) {
            if (int_ty.getWidth() == 8) {
                needs_int8 = true;
            }
            if (int_ty.getWidth() == 64) {
                needs_int64 = true;
            }
        }
    };
    module.walk([&](mlir::Operation* op) {
        for (auto t : op->getResultTypes()) {
            check_type(t);
        }
        for (auto v : op->getOperands()) {
            check_type(v.getType());
        }
    });

    llvm::SmallVector<mlir::spirv::Capability, 4> caps;
    llvm::SmallVector<mlir::spirv::Extension, 4> exts;
    caps.push_back(mlir::spirv::Capability::Shader);
    exts.push_back(mlir::spirv::Extension::SPV_KHR_storage_buffer_storage_class);
    if (needs_fp16) {
        caps.push_back(mlir::spirv::Capability::Float16);
        caps.push_back(mlir::spirv::Capability::StorageBuffer16BitAccess);
        exts.push_back(mlir::spirv::Extension::SPV_KHR_16bit_storage);
    }
    if (needs_int8) {
        caps.push_back(mlir::spirv::Capability::Int8);
        caps.push_back(mlir::spirv::Capability::StorageBuffer8BitAccess);
        exts.push_back(mlir::spirv::Extension::SPV_KHR_8bit_storage);
    }
    if (needs_int64) {
        caps.push_back(mlir::spirv::Capability::Int64);
    }

    auto vce = mlir::spirv::VerCapExtAttr::get(
        mlir::spirv::Version::V_1_0,
        caps,
        exts,
        ctx);
    auto target_env = mlir::spirv::TargetEnvAttr::get(
        vce,
        mlir::spirv::getDefaultResourceLimits(ctx),
        mlir::spirv::ClientAPI::Vulkan,
        mlir::spirv::Vendor::Unknown,
        mlir::spirv::DeviceType::Unknown,
        mlir::spirv::TargetEnvAttr::kUnknownDeviceID);
    module->setAttr(mlir::spirv::getTargetEnvAttrName(), target_env);

    {
        mlir::spirv::TargetEnv target_env_obj(target_env);
        const bool supports_kernel = target_env_obj.allows(mlir::spirv::Capability::Kernel);
        mlir::spirv::MemorySpaceToStorageClassMap memory_space_map =
            supports_kernel ? mlir::spirv::mapMemorySpaceToOpenCLStorageClass
                            : mlir::spirv::mapMemorySpaceToVulkanStorageClass;
        mlir::spirv::MemorySpaceToStorageClassConverter converter(memory_space_map);
        mlir::spirv::convertMemRefTypesAndAttrs(module, converter);
    }
    if (spirv_debug) {
        llvm::errs() << "[GFX][MLIR] After SPIR-V memory space mapping:\n";
        module.dump();
    }

    {
        mlir::PassManager pm(ctx);
        if (!preserve_compact_memref_abi) {
            pm.addPass(mlir::memref::createNormalizeMemRefsPass());
        }
        pm.addPass(mlir::createCanonicalizerPass());
        pm.addPass(mlir::createCSEPass());
        pm.addPass(mlir::createReconcileUnrealizedCastsPass());
        if (mlir::failed(pm.run(module))) {
            if (log) {
            *log = "MLIR pre-SPIR-V memref normalization failed";
            }
            return {};
        }
        if (spirv_debug) {
            llvm::errs() << "[GFX][MLIR] After pre-SPIR-V memref normalization:\n";
            module.dump();
        }
    }

    const bool has_explicit_parallel_dispatch =
        static_cast<bool>(module->getAttrOfType<mlir::BoolAttr>("gfx.parallel_dispatch"));
    const bool use_gpu_dispatch =
        has_explicit_parallel_dispatch
            ? module->getAttrOfType<mlir::BoolAttr>("gfx.parallel_dispatch").getValue()
            : has_parallel_loops(module);
    if (!has_explicit_parallel_dispatch) {
        module->setAttr("gfx.parallel_dispatch", mlir::BoolAttr::get(ctx, use_gpu_dispatch));
    }
    if (use_gpu_dispatch) {
        int64_t loop_dims = 0;
        module.walk([&](mlir::scf::ParallelOp op) {
            if (loop_dims == 0) {
                loop_dims = static_cast<int64_t>(op.getNumLoops());
            }
        });
        if (loop_dims > 0 && !module->getAttr("gfx.parallel_loop_dims")) {
            module->setAttr("gfx.parallel_loop_dims",
                            mlir::IntegerAttr::get(mlir::IntegerType::get(ctx, 64), loop_dims));
        }
    }

    if (use_gpu_dispatch) {
        ctx->loadDialect<mlir::gpu::GPUDialect>();
        map_parallel_loops_to_blocks(module);

        {
            mlir::PassManager gpu_pm(ctx);
            gpu_pm.addPass(mlir::createConvertParallelLoopToGpuPass());
            if (mlir::failed(gpu_pm.run(module))) {
                if (log) {
                    std::string module_dump;
                    {
                        llvm::raw_string_ostream os(module_dump);
                        module.print(os);
                    }
                    *log = "MLIR SCF->GPU conversion failed\n" +
                           (diag_log.empty() ? std::string{} : ("[Diagnostics]\n" + diag_log)) +
                           "\n[MLIR module]\n" + module_dump;
                }
                return {};
            }
        }

        if (!entry_point.empty()) {
            module.walk([&](mlir::gpu::LaunchOp launch) {
                launch.setKernelFuncAttr(mlir::SymbolRefAttr::get(ctx, entry_point));
            });
        }

        module.walk([&](mlir::gpu::GPUModuleOp gpu_mod) {
            if (!gpu_mod->hasAttr(mlir::spirv::getTargetEnvAttrName())) {
                gpu_mod->setAttr(mlir::spirv::getTargetEnvAttrName(), target_env);
            }
        });

        {
            mlir::PassManager gpu_pm(ctx);
            gpu_pm.addPass(mlir::createGpuKernelOutliningPass());
            if (mlir::failed(gpu_pm.run(module))) {
                if (log) {
                    std::string module_dump;
                    {
                        llvm::raw_string_ostream os(module_dump);
                        module.print(os);
                    }
                    *log = "MLIR GPU->SPIR-V conversion failed\n" +
                           (diag_log.empty() ? std::string{} : ("[Diagnostics]\n" + diag_log)) +
                           "\n[MLIR module]\n" + module_dump;
                }
                return {};
            }
        }

        {
            mlir::PassManager affine_pm(ctx);
            affine_pm.addPass(mlir::createLowerAffinePass());
            affine_pm.addPass(mlir::createCanonicalizerPass());
            if (mlir::failed(affine_pm.run(module))) {
                if (log) {
                    std::string module_dump;
                    {
                        llvm::raw_string_ostream os(module_dump);
                        module.print(os);
                    }
                    *log = "MLIR affine lowering failed\n" +
                           (diag_log.empty() ? std::string{} : ("[Diagnostics]\n" + diag_log)) +
                           "\n[MLIR module]\n" + module_dump;
                }
                return {};
            }
        }

        annotate_kernel_scalar_args(module);

        module.walk([&](mlir::gpu::GPUFuncOp func) {
            if (func->hasAttr(mlir::spirv::getEntryPointABIAttrName())) {
                return;
            }
            llvm::SmallVector<int32_t, 3> local_size{1, 1, 1};
            if (auto known = func->getAttrOfType<mlir::ArrayAttr>("known_block_size")) {
                if (known.size() == 3) {
                    for (int i = 0; i < 3; ++i) {
                        if (auto iattr = mlir::dyn_cast<mlir::IntegerAttr>(known[i])) {
                            local_size[static_cast<size_t>(i)] = static_cast<int32_t>(iattr.getInt());
                        }
                    }
                }
            }
            auto abi = mlir::spirv::getEntryPointABIAttr(ctx, local_size);
            func->setAttr(mlir::spirv::getEntryPointABIAttrName(), abi);
        });

        {
            mlir::PassManager gpu_pm(ctx);
            gpu_pm.addPass(mlir::createCanonicalizerPass());
            gpu_pm.addPass(mlir::createCSEPass());
            if (mlir::failed(gpu_pm.run(module)) ||
                !convert_gpu_modules_to_spirv_with_math(module)) {
                if (log) {
                    std::string module_dump;
                    {
                        llvm::raw_string_ostream os(module_dump);
                        module.print(os);
                    }
                    *log = "MLIR GPU->SPIR-V conversion failed\n" +
                           (diag_log.empty() ? std::string{} : ("[Diagnostics]\n" + diag_log)) +
                           "\n[MLIR module]\n" + module_dump;
                }
                return {};
            }
        }
    } else {
        mlir::SPIRVConversionOptions conv_opts;
        if (needs_fp16) {
            conv_opts.emulateLT32BitScalarTypes = false;
        }
        mlir::SPIRVTypeConverter type_converter(target_env, conv_opts);
        if (spirv_debug) {
            module.walk([&](mlir::func::FuncOp func) {
                llvm::errs() << "[GFX][MLIR] SPIR-V type conversion check for " << func.getName() << ":\n";
                for (auto arg_type : func.getFunctionType().getInputs()) {
                    auto converted = type_converter.convertType(arg_type);
                    llvm::errs() << "  arg " << arg_type << " -> ";
                    if (converted) {
                        llvm::errs() << converted << "\n";
                    } else {
                        llvm::errs() << "<null>\n";
                    }
                    if (auto memref = mlir::dyn_cast<mlir::MemRefType>(arg_type)) {
                        int64_t offset = 0;
                        llvm::SmallVector<int64_t, 4> strides;
                        if (mlir::failed(memref.getStridesAndOffset(strides, offset))) {
                            llvm::errs() << "    strides/offset: <failed>\n";
                        } else {
                            llvm::errs() << "    strides/offset:";
                            for (auto stride : strides) {
                                llvm::errs() << " " << stride;
                            }
                            llvm::errs() << " | " << offset << "\n";
                        }
                    }
                }
                for (auto res_type : func.getFunctionType().getResults()) {
                    auto converted = type_converter.convertType(res_type);
                    llvm::errs() << "  res " << res_type << " -> ";
                    if (converted) {
                        llvm::errs() << converted << "\n";
                    } else {
                        llvm::errs() << "<null>\n";
                    }
                }
            });
        }

        {
            if (spirv_debug) {
                llvm::errs() << "[GFX][MLIR] Running SPIR-V conversion patterns\n";
            }
            mlir::ScfToSPIRVContext scf_to_spirv_ctx;
            mlir::RewritePatternSet patterns(ctx);
            populate_spirv_patterns(type_converter, scf_to_spirv_ctx, patterns);
            auto target = mlir::SPIRVConversionTarget::get(target_env);
            target->addIllegalDialect<mlir::arith::ArithDialect,
                                      mlir::func::FuncDialect,
                                      mlir::memref::MemRefDialect,
                                      mlir::scf::SCFDialect,
                                      mlir::tensor::TensorDialect,
                                      mlir::vector::VectorDialect,
                                      mlir::math::MathDialect,
                                      mlir::cf::ControlFlowDialect>();
            target->addLegalOp<mlir::UnrealizedConversionCastOp>();
            if (mlir::failed(mlir::applyPartialConversion(module.getOperation(),
                                                          *target,
                                                          std::move(patterns)))) {
                if (spirv_debug) {
                    llvm::errs() << "[GFX][MLIR] SPIR-V conversion failed\n";
                    module.dump();
                }
                if (log) {
                    std::string module_dump;
                    {
                        llvm::raw_string_ostream os(module_dump);
                        module.print(os);
                    }
                    *log = "MLIR SPIR-V conversion failed:\n" +
                           (diag_log.empty() ? std::string{} : diag_log) +
                           "\n[MLIR module]\n" + module_dump;
                }
                return {};
            }
            {
                mlir::PassManager post_pm(ctx);
                post_pm.addPass(mlir::createReconcileUnrealizedCastsPass());
                if (mlir::failed(post_pm.run(module))) {
                    if (log) {
                        *log = "MLIR SPIR-V post-conversion cleanup failed";
                    }
                    return {};
                }
            }
            if (spirv_debug) {
                llvm::errs() << "[GFX][MLIR] After SPIR-V pattern conversion:\n";
                module.dump();
            }
        }
    }

    eliminate_memref_cast_chains(module);
    if (!erase_noop_unrealized_casts(module, log)) {
        return {};
    }

    if (mlir::spirv::needsInterfaceVarABIAttrs(target_env)) {
        module.walk([&](mlir::spirv::FuncOp func) {
            for (unsigned i = 0; i < func.getNumArguments(); ++i) {
                if (func.getArgAttr(i, mlir::spirv::getInterfaceVarABIAttrName())) {
                    continue;
                }
                auto ptr_type = mlir::dyn_cast<mlir::spirv::PointerType>(func.getArgument(i).getType());
                if (!ptr_type || ptr_type.getStorageClass() == mlir::spirv::StorageClass::Function) {
                    continue;
                }
                func.setArgAttr(
                    i,
                    mlir::spirv::getInterfaceVarABIAttrName(),
                    mlir::spirv::getInterfaceVarABIAttr(
                        /*descriptorSet=*/0,
                        /*binding=*/i,
                        std::nullopt,
                        ctx));
            }
        });
    }

    auto spirv_module = find_spirv_module(module);
    if (spirv_module && !spirv_module->hasAttr(mlir::spirv::getTargetEnvAttrName())) {
        spirv_module->setAttr(mlir::spirv::getTargetEnvAttrName(), target_env);
    }
    if (!spirv_module) {
        auto addr_model = mlir::spirv::getAddressingModel(target_env, /*use64bitAddress=*/false);
        auto mem_model = mlir::spirv::getMemoryModel(target_env);
        if (mlir::failed(mem_model)) {
            if (log) {
                *log = "Failed to resolve SPIR-V memory model";
            }
            return {};
        }

        auto vce_attr = target_env.getTripleAttr();
        mlir::OpBuilder builder(module.getBodyRegion());
        spirv_module = builder.create<mlir::spirv::ModuleOp>(module.getLoc(),
                                                             addr_model,
                                                             *mem_model,
                                                             vce_attr);
        spirv_module->setAttr(mlir::spirv::getTargetEnvAttrName(), target_env);

        auto& src_ops = module.getBody()->getOperations();
        auto* dst_block = spirv_module.getBody();
        for (auto it = src_ops.begin(); it != src_ops.end();) {
            auto& op = *it++;
            if (llvm::isa<mlir::spirv::FuncOp,
                          mlir::spirv::GlobalVariableOp,
                          mlir::spirv::SpecConstantOp>(op)) {
                op.moveBefore(dst_block, dst_block->end());
            }
        }
    }

    if (spirv_module) {
        auto spv_func = spirv_module.lookupSymbol<mlir::spirv::FuncOp>(entry_point);
        if (spv_func && !spv_func->hasAttr(mlir::spirv::getEntryPointABIAttrName())) {
            auto abi = mlir::spirv::getEntryPointABIAttr(ctx, resolve_spirv_local_size(module));
            spv_func->setAttr(mlir::spirv::getEntryPointABIAttrName(), abi);
        }

        {
            std::string pre_abi_diag;
            mlir::ScopedDiagnosticHandler pre_abi_handler(ctx, [&](mlir::Diagnostic& diag) {
                llvm::raw_string_ostream os(pre_abi_diag);
                diag.print(os);
                os << "\n";
                return mlir::success();
            });
            mlir::PassManager pre_abi_pm(ctx);
            pre_abi_pm.addPass(mlir::createReconcileUnrealizedCastsPass());
            if (mlir::failed(pre_abi_pm.run(module))) {
                if (log) {
                    std::string module_dump;
                    {
                        llvm::raw_string_ostream os(module_dump);
                        module.print(os);
                    }
                    *log = "MLIR SPIR-V pre-ABI cleanup failed\n" +
                           (pre_abi_diag.empty() ? std::string{} : ("[Diagnostics]\n" + pre_abi_diag)) +
                           "\n[MLIR module]\n" + module_dump;
                }
                return {};
            }
        }

        {
            std::string abi_diag_log;
            mlir::ScopedDiagnosticHandler abi_diag(ctx, [&](mlir::Diagnostic& diag) {
                llvm::raw_string_ostream os(abi_diag_log);
                diag.print(os);
                os << "\n";
                return mlir::success();
            });
            mlir::PassManager abi_pm(ctx);
            auto& abi_nested = abi_pm.nest<mlir::spirv::ModuleOp>();
            abi_nested.addPass(mlir::spirv::createSPIRVLowerABIAttributesPass());
            if (mlir::failed(abi_pm.run(module))) {
                if (log) {
                    std::string module_dump;
                    {
                        llvm::raw_string_ostream os(module_dump);
                        module.print(os);
                    }
                    *log = "MLIR SPIR-V ABI lowering failed\n" +
                           (abi_diag_log.empty() ? std::string{} : ("[Diagnostics]\n" + abi_diag_log)) +
                           "\n[MLIR module]\n" + module_dump;
                }
                return {};
            }
        }
        {
            mlir::PassManager vce_pm(ctx);
            auto& vce_nested = vce_pm.nest<mlir::spirv::ModuleOp>();
            vce_nested.addPass(mlir::spirv::createSPIRVUpdateVCEPass());
            if (mlir::failed(vce_pm.run(module))) {
                if (log) {
                    *log = "MLIR SPIR-V VCE update failed";
                }
                return {};
            }
        }
    }

    if (spirv_debug) {
        llvm::errs() << "[GFX][MLIR] SPIR-V lowering output module:\n";
        module.dump();
    }
    spirv_module = find_spirv_module(module);
    if (!spirv_module) {
        if (log) {
            *log = "No spirv.module found after lowering";
        }
        return {};
    }

    ensure_spirv_entry_point_interface(spirv_module);
    apply_spirv_local_size(module, spirv_module, entry_point);
    if (!validate_spirv_module(spirv_module, log)) {
        return {};
    }
    dump_spirv_mlir_if_requested(spirv_module, entry_point);

    return serialize_spirv(spirv_module, log);
}

std::vector<uint32_t> lower_to_spirv(mlir::ModuleOp module,
                                     const std::string& entry_point,
                                     std::string* log) {
    if (!module) {
        if (log) {
            *log = "MLIR module is null";
        }
        return {};
    }

    std::string module_text;
    {
        llvm::raw_string_ostream os(module_text);
        module.print(os);
    }

    std::string primary_log;
    bool prefer_parallel = false;
    if (auto attr = module->getAttrOfType<mlir::BoolAttr>("gfx.prefer_parallel")) {
        prefer_parallel = attr.getValue();
    }
    auto spirv = lower_to_spirv_impl(module,
                                     entry_point,
                                     &primary_log,
                                     /*use_alloca=*/false,
                                     /*use_parallel_loops=*/prefer_parallel);
    if (!spirv.empty()) {
        if (log) {
            *log = primary_log;
        }
        return spirv;
    }

    if (primary_log.find("Unresolved unrealized_conversion_cast") == std::string::npos) {
        if (log) {
            *log = primary_log;
        }
        return {};
    }

    auto parsed = mlir::parseSourceString<mlir::ModuleOp>(module_text, module.getContext());
    if (!parsed) {
        if (log) {
            *log = primary_log + "\nFallback parse failed";
        }
        return {};
    }

    std::string fallback_log;
    auto fallback = lower_to_spirv_impl(*parsed,
                                        entry_point,
                                        &fallback_log,
                                        /*use_alloca=*/true,
                                        /*use_parallel_loops=*/false);
    if (!fallback.empty()) {
        if (log) {
            *log = fallback_log;
        }
        return fallback;
    }

    if (log) {
        *log = primary_log + "\nFallback (alloca) failed:\n" + fallback_log;
    }
    return {};
}

}  // namespace gfx_plugin
}  // namespace ov
