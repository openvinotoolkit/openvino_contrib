// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "mlir/spirv_codegen.hpp"

#include "mlir/Conversion/ArithToSPIRV/ArithToSPIRV.h"
#include "mlir/Conversion/ControlFlowToSPIRV/ControlFlowToSPIRV.h"
#include "mlir/Conversion/FuncToSPIRV/FuncToSPIRV.h"
#include "mlir/Conversion/FuncToSPIRV/FuncToSPIRVPass.h"
#include "mlir/Conversion/GPUToSPIRV/GPUToSPIRV.h"
#include "mlir/Conversion/IndexToSPIRV/IndexToSPIRV.h"
#include "mlir/Conversion/MathToSPIRV/MathToSPIRV.h"
#include "mlir/Conversion/MemRefToSPIRV/MemRefToSPIRV.h"
#include "mlir/Conversion/MemRefToSPIRV/MemRefToSPIRVPass.h"
#include "mlir/Conversion/Passes.h"
#include "mlir/Conversion/SCFToSPIRV/SCFToSPIRV.h"
#include "mlir/Conversion/TensorToSPIRV/TensorToSPIRV.h"
#include "mlir/Conversion/UBToSPIRV/UBToSPIRV.h"
#include "mlir/Conversion/VectorToSPIRV/VectorToSPIRV.h"
#include "mlir/Conversion/VectorToSPIRV/VectorToSPIRVPass.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Math/IR/Math.h"
#include "mlir/Dialect/Arith/Transforms/Passes.h"
#include "mlir/Dialect/Arith/Transforms/BufferizableOpInterfaceImpl.h"
#include "mlir/Dialect/Bufferization/IR/Bufferization.h"
#include "mlir/Dialect/Bufferization/Transforms/FuncBufferizableOpInterfaceImpl.h"
#include "mlir/Dialect/Bufferization/Transforms/Passes.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/GPU/IR/GPUDialect.h"
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
#include "mlir/Pass/PassManager.h"
#include "mlir/Target/SPIRV/Serialization.h"
#include "mlir/Parser/Parser.h"
#include "mlir/Transforms/DialectConversion.h"
#include "mlir/Transforms/Passes.h"

#include "llvm/ADT/SmallVector.h"

#include "mlir/mlir_passes.hpp"
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

std::vector<uint32_t> lower_to_spirv(mlir::ModuleOp module,
                                     const std::string& entry_point,
                                     std::string* log) {
    if (!module) {
        if (log) {
            *log = "MLIR module is null";
        }
        return {};
    }

    auto* ctx = module.getContext();
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

    const auto env_verify_dump = std::getenv("GFX_MLIR_DEBUG");
    if (env_verify_dump && std::string(env_verify_dump) != "0") {
        llvm::errs() << "[GFX][MLIR] Pre-verify module:\n";
        module.dump();
    }
    if (mlir::failed(mlir::verify(module))) {
        if (log) {
            *log = "MLIR module verification failed";
        }
        return {};
    }

    try {
        run_mlir_pipeline(module);
    } catch (const std::exception& e) {
        if (log) {
            *log = std::string("MLIR preprocessing failed: ") + e.what();
        }
        return {};
    }

    const auto env_spirv_debug = std::getenv("GFX_MLIR_SPIRV_DEBUG");
    const bool spirv_debug = env_spirv_debug && std::string(env_spirv_debug) != "0";

    if (spirv_debug) {
        ctx->disableMultithreading();
        llvm::errs() << "[GFX][MLIR] SPIR-V lowering input module:\n";
        module.dump();
    }

    bool needs_fp16 = false;
    bool needs_int64 = false;
    auto check_type = [&](mlir::Type t) {
        if (auto shaped = mlir::dyn_cast<mlir::ShapedType>(t)) {
            t = shaped.getElementType();
        }
        if (mlir::isa<mlir::Float16Type>(t)) {
            needs_fp16 = true;
        }
        if (auto int_ty = mlir::dyn_cast<mlir::IntegerType>(t)) {
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
        pm.addPass(mlir::memref::createNormalizeMemRefsPass());
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
        target->addLegalOp<mlir::UnrealizedConversionCastOp>();
        if (mlir::failed(mlir::applyPartialConversion(module.getOperation(),
                                                      *target,
                                                      std::move(patterns)))) {
            if (spirv_debug) {
                llvm::errs() << "[GFX][MLIR] SPIR-V conversion failed\n";
                module.dump();
            }
            if (log) {
                *log = "MLIR SPIR-V conversion failed";
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

    if (mlir::spirv::needsInterfaceVarABIAttrs(target_env)) {
        module.walk([&](mlir::spirv::FuncOp func) {
            for (unsigned i = 0; i < func.getNumArguments(); ++i) {
                if (func.getArgAttr(i, mlir::spirv::getInterfaceVarABIAttrName())) {
                    continue;
                }
                auto ptr_type = mlir::dyn_cast<mlir::spirv::PointerType>(func.getArgument(i).getType());
                if (!ptr_type || ptr_type.getStorageClass() != mlir::spirv::StorageClass::StorageBuffer) {
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
            auto abi = mlir::spirv::getEntryPointABIAttr(ctx, {1, 1, 1});
            spv_func->setAttr(mlir::spirv::getEntryPointABIAttrName(), abi);
        }

        mlir::PassManager pm(ctx);
        auto& spirv_pm = pm.nest<mlir::spirv::ModuleOp>();
        spirv_pm.addPass(mlir::spirv::createSPIRVLowerABIAttributesPass());
        spirv_pm.addPass(mlir::spirv::createSPIRVUpdateVCEPass());
        if (mlir::failed(pm.run(module))) {
            if (log) {
                *log = "MLIR SPIR-V post-processing failed";
            }
            return {};
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

    return serialize_spirv(spirv_module, log);
}

}  // namespace gfx_plugin
}  // namespace ov
