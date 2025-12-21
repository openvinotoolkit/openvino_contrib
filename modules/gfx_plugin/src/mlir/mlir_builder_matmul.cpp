// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "mlir/mlir_builder.hpp"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Math/IR/Math.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/MLIRContext.h"

#include "openvino/core/except.hpp"
#include "openvino/core/model.hpp"
#include "openvino/op/matmul.hpp"

namespace ov {
namespace gfx_plugin {

namespace {
std::shared_ptr<const ov::op::v0::MatMul> find_single_matmul(const std::shared_ptr<const ov::Model>& model) {
    std::shared_ptr<const ov::op::v0::MatMul> matmul;
    for (const auto& node : model->get_ordered_ops()) {
        if (auto mm = ov::as_type_ptr<const ov::op::v0::MatMul>(node)) {
            OPENVINO_ASSERT(!matmul, "Only single MatMul is supported in MLIR path for now");
            matmul = mm;
        }
    }
    OPENVINO_ASSERT(matmul, "MLIR MatMul builder: MatMul op not found");
    return matmul;
}

std::vector<int64_t> flatten_to_3d(const ov::Shape& s) {
    OPENVINO_ASSERT(!s.empty(), "MatMul: empty shape");
    OPENVINO_ASSERT(s.size() >= 2 && s.size() <= 4, "MatMul supports ranks 2–4");
    int64_t batch = 1;
    for (size_t i = 0; i + 2 < s.size(); ++i)
        batch *= static_cast<int64_t>(s[i]);
    int64_t m = static_cast<int64_t>(s[s.size() - 2]);
    int64_t k = static_cast<int64_t>(s[s.size() - 1]);
    return {batch, m, k};
}
}  // namespace

mlir::ModuleOp build_mlir_module_from_model(const std::shared_ptr<const ov::Model>& model,
                                            mlir::MLIRContext& ctx) {
    ctx.loadDialect<mlir::func::FuncDialect, mlir::linalg::LinalgDialect, mlir::tensor::TensorDialect,
                    mlir::arith::ArithDialect, mlir::math::MathDialect>();

    auto matmul = find_single_matmul(model);
    const auto shape_a = matmul->get_input_shape(0);
    const auto shape_b = matmul->get_input_shape(1);
    OPENVINO_ASSERT(!shape_a.empty() && !shape_b.empty(), "MatMul: shapes required");
    OPENVINO_ASSERT(shape_a.size() >= 2 && shape_a.size() <= 4, "MatMul supports ranks 2–4");
    OPENVINO_ASSERT(shape_b.size() >= 2 && shape_b.size() <= 4, "MatMul supports ranks 2–4");

    const bool ta = matmul->get_transpose_a();
    const bool tb = matmul->get_transpose_b();
    auto a3 = flatten_to_3d(shape_a);
    auto b3 = flatten_to_3d(shape_b);
    const int64_t M = ta ? a3[2] : a3[1];
    const int64_t K_a = ta ? a3[1] : a3[2];
    int64_t K_b = tb ? b3[2] : b3[1];
    int64_t N = tb ? b3[1] : b3[2];
    if (!tb && K_b != K_a && b3[2] == K_a) {
        // Pre-transposed weights [N, K] case.
        K_b = b3[2];
        N = b3[1];
    }
    OPENVINO_ASSERT(K_a == K_b, "MatMul K dimension mismatch");
    const int64_t K = K_a;

    auto f32 = mlir::Float32Type::get(&ctx);

    auto type_a = mlir::RankedTensorType::get({M, K}, f32);
    auto type_b = mlir::RankedTensorType::get({K, N}, f32);
    auto type_c = mlir::RankedTensorType::get({M, N}, f32);

    mlir::OpBuilder module_builder(&ctx);
    auto module = mlir::ModuleOp::create(mlir::UnknownLoc::get(&ctx));
    module_builder.setInsertionPointToStart(module.getBody());

    auto func_type = module_builder.getFunctionType({type_a, type_b}, {type_c});
    auto func = module_builder.create<mlir::func::FuncOp>(mlir::UnknownLoc::get(&ctx), "matmul_main", func_type);
    func.addEntryBlock();

    mlir::OpBuilder b(func.getBody());
    b.setInsertionPointToStart(&func.getBody().front());
    auto empty = b.create<mlir::tensor::EmptyOp>(mlir::UnknownLoc::get(&ctx), mlir::ArrayRef<int64_t>({M, N}), f32);
    auto mm = b.create<mlir::linalg::MatmulOp>(mlir::UnknownLoc::get(&ctx),
                                               mlir::ValueRange{func.getArgument(0), func.getArgument(1)},
                                               mlir::ValueRange{empty});
    b.create<mlir::func::ReturnOp>(mlir::UnknownLoc::get(&ctx), mm.getResults());

    return module;
}

}  // namespace gfx_plugin
}  // namespace ov
