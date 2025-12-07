// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//
// Included from mlir_backend.mm to keep access to MlirBackend::Impl internals.

bool MlirBackend::Impl::try_build_matmul(const std::shared_ptr<const ov::Model>& model,
                                         const ModelAnalysis& analysis,
                                         MetalKernelCompiler& compiler,
                                         std::string& log) {
    if (!analysis.has_matmul)
        return false;

    bool built_from_mlir = false;
    if (analysis.compute_ops == 1) {
        METAL_LOG_DEBUG("mlir", "[MlirBackend compile] single MatMul path");
        try {
            mlir::MLIRContext ctx;
            auto module = build_mlir_module_from_model(model, ctx);
            run_mlir_pipeline(module);

            int64_t M = 0, N = 0, K = 0;
            extract_matmul_shape(module, M, N, K);
            OPENVINO_ASSERT(M > 0 && N > 0 && K > 0, "MlirBackend: invalid MatMul dims");

            MetalKernelIR ir;
            ir.tensors.push_back({"a", {M, K}});
            ir.tensors.push_back({"b", {K, N}});
            ir.tensors.push_back({"c", {M, N}});

            KernelOp op{};
            op.kind = KernelOpKind::MatMul;
            op.input0 = &ir.tensors[0];
            op.input1 = &ir.tensors[1];
            op.output = &ir.tensors[2];
            op.M = M;
            op.N = N;
            op.K = K;
            op.batch = 1;
            op.batch_a = 1;
            op.batch_b = 1;
            op.b_is_nk_layout = false;
            ir.ops.push_back(op);

            m_ir = std::move(ir);
            m_ops = m_ir.ops;
            m_segments.clear();
            if (!m_ops.empty()) {
                m_segments.push_back(Segment{0, m_ops.size()});
            }
            m_ops_from_flat_segment = false;
            m_pipelines.clear();
            MatMulCodegenDesc desc;
            const auto& kop = m_ops[0];
            desc.M = kop.M;
            desc.N = kop.N;
            desc.K = kop.K;
            desc.batch = kop.batch;
            desc.batch_a = kop.batch_a;
            desc.batch_b = kop.batch_b;
            desc.a_transpose = kop.a_transpose;
            desc.b_transpose = kop.b_transpose;
            desc.b_is_nk_layout = kop.b_is_nk_layout;
            auto source = generate_msl_from_mlir(module, desc);
            m_pipelines.push_back(compiler.compile_msl_from_source(source, "matmul_kernel", log));
            built_from_mlir = true;
        } catch (const std::exception&) {
            METAL_LOG_DEBUG("mlir", "[MlirBackend compile] MLIR MatMul path failed, fallback to manual MatMul");
            // fall through to manual MatMul handling (no Add fallback)
        }
    }
    if (built_from_mlir) {
        return true;
    }

    if (analysis.compute_ops == 1) {
        // Manual shape extraction for MatMul (including simple batch broadcast)
        MetalKernelIR ir = build_kernel_ir_for_matmul(model);
        m_ir = std::move(ir);
        m_ops = m_ir.ops;
        m_segments.clear();
        if (!m_ops.empty()) {
            m_segments.push_back(Segment{0, m_ops.size()});
        }
        m_ops_from_flat_segment = false;
        m_pipelines.clear();
        m_pipelines.push_back(compiler.compile_matmul_kernel(m_ops[0], log));
        METAL_LOG_DEBUG("mlir",
                        "MlirBackend MatMul: M=" + std::to_string(m_ops[0].M) + " N=" + std::to_string(m_ops[0].N) +
                            " K=" + std::to_string(m_ops[0].K) + " batch=" + std::to_string(m_ops[0].batch) +
                            " batch_a=" + std::to_string(m_ops[0].batch_a) +
                            " batch_b=" + std::to_string(m_ops[0].batch_b));
        return true;
    }

    return false;
}

