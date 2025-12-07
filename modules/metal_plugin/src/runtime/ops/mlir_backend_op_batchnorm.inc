// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//
// Included from mlir_backend.mm. Handles BatchNorm2D standalone and Conv2D+BN+Unary fused pattern.

bool MlirBackend::Impl::try_build_batchnorm(const std::shared_ptr<const ov::Model>& model,
                                            const ModelAnalysis& analysis,
                                            MetalKernelCompiler& compiler,
                                            std::string& log) {
    const bool has_batchnorm = analysis.has_batchnorm;
    if (!has_batchnorm)
        return false;

    // Conv2D + BatchNorm + Unary is already handled inside try_build_conv; avoid double-handling.
    if (analysis.has_conv2d && analysis.has_unary && analysis.compute_ops == 3)
        return false;

    if (!analysis.has_matmul && !analysis.has_add && !analysis.has_add_broadcast && !analysis.has_unary &&
        !analysis.has_softmax && !analysis.has_maxpool && !analysis.has_avgpool && !analysis.has_conv2d) {
        bool const_params = false;
        try {
            mlir::MLIRContext ctx;
            auto module = build_mlir_batchnorm_from_model(model, ctx);
            run_mlir_pipeline(module);
        } catch (const std::exception& e) {
            METAL_LOGSTREAM_TRACE("mlir") << "[MlirBackend] BatchNorm MLIR build failed, fallback: " << e.what() << "\n";
            m_force_fallback = true;
            ensure_fallback(m_original_model);
            return true;
        }
        MetalKernelIR ir = build_kernel_ir_for_batchnorm(model, const_params);
        if (!const_params) {
            m_force_fallback = true;
            ensure_fallback(m_original_model);
            return true;
        }
        if (!ir.ops.empty()) {
            m_const_bn = ir.ops[0].bn_params;
        }
        m_ir = std::move(ir);
        m_ops = m_ir.ops;
        m_segments.clear();
        if (!m_ops.empty()) m_segments.push_back(Segment{0, m_ops.size()});
        m_ops_from_flat_segment = true;
        m_pipelines.clear();
        m_pipelines.push_back(compiler.compile_batchnorm2d_kernel(m_ops[0], log));
        return true;
    }

    return false;
}

