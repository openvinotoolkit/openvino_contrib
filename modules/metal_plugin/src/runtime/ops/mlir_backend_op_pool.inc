// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//
// Included from mlir_backend.mm. Handles MaxPool2D / AvgPool2D stand-alone cases.

bool MlirBackend::Impl::try_build_pool(const std::shared_ptr<const ov::Model>& model,
                                       const ModelAnalysis& analysis,
                                       MetalKernelCompiler& compiler,
                                       std::string& log) {
    if (analysis.has_maxpool && !analysis.has_unary && !analysis.has_softmax && !analysis.has_avgpool &&
        !analysis.has_matmul && !analysis.has_add && !analysis.has_add_broadcast && !analysis.has_conv2d) {
        try {
            mlir::MLIRContext ctx;
            auto module = build_mlir_maxpool_from_model(model, ctx);
            run_mlir_pipeline(module);
        } catch (const std::exception& e) {
            METAL_LOGSTREAM_TRACE("mlir") << "[MlirBackend] MaxPool MLIR build failed, fallback: " << e.what() << "\n";
            m_force_fallback = true;
            ensure_fallback(m_original_model);
            return true;
        }

        MetalKernelIR ir = build_kernel_ir_for_maxpool(model);
        m_ir = std::move(ir);
        m_ops = m_ir.ops;
        m_segments.clear();
        if (!m_ops.empty()) m_segments.push_back(Segment{0, m_ops.size()});
        m_ops_from_flat_segment = true;
        m_pipelines.clear();
        if (use_handwritten_msl()) {
            METAL_LOG_DEBUG("mlir", "[METAL MLIR] MaxPool: OV_METAL_USE_HANDWRITTEN_MSL=1, using fallback kernel");
            m_pipelines.push_back(compiler.compile_maxpool2d_kernel(m_ops[0], log));
        } else {
            try {
                mlir::MLIRContext ctx;
                auto module = build_mlir_maxpool_from_model(model, ctx);
                run_mlir_pipeline(module);
                Pool2DCodegenDesc desc;
                const auto& kop = m_ops[0];
                desc.kind = KernelOpKind::MaxPool2D;
                desc.N = kop.pool.N;
                desc.C = kop.pool.C;
                desc.H = kop.pool.H;
                desc.W = kop.pool.W;
                desc.kH = kop.pool.kernelH;
                desc.kW = kop.pool.kernelW;
                desc.strideH = kop.pool.strideH;
                desc.strideW = kop.pool.strideW;
                desc.padTop = kop.pool.padTop;
                desc.padLeft = kop.pool.padLeft;
                desc.padBottom = 0;
                desc.padRight = 0;
                desc.outH = kop.pool.outH;
                desc.outW = kop.pool.outW;
                desc.is_avg = false;
                desc.exclude_pad = kop.pool.exclude_pad;
                auto source = generate_msl_from_mlir(module, desc);
                m_pipelines.push_back(compiler.compile_msl_from_source(source, "pool2d_kernel", log));
            } catch (const std::exception& e) {
                METAL_LOG_DEBUG("mlir", std::string("[METAL MLIR] MaxPool MLIR→MSL failed, fallback: ") + e.what());
                m_pipelines.push_back(compiler.compile_maxpool2d_kernel(m_ops[0], log));
            }
        }
        return true;
    }

    if (analysis.has_avgpool && !analysis.has_unary && !analysis.has_softmax && !analysis.has_maxpool &&
        !analysis.has_matmul && !analysis.has_add && !analysis.has_add_broadcast && !analysis.has_conv2d) {
        try {
            mlir::MLIRContext ctx;
            auto module = build_mlir_avgpool_from_model(model, ctx);
            run_mlir_pipeline(module);
        } catch (const std::exception& e) {
            METAL_LOGSTREAM_TRACE("mlir") << "[MlirBackend] AvgPool MLIR build failed, fallback: " << e.what() << "\n";
            m_force_fallback = true;
            ensure_fallback(m_original_model);
            return true;
        }

        MetalKernelIR ir = build_kernel_ir_for_avgpool(model);
        m_ir = std::move(ir);
        m_ops = m_ir.ops;
        m_segments.clear();
        if (!m_ops.empty()) m_segments.push_back(Segment{0, m_ops.size()});
        m_ops_from_flat_segment = true;
        m_pipelines.clear();
        if (use_handwritten_msl()) {
            METAL_LOG_DEBUG("mlir", "[METAL MLIR] AvgPool: OV_METAL_USE_HANDWRITTEN_MSL=1, using fallback kernel");
            m_pipelines.push_back(compiler.compile_avgpool2d_kernel(m_ops[0], log));
        } else {
            try {
                mlir::MLIRContext ctx;
                auto module = build_mlir_avgpool_from_model(model, ctx);
                run_mlir_pipeline(module);
                Pool2DCodegenDesc desc;
                const auto& kop = m_ops[0];
                desc.kind = KernelOpKind::AvgPool2D;
                desc.N = kop.pool.N;
                desc.C = kop.pool.C;
                desc.H = kop.pool.H;
                desc.W = kop.pool.W;
                desc.kH = kop.pool.kernelH;
                desc.kW = kop.pool.kernelW;
                desc.strideH = kop.pool.strideH;
                desc.strideW = kop.pool.strideW;
                desc.padTop = kop.pool.padTop;
                desc.padLeft = kop.pool.padLeft;
                desc.padBottom = 0;
                desc.padRight = 0;
                desc.outH = kop.pool.outH;
                desc.outW = kop.pool.outW;
                desc.is_avg = true;
                desc.exclude_pad = kop.pool.exclude_pad;
                auto source = generate_msl_from_mlir(module, desc);
                m_pipelines.push_back(compiler.compile_msl_from_source(source, "pool2d_kernel", log));
            } catch (const std::exception& e) {
                METAL_LOG_DEBUG("mlir", std::string("[METAL MLIR] AvgPool MLIR→MSL failed, fallback: ") + e.what());
                m_pipelines.push_back(compiler.compile_avgpool2d_kernel(m_ops[0], log));
            }
        }
        return true;
    }

    return false;
}

