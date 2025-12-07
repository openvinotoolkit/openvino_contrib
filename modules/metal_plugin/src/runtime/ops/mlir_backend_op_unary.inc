// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//
// Included from mlir_backend.mm. Handles stand-alone Unary (activation) cases.

bool MlirBackend::Impl::try_build_unary(const std::shared_ptr<const ov::Model>& model,
                                        const ModelAnalysis& analysis,
                                        MetalKernelCompiler& compiler,
                                        std::string& log) {
    if (!(analysis.has_unary && analysis.compute_ops == 1))
        return false;
    // reject if mixed with other ops
    if (analysis.has_matmul || analysis.has_softmax || analysis.has_conv2d || analysis.has_conv3d ||
        analysis.has_add || analysis.has_add_broadcast || analysis.has_mul || analysis.has_div ||
        analysis.has_mod || analysis.has_floor_mod || analysis.has_pow || analysis.has_batchnorm ||
        analysis.has_maxpool || analysis.has_avgpool)
        return false;

    try {
        mlir::MLIRContext ctx;
        auto op_node = model->get_ordered_ops().back();
        auto module = build_mlir_unary_from_node(op_node, ctx, analysis.unary_kind, analysis.unary_alpha);
        run_mlir_pipeline(module);
    } catch (const std::exception& e) {
        METAL_LOGSTREAM_TRACE("mlir") << "[MlirBackend] Unary MLIR build failed, fallback: " << e.what() << "\n";
        m_force_fallback = true;
        ensure_fallback(m_original_model);
        return true;
    }

    // Build kernel IR directly from node
    const std::shared_ptr<const ov::Node> unary_node = [&]() -> std::shared_ptr<const ov::Node> {
        for (const auto& node : model->get_ordered_ops()) {
            if (!ov::is_type<ov::op::v0::Parameter>(node.get()) &&
                !ov::is_type<ov::op::v0::Constant>(node.get()) &&
                !ov::is_type<ov::op::v0::Result>(node.get())) {
                return node;
            }
        }
        return nullptr;
    }();
    if (!unary_node) {
        m_force_fallback = true;
        ensure_fallback(m_original_model);
        return true;
    }
    MetalKernelIR ir = build_kernel_ir_for_unary(unary_node, analysis.unary_kind, analysis.unary_alpha);
    m_ir = std::move(ir);
    m_ops = m_ir.ops;
    m_segments.clear();
    if (!m_ops.empty()) m_segments.push_back(Segment{0, m_ops.size()});
    m_ops_from_flat_segment = true;
    m_pipelines.clear();
    m_pipelines.push_back(compiler.compile_unary_kernel(m_ops[0], log));
    return true;
}

