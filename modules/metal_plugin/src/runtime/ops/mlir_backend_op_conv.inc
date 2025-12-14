// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//
// Included from mlir_backend.mm to access Impl internals. Handles Conv2D/Conv3D paths
// (solo, with Unary, with BatchNorm+Unary).

bool MlirBackend::Impl::try_build_conv(const std::shared_ptr<const ov::Model>& model,
                                       const ModelAnalysis& analysis,
                                       MetalKernelCompiler& compiler,
                                       std::string& log) {
    const bool has_conv2d = analysis.has_conv2d;
    const bool has_conv3d = analysis.has_conv3d;
    if (!has_conv2d && !has_conv3d)
        return false;

    auto load_conv_weights_if_const = [&](const std::shared_ptr<const ov::Node>& conv_node) {
        if (!conv_node) return;
        if (auto c = std::dynamic_pointer_cast<const ov::op::v0::Constant>(conv_node->get_input_node_shared_ptr(1))) {
            m_const_w = c->cast_vector<float>();
            m_has_const_w = true;
        }
    };

    // Conv3D solo
    if (has_conv3d && !analysis.has_unary && !analysis.has_softmax && !analysis.has_batchnorm &&
        !analysis.has_add && !analysis.has_add_broadcast && !analysis.has_mul && !analysis.has_div &&
        !analysis.has_pow && !analysis.has_mod && !analysis.has_floor_mod && !analysis.has_split) {
        bool const_w = false;
        MetalKernelIR ir = build_kernel_ir_for_conv3d(model, const_w);
        m_ir = std::move(ir);
        m_ops = m_ir.ops;
        m_segments.clear();
        if (!m_ops.empty()) {
            m_segments.push_back(Segment{0, m_ops.size()});
        }
        m_ops_from_flat_segment = true;
        METAL_LOG_DEBUG("mlir", "[METAL MLIR] Conv3D selected: input " + describe_shape(m_ir.ops[0].input0) +
                                  " weights " + describe_shape(m_ir.ops[0].input1) +
                                  " output " + describe_shape(m_ir.ops[0].output));
        if (const_w && !m_ops.empty()) {
            std::shared_ptr<const ov::Node> conv_node;
            for (const auto& node : model->get_ordered_ops()) {
                if (auto c = ov::as_type_ptr<const ov::op::v1::Convolution>(node)) {
                    if (c->get_input_shape(0).size() == 5) { conv_node = c; break; }
                }
            }
            load_conv_weights_if_const(conv_node);
            if (m_has_const_w)
                METAL_LOG_DEBUG("mlir", "[METAL MLIR] Conv3D const weights size=" + std::to_string(m_const_w.size()));
        }
        m_pipelines.clear();
        try {
            mlir::MLIRContext ctx;
            auto module = build_mlir_conv3d_from_model(model, ctx);
            run_mlir_pipeline(module);
            Conv3DCodegenDesc desc;
            const auto& kop = m_ops[0];
            desc.kind = KernelOpKind::Conv3D;
            desc.N = kop.conv3d.N;
            desc.C_in = kop.conv3d.C_in;
            desc.D = kop.conv3d.D;
            desc.H = kop.conv3d.H;
            desc.W = kop.conv3d.W;
            desc.C_out = kop.conv3d.C_out;
            desc.kD = kop.conv3d.kernelD;
            desc.kH = kop.conv3d.kernelH;
            desc.kW = kop.conv3d.kernelW;
            desc.strideD = kop.conv3d.strideD;
            desc.strideH = kop.conv3d.strideH;
            desc.strideW = kop.conv3d.strideW;
            desc.dilationD = kop.conv3d.dilationD;
            desc.dilationH = kop.conv3d.dilationH;
            desc.dilationW = kop.conv3d.dilationW;
            desc.padFront = kop.conv3d.padFront;
            desc.padTop = kop.conv3d.padTop;
            desc.padLeft = kop.conv3d.padLeft;
            desc.padBack = kop.conv3d.padBack;
            desc.padBottom = kop.conv3d.padBottom;
            desc.padRight = kop.conv3d.padRight;
            desc.outD = kop.conv3d.outD;
            desc.outH = kop.conv3d.outH;
            desc.outW = kop.conv3d.outW;
            auto source = generate_msl_from_mlir(module, desc);
            m_pipelines.push_back(compiler.compile_msl_from_source(source, "conv3d_kernel", log));
        } catch (const std::exception& e) {
            METAL_LOG_DEBUG("mlir", std::string("[METAL MLIR] Conv3D MLIR→MSL failed, fallback to CPU: ") + e.what());
            m_force_fallback = true;
            ensure_fallback(m_original_model);
            return true;
        }
        return true;
    }

    // Conv2D + Unary
    if (has_conv2d && analysis.has_unary && analysis.compute_ops == 2) {
        bool const_w = false;
        try {
            mlir::MLIRContext ctx;
            auto module = build_mlir_conv2d_from_model(model, ctx);
            run_mlir_pipeline(module);
        } catch (const std::exception& e) {
            METAL_LOGSTREAM_TRACE("mlir") << "[MlirBackend] Conv2D+Unary MLIR build failed, fallback: " << e.what() << "\n";
            m_force_fallback = true;
            ensure_fallback(m_original_model);
            return true;
        }
        MetalKernelIR ir = build_kernel_ir_for_conv2d(model, const_w);
        KernelOp unary_op;
        unary_op.kind = KernelOpKind::Unary;
        unary_op.activation = analysis.unary_kind;
        unary_op.alpha = analysis.unary_alpha;
        ir.ops.push_back(unary_op);
        ir.ops[1].input0 = ir.ops[0].output;
        ir.ops[1].output = ir.ops[0].output;
        m_ir = std::move(ir);
        m_ops = m_ir.ops;
        m_segments.clear();
        if (!m_ops.empty()) {
            m_segments.push_back(Segment{0, m_ops.size()});
        }
        m_ops_from_flat_segment = true;

        if (const_w && !m_ops.empty()) {
            std::shared_ptr<const ov::Node> conv_node;
            for (const auto& node : model->get_ordered_ops()) {
                if (auto c = ov::as_type_ptr<const ov::op::v1::Convolution>(node)) { conv_node = c; break; }
                if (auto g = ov::as_type_ptr<const ov::op::v1::GroupConvolution>(node)) { conv_node = g; break; }
            }
            load_conv_weights_if_const(conv_node);
        }

        m_pipelines.clear();
        m_pipelines.push_back(compiler.compile_conv2d_kernel(m_ops[0], log));
        m_pipelines.push_back(compiler.compile_unary_kernel(m_ops[1], log));
        return true;
    }

    // Conv2D + BatchNorm + Unary
    if (has_conv2d && analysis.has_unary && analysis.has_batchnorm && analysis.compute_ops == 3) {
        bool const_w = false;
        bool const_bn = false;
        try {
            mlir::MLIRContext ctx;
            auto module = build_mlir_conv2d_from_model(model, ctx);
            run_mlir_pipeline(module);
            auto module_bn = build_mlir_batchnorm_from_model(model, ctx);
            run_mlir_pipeline(module_bn);
        } catch (const std::exception& e) {
            METAL_LOGSTREAM_TRACE("mlir") << "[MlirBackend] Conv2D+BN+Unary MLIR build failed, fallback: " << e.what() << "\n";
            m_force_fallback = true;
            ensure_fallback(m_original_model);
            return true;
        }

        MetalKernelIR conv_ir = build_kernel_ir_for_conv2d(model, const_w);
        MetalKernelIR bn_ir = build_kernel_ir_for_batchnorm(model, const_bn);
        if (!const_bn) {
            m_force_fallback = true;
            ensure_fallback(m_original_model);
            return true;
        }

        MetalKernelIR ir;
        ir.tensors.push_back(conv_ir.tensors[0]);  // input
        KernelTensor tmp1 = {"tmp1", conv_ir.ops[0].output ? conv_ir.ops[0].output->shape : conv_ir.tensors[0].shape};
        KernelTensor tmp2 = {"tmp2", bn_ir.ops[0].output ? bn_ir.ops[0].output->shape : conv_ir.tensors[0].shape};
        ir.tensors.push_back(tmp1);
        ir.tensors.push_back(tmp2);

        auto conv_op = conv_ir.ops[0];
        conv_op.input0 = &ir.tensors[0];
        conv_op.input1 = conv_ir.ops[0].input1;  // weight tensor remains owned by conv_ir
        conv_op.output = &ir.tensors[1];

        auto bn_op = bn_ir.ops[0];
        bn_op.input0 = conv_op.output;
        bn_op.output = &ir.tensors[2];

        KernelOp unary_op;
        unary_op.kind = KernelOpKind::Unary;
        unary_op.activation = analysis.unary_kind;
        unary_op.alpha = analysis.unary_alpha;
        unary_op.input0 = bn_op.output;
        unary_op.output = bn_op.output;

        ir.ops.push_back(conv_op);
        ir.ops.push_back(bn_op);
        ir.ops.push_back(unary_op);

        m_ir = std::move(ir);
        m_ops = m_ir.ops;
        m_segments.clear();
        if (!m_ops.empty()) {
            m_segments.push_back(Segment{0, m_ops.size()});
        }
        m_ops_from_flat_segment = true;

        if (const_w && !m_ops.empty()) {
            std::shared_ptr<const ov::Node> conv_node;
            for (const auto& node : model->get_ordered_ops()) {
                if (auto c = ov::as_type_ptr<const ov::op::v1::Convolution>(node)) { conv_node = c; break; }
                if (auto g = ov::as_type_ptr<const ov::op::v1::GroupConvolution>(node)) { conv_node = g; break; }
            }
            load_conv_weights_if_const(conv_node);
        }
        if (const_bn) {
            m_const_bn = bn_ir.ops[0].bn_params;
        }

        m_pipelines.clear();
        m_pipelines.push_back(compiler.compile_conv2d_kernel(m_ops[0], log));
        m_pipelines.push_back(compiler.compile_batchnorm2d_kernel(m_ops[1], log));
        m_pipelines.push_back(compiler.compile_unary_kernel(m_ops[2], log));
        return true;
    }

    // Conv2D solo
    if (has_conv2d && !analysis.has_unary && !analysis.has_softmax && !analysis.has_batchnorm) {
        bool const_w = false;
        try {
            mlir::MLIRContext ctx;
            auto module = build_mlir_conv2d_from_model(model, ctx);
            run_mlir_pipeline(module);
        } catch (const std::exception& e) {
            METAL_LOGSTREAM_TRACE("mlir") << "[MlirBackend] Conv2D MLIR build failed, fallback: " << e.what() << "\n";
            m_force_fallback = true;
            ensure_fallback(m_original_model);
            return true;
        }
        MetalKernelIR ir = build_kernel_ir_for_conv2d(model, const_w);
        m_ir = std::move(ir);
        m_ops = m_ir.ops;
        m_segments.clear();
        if (!m_ops.empty()) {
            m_segments.push_back(Segment{0, m_ops.size()});
        }
        m_ops_from_flat_segment = true;
        METAL_LOG_DEBUG("mlir", "[METAL MLIR] Conv2D selected: input " + describe_shape(m_ir.ops[0].input0) +
                              " weights " + describe_shape(m_ir.ops[0].input1) +
                              " output " + describe_shape(m_ir.ops[0].output));
        if (const_w && !m_ops.empty()) {
            std::shared_ptr<const ov::Node> conv_node;
            for (const auto& node : model->get_ordered_ops()) {
                if (auto c = ov::as_type_ptr<const ov::op::v1::Convolution>(node)) { conv_node = c; break; }
                if (auto g = ov::as_type_ptr<const ov::op::v1::GroupConvolution>(node)) { conv_node = g; break; }
            }
            load_conv_weights_if_const(conv_node);
            if (m_has_const_w && !m_const_w.empty()) {
                const auto* wptr = m_const_w.data();
                std::ostringstream oss;
                oss << "[METAL MLIR] Conv2D W first: ";
                for (size_t i = 0; i < std::min<size_t>(m_const_w.size(), 8); ++i) oss << wptr[i] << " ";
                METAL_LOG_DEBUG("mlir", oss.str());
            }
        }
        m_pipelines.clear();
        if (use_handwritten_msl()) {
            METAL_LOG_DEBUG("mlir", "[METAL MLIR] Conv2D: OV_METAL_USE_HANDWRITTEN_MSL=1, using fallback kernel");
            m_pipelines.push_back(compiler.compile_conv2d_kernel(m_ops[0], log));
        } else {
            try {
                mlir::MLIRContext ctx;
                auto module = build_mlir_conv2d_from_model(model, ctx);
                run_mlir_pipeline(module);
                auto dim_or_one = [](uint32_t v) { return v == 0 ? 1u : v; };

                Conv2DCodegenDesc desc;
                const auto& kop = m_ops[0];
                desc.kind = KernelOpKind::Conv2D;
                desc.N = dim_or_one(kop.conv2d.N);
                desc.C_in = dim_or_one(kop.conv2d.C_in);
                desc.H = dim_or_one(kop.conv2d.H);
                desc.W = dim_or_one(kop.conv2d.W);
                desc.C_out = dim_or_one(kop.conv2d.C_out);
                desc.kH = dim_or_one(kop.conv2d.kernelH);
                desc.kW = dim_or_one(kop.conv2d.kernelW);
                desc.strideH = kop.conv2d.strideH;
                desc.strideW = kop.conv2d.strideW;
                desc.dilationH = kop.conv2d.dilationH;
                desc.dilationW = kop.conv2d.dilationW;
                desc.padTop = kop.conv2d.padTop;
                desc.padLeft = kop.conv2d.padLeft;
                desc.padBottom = kop.conv2d.padBottom;
                desc.padRight = kop.conv2d.padRight;
                desc.outH = dim_or_one(kop.conv2d.outH);
                desc.outW = dim_or_one(kop.conv2d.outW);
                desc.groups = kop.conv2d.groups;
                if (kop.output && kop.output->dtype.ov_type != ov::element::dynamic)
                    desc.element_type = kop.output->dtype.ov_type;
                auto source = generate_msl_from_mlir(module, desc);
                m_pipelines.push_back(compiler.compile_msl_from_source(source, "conv2d_kernel", log));
            } catch (const std::exception& e) {
                METAL_LOG_DEBUG("mlir", std::string("[METAL MLIR] Conv2D MLIR→MSL failed, fallback to hand-written kernel: ") + e.what());
                m_pipelines.push_back(compiler.compile_conv2d_kernel(m_ops[0], log));
            }
        }
        return true;
    }

    return false;
}
