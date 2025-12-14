void MlirBackend::Impl::compile(const std::shared_ptr<const ov::Model>& model) {
        MetalKernelCompiler compiler(m_device);
        std::string log;
        // per-op builders are split into helpers; keep core state reset here
        m_force_fallback = false;
        m_has_const_b = false;
        m_has_const_w = false;
        m_has_const_mul = false;
        m_const_b.clear();
        m_const_w.clear();
        m_const_mul.clear();
        m_const_bn.clear();
        m_has_const_mm0 = m_has_const_mm1 = false;
        m_const_mm0.clear();
        m_const_mm1.clear();
        m_ops_from_flat_segment = false;
        m_add_const_input_index = -1;
        m_ops.clear();
        m_segments.clear();

        auto softmax_supported_globally = [&](const std::shared_ptr<const ov::Model>& m) -> bool {
            for (const auto& node : m->get_ordered_ops()) {
                if (auto s1 = ov::as_type_ptr<const ov::op::v1::Softmax>(node)) {
                    if (!is_softmax_shape_supported(node->get_input_partial_shape(0), s1->get_axis()))
                        return false;
                } else if (auto s8 = ov::as_type_ptr<const ov::op::v8::Softmax>(node)) {
                    if (!is_softmax_shape_supported(node->get_input_partial_shape(0), s8->get_axis()))
                        return false;
                }
            }
            return true;
        };

        // Softmax always compiled for METAL; no dynamic-only host path, no CPU fallback.
        m_softmax_dynamic_only = false;

        const auto caps = default_capabilities();
        const auto analysis = analyze_model_for_mlir(model, caps);
        const bool segments_enabled = true;  // enable flat segments by default
        auto force_cpu_fallback = [&](const std::shared_ptr<const ov::Model>& m) {
            fprintf(stderr, "[METAL][compile][force_fallback] allow_partial=%d\n", m_allow_partial_offload ? 1 : 0);
            if (!m_allow_partial_offload) {
                OPENVINO_THROW("METAL: model contains unsupported ops for this device");
            }
            m_force_fallback = true;
            ensure_fallback(m);
        };
        auto force_cpu_fallback_default = [&]() { force_cpu_fallback(m_original_model); };

        auto precision_supported = [](ov::element::Type t) {
            return t == ov::element::f32 || t == ov::element::f16;
        };

        if (!precision_supported(m_inference_precision)) {
            METAL_LOG_DEBUG("mlir", "[METAL MLIR] Unsupported inference_precision " +
                      std::string(m_inference_precision.get_type_name()));
            force_cpu_fallback_default();
            return;
        }

        auto set_ops_from_ir = [&]() {
            m_ops = m_ir.ops;
            m_segments.clear();
            if (!m_ops.empty()) {
                m_segments.push_back(Segment{0, m_ops.size()});
            }
            m_ops_from_flat_segment = false;
            fprintf(stderr, "[METAL][compile] set_ops_from_ir ops=%zu segments=%zu\n",
                    m_ops.size(), m_segments.size());
        };
        if (analysis.has_unsupported_ops) {
            fprintf(stderr, "[METAL][compile][ERR] unsupported ops: %zu\n", analysis.unsupported_list.size());
            log_list("[MlirBackend] Unsupported ops → CPU fallback: ", analysis.unsupported_list);
            force_cpu_fallback(m_original_model);
            return;
        }

        if (analysis.has_disabled_ops) {
            fprintf(stderr, "[METAL][compile][ERR] disabled ops: %zu\n", analysis.disabled_list.size());
            log_list("[MlirBackend] Ops disabled by capability flags → CPU fallback: ", analysis.disabled_list);
            force_cpu_fallback(m_original_model);
            return;
        }

        if (analysis.has_future_ops) {
            fprintf(stderr, "[METAL][compile][ERR] future ops: %zu\n", analysis.future_list.size());
            log_list("[MlirBackend] Ops recognized but not lowered yet → CPU fallback: ", analysis.future_list);
            force_cpu_fallback(m_original_model);
            return;
        }

        METAL_LOG_DEBUG("mlir", "[MlirBackend] analysis compute_ops=" + std::to_string(analysis.compute_ops) +
                  " matmul=" + (analysis.has_matmul ? "1" : "0") +
                  " softmax=" + (analysis.has_softmax ? "1" : "0"));

        // Always harvest a flat IR so segmented execution and op-specific lowering
        // can reuse the same preparation path (even in release builds).
        build_flat_ir(model);
        build_segments_from_flat_ir();


        auto fill_constants_for_segment = [&](const Segment& seg) {
            // Walk ordered_ops in lockstep with m_flat_ops to harvest constants
            size_t flat_idx = 0;
            size_t mm_seen = 0;
            auto tensor_num_elems = [](const KernelTensor* t) -> size_t {
                if (!t) return 0;
                size_t elems = 1;
                for (auto d : t->shape) elems *= static_cast<size_t>(d);
                return elems;
            };
            for (const auto& node : model->get_ordered_ops()) {
                if (ov::is_type<ov::op::v0::Parameter>(node.get()) ||
                    ov::is_type<ov::op::v0::Result>(node.get()) ||
                    ov::is_type<ov::op::v0::Constant>(node.get())) {
                continue;
                }

                auto node_kind = [&](const std::shared_ptr<const ov::Node>& n, KernelOpKind& kind_out) -> bool {
                    if (ov::as_type_ptr<const ov::op::v0::MatMul>(n)) { kind_out = KernelOpKind::MatMul; return true; }
                    if (ov::as_type_ptr<const ov::op::v1::Add>(n)) { kind_out = KernelOpKind::ElementwiseAdd; return true; }
                    if (ov::as_type_ptr<const ov::op::v1::Subtract>(n)) { kind_out = KernelOpKind::ElementwiseSub; return true; }
                    if (ov::as_type_ptr<const ov::op::v1::Divide>(n)) { kind_out = KernelOpKind::ElementwiseDiv; return true; }
                    if (ov::as_type_ptr<const ov::op::v1::Power>(n)) { kind_out = KernelOpKind::ElementwisePow; return true; }
                    if (ov::as_type_ptr<const ov::op::v1::Multiply>(n)) { kind_out = KernelOpKind::ElementwiseMul; return true; }
                    if (ov::as_type_ptr<const ov::op::v1::Mod>(n)) { kind_out = KernelOpKind::ElementwiseMod; return true; }
                    if (ov::as_type_ptr<const ov::op::v1::FloorMod>(n)) { kind_out = KernelOpKind::ElementwiseFloorMod; return true; }
                    if (ov::as_type_ptr<const ov::op::v0::SquaredDifference>(n)) { kind_out = KernelOpKind::ElementwiseMul; return true; }
                    if (ov::as_type_ptr<const ov::op::v0::Relu>(n) || ov::as_type_ptr<const ov::op::v0::Sigmoid>(n) ||
                        ov::as_type_ptr<const ov::op::v0::Tanh>(n) || ov::as_type_ptr<const ov::op::v0::Elu>(n) ||
                        ov::as_type_ptr<const ov::op::v0::PRelu>(n) || ov::as_type_ptr<const ov::op::v0::Gelu>(n) ||
                        ov::as_type_ptr<const ov::op::v4::Swish>(n)) { kind_out = KernelOpKind::Unary; return true; }
                    if (ov::is_type<ov::op::v1::Softmax>(n.get()) || ov::is_type<ov::op::v8::Softmax>(n.get())) { kind_out = KernelOpKind::Softmax; return true; }
                    if (ov::as_type_ptr<const ov::op::v1::MaxPool>(n)) { kind_out = KernelOpKind::MaxPool2D; return true; }
                    if (ov::as_type_ptr<const ov::op::v1::AvgPool>(n)) { kind_out = KernelOpKind::AvgPool2D; return true; }
                    if (auto cv = ov::as_type_ptr<const ov::op::v1::Convolution>(n)) {
                        const auto rank = cv->get_input_partial_shape(0).rank();
                        if (rank.is_static() && rank.get_length() == 5) { kind_out = KernelOpKind::Conv3D; return true; }
                        kind_out = KernelOpKind::Conv2D; return true;
                    }
                    if (auto gcv = ov::as_type_ptr<const ov::op::v1::GroupConvolution>(n)) {
                        const auto rank = gcv->get_input_partial_shape(0).rank();
                        if (rank.is_static() && rank.get_length() == 5) { kind_out = KernelOpKind::Conv3D; return true; }
                        kind_out = KernelOpKind::Conv2D; return true;
                    }
                    if (ov::as_type_ptr<const ov::op::v0::BatchNormInference>(n) || ov::as_type_ptr<const ov::op::v5::BatchNormInference>(n)) {
                        kind_out = KernelOpKind::BatchNorm2D; return true; }
                    if (ov::as_type_ptr<const ov::op::v8::Slice>(n)) { kind_out = KernelOpKind::Slice; return true; }
                    return false;
                };

                KernelOpKind kind;
                if (!node_kind(node, kind))
                    continue;  // shape-only or unsupported

                if (flat_idx < seg.first_op_index) {
                    ++flat_idx;
                    continue;
                }
                if (flat_idx >= seg.first_op_index + seg.op_count)
                    break;

                auto& op = m_flat_ops[flat_idx];
                switch (kind) {
                    case KernelOpKind::MatMul: {
                        auto weight = std::dynamic_pointer_cast<const ov::op::v0::Constant>(node->get_input_node_shared_ptr(1));
                        auto mm_node = std::dynamic_pointer_cast<const ov::op::v0::MatMul>(node);
                        const bool transpose_b = mm_node ? mm_node->get_transpose_b() : false;
                        // Reflect transpose_b in op flag
                        op.b_is_nk_layout = transpose_b;
                        m_flat_ops[flat_idx].b_is_nk_layout = transpose_b;
                        if (weight) {
                            auto vec = weight->cast_vector<float>();
                            const auto& wshape = weight->get_shape();  // original constant shape
                            auto desired_K = op.K;
                            auto desired_N = op.N;
                            OPENVINO_ASSERT(desired_K > 0 && desired_N > 0, "MatMul weights: invalid K/N");
                            const bool produce_nk = transpose_b;  // true -> store N x K layout
                            auto normalize_layout = [&](std::vector<float>& data) {
                                size_t batches = 1;
                                if (wshape.size() == 3)
                                    batches = wshape[0];
                                const size_t K = static_cast<size_t>(desired_K);
                                const size_t N = static_cast<size_t>(desired_N);
                                std::vector<float> dst(static_cast<size_t>(batches * K * N), 0.f);

                                auto read_at = [&](size_t b, size_t r, size_t c) -> float {
                                    // r,c follow source ordering as stored
                                    if (wshape.size() == 2) {
                                        if (r >= wshape[0] || c >= wshape[1]) return 0.f;
                                        return data[r * wshape[1] + c];
                                    } else {  // rank-3: [B, R, C]
                                        if (b >= wshape[0] || r >= wshape[1] || c >= wshape[2]) return 0.f;
                                        return data[(b * wshape[1] + r) * wshape[2] + c];
                                    }
                                };

                                auto write_dst = [&](size_t b, size_t k, size_t n, float v) {
                                    if (produce_nk) {
                                        // store as [N, K] row-major (n major)
                                        dst[(b * N + n) * K + k] = v;
                                    } else {
                                        // store as [K, N] row-major (k major)
                                        dst[(b * K + k) * N + n] = v;
                                    }
                                };

                                auto fill_from = [&](bool source_is_kxn) {
                                    for (size_t b = 0; b < batches; ++b) {
                                        for (size_t k = 0; k < K; ++k) {
                                            for (size_t n = 0; n < N; ++n) {
                                                float v = source_is_kxn ? read_at(b, k, n) : read_at(b, n, k);
                                                write_dst(b, k, n, v);
                                            }
                                        }
                                    }
                                };

                                if (wshape.size() == 2) {
                                    const size_t W0 = wshape[0], W1 = wshape[1];
                                    if (W0 == K && W1 == N) {
                                        fill_from(true);
                                    } else if (W0 == N && W1 == K) {
                                        fill_from(false);
                                    } else {
                                        OPENVINO_ASSERT(false, "MatMul weights shape unsupported");
                                    }
                                } else if (wshape.size() == 3) {
                                    const size_t W1 = wshape[1], W2 = wshape[2];
                                    if (W1 == K && W2 == N) {
                                        fill_from(true);
                                    } else if (W1 == N && W2 == K) {
                                        fill_from(false);
                                    } else {
                                        OPENVINO_ASSERT(false, "MatMul weights rank-3 shape unsupported");
                                    }
                                } else {
                                    OPENVINO_ASSERT(false, "MatMul weights rank not supported");
                                }
                                data.swap(dst);
                            };

                            normalize_layout(vec);
                            if (mm_seen == 0) {
                                m_const_mm0 = vec;
                                m_has_const_mm0 = true;
                            } else {
                                m_const_mm1 = vec;
                                m_has_const_mm1 = true;
                            }
                            m_const_b = vec;  // reused for first MatMul weight
                            m_has_const_b = true;
                        }
                        ++mm_seen;
                        break;
                    }
                    case KernelOpKind::ElementwiseMul: {
                        for (size_t i = 0; i < node->get_input_size(); ++i) {
                            if (auto c = std::dynamic_pointer_cast<const ov::op::v0::Constant>(node->get_input_node_shared_ptr(i))) {
                                m_const_mul = c->cast_vector<float>();
                                m_has_const_mul = true;
                                break;
                            }
                        }
                        break;
                    }
                    case KernelOpKind::ElementwiseAdd: {
                        size_t add_const_elems = 0;
                        for (size_t i = 0; i < node->get_input_size(); ++i) {
                            if (auto c = std::dynamic_pointer_cast<const ov::op::v0::Constant>(node->get_input_node_shared_ptr(i))) {
                                const auto& c_shape = c->get_shape();
                                add_const_elems = std::accumulate(c_shape.begin(), c_shape.end(), static_cast<size_t>(1),
                                                                  std::multiplies<size_t>());
                                if (!m_has_const_b) {
                                    m_const_b = c->cast_vector<float>();
                                    m_has_const_b = true;
                                    m_add_const_input_index = static_cast<int>(i);
                                    // If this is a pure broadcastable constant, pre-broadcast it and disable kernel-side broadcast.
                                    auto try_expand_const = [&](const ov::Shape& shape) {
                                        if (!m_const_b.empty()) {
                                            std::ostringstream dbg;
                                            dbg << "[METAL MLIR] try_expand_const: original first=" << m_const_b[0]
                                                << " size=" << m_const_b.size();
                                            METAL_LOG_DEBUG("mlir", dbg.str());
                                        }
                                        if (!op.output) return false;
                                        const size_t out_elems = tensor_num_elems(op.output);
                                        if (out_elems == 0) return false;
                                        // Scalar
                                        if (m_const_b.size() == 1) {
                                            float v = m_const_b[0];
                                            m_const_b.assign(out_elems, v);
                                            return true;
                                        }
                                        // Channel broadcast for NCHW/NCL/NC
                                        if (op.out_shape.size() >= 2) {
                                            size_t Cdim = 1;
                                            size_t C = static_cast<size_t>(op.out_shape[Cdim]);
                                            if (m_const_b.size() == C) {
                                                m_const_b.assign(out_elems, 0.f);
                                                if (op.out_shape.size() == 2) {
                                                    size_t N = static_cast<size_t>(op.out_shape[0]);
                                                    for (size_t n = 0; n < N; ++n)
                                                        for (size_t cidx = 0; cidx < C; ++cidx)
                                                            m_const_b[n * C + cidx] = c->cast_vector<float>()[cidx];
                                                } else if (op.out_shape.size() == 3) {
                                                    size_t N = static_cast<size_t>(op.out_shape[0]);
                                                    size_t L = static_cast<size_t>(op.out_shape[2]);
                                                    for (size_t n = 0; n < N; ++n)
                                                        for (size_t cidx = 0; cidx < C; ++cidx)
                                                            for (size_t l = 0; l < L; ++l)
                                                                m_const_b[(n * C + cidx) * L + l] = c->cast_vector<float>()[cidx];
                                                } else if (op.out_shape.size() == 4) {
                                                    size_t N = static_cast<size_t>(op.out_shape[0]);
                                                    size_t H = static_cast<size_t>(op.out_shape[2]);
                                                    size_t W = static_cast<size_t>(op.out_shape[3]);
                                                    for (size_t n = 0; n < N; ++n)
                                                        for (size_t cidx = 0; cidx < C; ++cidx)
                                                            for (size_t h = 0; h < H; ++h)
                                                                for (size_t w = 0; w < W; ++w)
                                                                    m_const_b[((n * C + cidx) * H + h) * W + w] = c->cast_vector<float>()[cidx];
                                                }
                                                return true;
                                            }
                                        }
                                        if (m_const_b.size() == out_elems)
                                            return true;
                                        return false;
                                    };
                                    if (op.is_broadcast) {
                                        bool expanded = try_expand_const(c_shape);
                                        if (expanded) {
                                            op.is_broadcast = false;
                                            op.stride0.clear();
                                            op.stride1.clear();
                                        }
                                    }
                                }
                                break;
                            }
                        }
                        break;
                    }
                    case KernelOpKind::Conv2D:
                    case KernelOpKind::Conv3D: {
                        if (auto c = std::dynamic_pointer_cast<const ov::op::v0::Constant>(node->get_input_node_shared_ptr(1))) {
                            m_const_w = c->cast_vector<float>();
                            m_has_const_w = true;
                        }
                        break;
                    }
                    case KernelOpKind::BatchNorm2D: {
                        const auto bn_v5 = ov::as_type_ptr<const ov::op::v5::BatchNormInference>(node);
                        const auto bn_v0 = ov::as_type_ptr<const ov::op::v0::BatchNormInference>(node);
                        const bool use_v5 = static_cast<bool>(bn_v5);
                        auto get_const = [&](size_t idx) {
                            return std::dynamic_pointer_cast<const ov::op::v0::Constant>(use_v5
                                ? bn_v5->get_input_node_shared_ptr(idx)
                                : bn_v0->get_input_node_shared_ptr(idx));
                        };
                        auto gamma = get_const(1);
                        auto beta = get_const(2);
                        auto mean = get_const(3);
                        auto var = get_const(4);
                        if (gamma && beta && mean && var) {
                            const size_t C = node->get_input_shape(0)[1];
                            op.bn_params.assign(4 * C + 1, 0.f);
                            auto copy_vec = [&](const std::shared_ptr<const ov::op::v0::Constant>& cst, size_t offset) {
                                auto vec = cst->cast_vector<float>();
                                if (vec.size() == C)
                                    std::copy(vec.begin(), vec.end(), op.bn_params.begin() + offset);
                            };
                            copy_vec(gamma, 0 * C);
                            copy_vec(beta, 1 * C);
                            copy_vec(mean, 2 * C);
                            copy_vec(var, 3 * C);
                            op.bn_params[4 * C] = static_cast<float>(use_v5 ? bn_v5->get_eps_value() : bn_v0->get_eps_value());
                            m_const_bn = op.bn_params;
                        }
                        break;
                    }
                    default:
                        break;
                }

                ++flat_idx;
            }
        };

        if (!m_flat_ops.empty()) {
            Segment all_seg{0, m_flat_ops.size()};
            fill_constants_for_segment(all_seg);
        }

        auto try_pick_single_flat_segment = [&]() {
            if (!m_ops.empty())
                return;
            auto allow_kind = [](KernelOpKind k) {
                switch (k) {
                    case KernelOpKind::Unary:
                    case KernelOpKind::Softmax:
                    case KernelOpKind::Conv2D:
                    case KernelOpKind::Conv3D:
                    case KernelOpKind::BatchNorm2D:
                    case KernelOpKind::MatMul:
                    case KernelOpKind::ElementwiseAdd:
                    case KernelOpKind::ElementwiseSub:
                    case KernelOpKind::ElementwiseDiv:
                    case KernelOpKind::ElementwiseMul:
                    case KernelOpKind::ElementwiseMod:
                    case KernelOpKind::ElementwiseFloorMod:
                        return true;
                    default:
                        return false;
                }
            };
            for (const auto& seg : m_flat_segments) {
                if (seg.op_count != 1)
                    continue;
                // only if covers full model
                if (seg.first_op_index != 0 || seg.op_count != m_flat_ops.size())
                    continue;
                const auto& op = m_flat_ops[seg.first_op_index];
                if (!allow_kind(op.kind))
                    continue;
                m_ops.clear();
                m_segments.clear();
                fill_constants_for_segment(seg);
                m_ops.push_back(op);
                Segment io_seg{};
                io_seg.first_op_index = 0;
                io_seg.op_count = 1;
                for (const auto& in_port : model->inputs()) io_seg.input_ports.push_back(in_port);
                for (const auto& out_port : model->outputs()) io_seg.output_ports.push_back(out_port);
                m_segments.push_back(std::move(io_seg));
                m_ops_from_flat_segment = true;
#if METAL_MLIR_DEBUG
                METAL_LOG_DEBUG("mlir", "[METAL MLIR] Using single-op flat segment kind=" + std::to_string(static_cast<int>(op.kind)));
#endif
                // compile pipeline for this single op
                MetalKernelCompiler compiler(m_device);
                std::string log;
                m_pipelines.clear();
                bool compile_ok = true;
                switch (op.kind) {
                    case KernelOpKind::Unary:
                        m_pipelines.push_back(compiler.compile_unary_kernel(op, log));
                        break;
                    case KernelOpKind::Softmax:
                        m_pipelines.push_back(compiler.compile_softmax_kernel(op, log));
                        break;
                    case KernelOpKind::Conv2D:
                        m_pipelines.push_back(compiler.compile_conv2d_kernel(op, log));
                        break;
                    case KernelOpKind::Conv3D: {
                        try {
                            mlir::MLIRContext ctx;
                            auto module = build_mlir_conv3d_from_model(m_model, ctx);
                            run_mlir_pipeline(module);
                            Conv3DCodegenDesc desc;
                            const auto& kop = op;
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
                            METAL_LOG_DEBUG("mlir", std::string("[METAL MLIR] Conv3D flat segment compile failed: ") + e.what());
                            compile_ok = false;
                        }
                        break;
                    }
                    case KernelOpKind::BatchNorm2D:
                        m_pipelines.push_back(compiler.compile_batchnorm2d_kernel(op, log));
                        break;
                    case KernelOpKind::MatMul:
                        m_pipelines.push_back(compiler.compile_matmul_kernel(op, log));
                        break;
                    case KernelOpKind::ElementwiseAdd:
                        m_pipelines.push_back(compiler.compile_add_kernel(op, log));
                        break;
                    case KernelOpKind::ElementwiseSub:
                        m_pipelines.push_back(compiler.compile_sub_kernel(op, log));
                        break;
                    case KernelOpKind::ElementwiseDiv:
                        m_pipelines.push_back(compiler.compile_div_kernel(op, log));
                        break;
                    case KernelOpKind::ElementwiseMul:
                        m_pipelines.push_back(compiler.compile_mul_kernel(op, log));
                        break;
                    case KernelOpKind::ElementwiseMod:
                        m_pipelines.push_back(compiler.compile_mod_kernel(op, log));
                        break;
                    case KernelOpKind::ElementwiseFloorMod:
                        m_pipelines.push_back(compiler.compile_floor_mod_kernel(op, log));
                        break;
                    case KernelOpKind::Slice:
                        m_pipelines.push_back(compiler.compile_slice_kernel(op, log));
                        break;
                    default:
                        break;
                }
                break;
            }
        };

        if (segments_enabled) {
            try_pick_single_flat_segment();
        }
        auto try_pick_multi_flat_segment = [&]() {
            if (!m_ops.empty())
                return;
            // Scan flat ops for the first MatMul -> Softmax -> MatMul chain.
            for (size_t i = 0; i + 2 < m_flat_ops.size(); ++i) {
                const auto& op0 = m_flat_ops[i];
                const auto& op1 = m_flat_ops[i + 1];
                const auto& op2 = m_flat_ops[i + 2];
                if (!(op0.kind == KernelOpKind::MatMul &&
                      op1.kind == KernelOpKind::Softmax &&
                      op2.kind == KernelOpKind::MatMul)) {
                    continue;
                }
                // Basic chaining: outputs/inputs must be the same tensor objects.
                if (!(op0.output && op1.input0 && op0.output == op1.input0))
                    continue;
                if (!(op1.output && op2.input0 && op1.output == op2.input0))
                    continue;
                m_ops.clear();
                m_segments.clear();
                m_pipelines.clear();
                m_ops_from_flat_segment = true;
                Segment seg{i, 3};
                if (!m_model->inputs().empty())
                    seg.input_ports.push_back(m_model->inputs()[0]);
                if (!m_model->outputs().empty())
                    seg.output_ports.push_back(m_model->outputs()[0]);
                fill_constants_for_segment(seg);
                MetalKernelCompiler compiler(m_device);
                std::string log;
                m_ops.push_back(m_flat_ops[i + 0]);
                m_ops.push_back(m_flat_ops[i + 1]);
                m_ops.push_back(m_flat_ops[i + 2]);
                m_pipelines.push_back(compiler.compile_matmul_kernel(m_ops[0], log));
                m_pipelines.push_back(compiler.compile_softmax_kernel(m_ops[1], log));
                m_pipelines.push_back(compiler.compile_matmul_kernel(m_ops[2], log));
                m_segments.push_back(Segment{0, 3});
#if METAL_MLIR_DEBUG
                METAL_LOG_DEBUG("mlir", "[METAL MLIR] Using flat segment MatMul->Softmax->MatMul len=3");
#endif
                break;
            }
        };
        if (segments_enabled) {
            try_pick_multi_flat_segment();
            // If no template matched, try to run full flat IR as one segment
            if (m_ops.empty() && !m_flat_ops.empty() && !m_flat_segments.empty()) {
                m_ops.clear();
                m_segments.clear();
                for (const auto& op : m_flat_ops) {
                    m_ops.push_back(op);
                }
                m_segments = m_flat_segments;
                m_ops_from_flat_segment = true;
                MetalKernelCompiler compiler(m_device);
                std::string log;
                m_pipelines.clear();
                bool compile_ok = true;
                for (const auto& op : m_ops) {
                    switch (op.kind) {
                        case KernelOpKind::MatMul:
                            m_pipelines.push_back(compiler.compile_matmul_kernel(op, log));
                            break;
                        case KernelOpKind::Softmax:
                            m_pipelines.push_back(compiler.compile_softmax_kernel(op, log));
                            break;
                        case KernelOpKind::ElementwiseAdd:
                            m_pipelines.push_back(compiler.compile_add_kernel(op, log));
                            break;
                        case KernelOpKind::ElementwiseSub:
                            m_pipelines.push_back(compiler.compile_sub_kernel(op, log));
                            break;
                        case KernelOpKind::ElementwiseDiv:
                            m_pipelines.push_back(compiler.compile_div_kernel(op, log));
                            break;
                        case KernelOpKind::ElementwisePow:
                            m_pipelines.push_back(compiler.compile_pow_kernel(op, log));
                            break;
                        case KernelOpKind::ElementwiseMul:
                            m_pipelines.push_back(compiler.compile_mul_kernel(op, log));
                            break;
                        case KernelOpKind::ElementwiseMod:
                            m_pipelines.push_back(compiler.compile_mod_kernel(op, log));
                            break;
                        case KernelOpKind::ElementwiseFloorMod:
                            m_pipelines.push_back(compiler.compile_floor_mod_kernel(op, log));
                            break;
                        case KernelOpKind::Unary:
                            m_pipelines.push_back(compiler.compile_unary_kernel(op, log));
                            break;
                        case KernelOpKind::Convert:
                            m_pipelines.push_back(compiler.compile_convert_kernel(op, log));
                            break;
                        case KernelOpKind::Transpose:
                            m_pipelines.push_back(compiler.compile_transpose_kernel(op, log));
                            break;
                        case KernelOpKind::Reshape:
                            m_pipelines.push_back(compiler.compile_reshape_kernel(op, log));
                            break;
                        case KernelOpKind::Conv2D:
                            m_pipelines.push_back(compiler.compile_conv2d_kernel(op, log));
                            break;
                        case KernelOpKind::BatchNorm2D:
                            m_pipelines.push_back(compiler.compile_batchnorm2d_kernel(op, log));
                            break;
                        case KernelOpKind::MaxPool2D:
                            m_pipelines.push_back(compiler.compile_maxpool2d_kernel(op, log));
                            break;
                        case KernelOpKind::AvgPool2D:
                            m_pipelines.push_back(compiler.compile_avgpool2d_kernel(op, log));
                            break;
                        case KernelOpKind::Interpolate:
                            m_pipelines.push_back(compiler.compile_interpolate_kernel(op, log));
                            break;
                        case KernelOpKind::Concat:
                            m_pipelines.push_back(compiler.compile_concat_kernel(op, log));
                            break;
                        case KernelOpKind::Slice:
                            m_pipelines.push_back(compiler.compile_slice_kernel(op, log));
                            break;
                        case KernelOpKind::Split:
                            m_pipelines.push_back(compiler.compile_split_kernel(op, log));
                            break;
                        default:
                            compile_ok = false;
                            break;
                    }
                }
                if (!compile_ok || m_pipelines.size() != m_ops.size()) {
                    METAL_LOG_DEBUG("mlir", "[METAL MLIR] Fallback: flat IR contains unsupported op for Metal dispatch");
                    m_ops.clear();
                    m_segments.clear();
                } else {
#if METAL_MLIR_DEBUG
                    METAL_LOG_DEBUG("mlir", "[METAL MLIR] Using full flat IR segment count=" + std::to_string(m_ops.size()));
#endif
                }
            }
        }

        if (!m_ops.empty()) {
            return;
        }

        if (analysis.compute_ops > 32) {
            METAL_LOG_DEBUG("mlir", "[MlirBackend] Warning: large compute_ops=" + std::to_string(analysis.compute_ops) +
                      " (not forcing fallback)");
        }

        bool has_matmul = analysis.has_matmul;
        bool has_add = analysis.has_add;
        bool has_mul = analysis.has_mul;
        bool has_add_broadcast = analysis.has_add_broadcast;
        bool has_maxpool = analysis.has_maxpool;
        bool has_avgpool = analysis.has_avgpool;
        bool has_conv2d = analysis.has_conv2d;
        bool has_conv3d = analysis.has_conv3d;
        bool has_batchnorm = analysis.has_batchnorm;
        // Probe for constant B to support single-input matmuls
        std::shared_ptr<const ov::op::v0::MatMul> matmul_node;
        for (const auto& node : model->get_ordered_ops()) {
            if (auto mm = ov::as_type_ptr<const ov::op::v0::MatMul>(node)) {
                matmul_node = mm;
                break;
            }
        }
        if (!has_matmul && !has_add && !has_add_broadcast && !has_mul && !analysis.has_pow &&
            !analysis.has_unary && !analysis.has_softmax && !analysis.has_maxpool && !analysis.has_avgpool &&
            !analysis.has_conv2d && !analysis.has_batchnorm && !analysis.has_split) {
            if (analysis.compute_ops == 0) {
                METAL_LOG_DEBUG("mlir", "[MlirBackend] shape-only graph after template probe; skipping Metal pipelines");
                return;
            }
            force_cpu_fallback(m_original_model);
            return;
        }
        if (matmul_node) {
            auto input1 = matmul_node->get_input_node_shared_ptr(1);
            if (auto c = std::dynamic_pointer_cast<const ov::op::v0::Constant>(input1)) {
                m_has_const_b = true;
                m_const_b = c->cast_vector<float>();
            }
        }

        if (has_matmul) {
            if (try_build_matmul(model, analysis, compiler, log))
                return;
        }

        if (has_matmul && (analysis.has_softmax || analysis.has_unary) && analysis.compute_ops == 2) {
            MetalKernelIR ir = build_kernel_ir_for_matmul(model);
            KernelTensor out_tensor = ir.tensors.back();
            KernelTensor second_out = {"stage1_out", out_tensor.shape};
            ir.tensors.push_back(second_out);

            KernelOp second{};
            if (analysis.has_softmax) {
                second.kind = KernelOpKind::Softmax;
                if (out_tensor.shape.size() == 2) {
                    second.rows = out_tensor.shape[0];
                    second.cols = out_tensor.shape[1];
                    second.inner = 1;
                } else if (out_tensor.shape.size() == 3) {
                    second.rows = out_tensor.shape[0] * out_tensor.shape[1];
                    second.cols = out_tensor.shape[2];
                    second.inner = 1;
                } else {
                    second.rows = 1;
                    for (size_t i = 0; i + 1 < out_tensor.shape.size(); ++i)
                        second.rows *= out_tensor.shape[i];
                    second.cols = out_tensor.shape.back();
                    second.inner = 1;
                }
            } else {
                second.kind = KernelOpKind::Unary;
                second.activation = analysis.unary_kind;
                second.alpha = analysis.unary_alpha;
            }
            second.input0 = &ir.tensors[ir.tensors.size() - 2];
            second.output = &ir.tensors.back();
            ir.ops.push_back(second);

            m_ir = std::move(ir);
            set_ops_from_ir();
            m_pipelines.clear();
            m_pipelines.push_back(compiler.compile_matmul_kernel(m_ops[0], log));
            if (analysis.has_softmax)
            m_pipelines.push_back(compiler.compile_softmax_kernel(m_ops[1], log));
            else
                m_pipelines.push_back(compiler.compile_unary_kernel(m_ops[1], log));
            return;
        }

        if (has_matmul && analysis.has_softmax && !has_add && !has_add_broadcast && !analysis.has_unary &&
            !has_maxpool && !has_avgpool && !has_conv2d && !has_batchnorm && analysis.compute_ops == 3) {
            std::shared_ptr<const ov::op::v0::MatMul> mm0, mm1;
            std::shared_ptr<const ov::Node> sm;
            for (const auto& node : model->get_ordered_ops()) {
                if (!mm0) {
                    if (auto m = ov::as_type_ptr<const ov::op::v0::MatMul>(node)) { mm0 = m; continue; }
                } else if (!sm) {
                    if (ov::is_type<ov::op::v1::Softmax>(node.get()) || ov::is_type<ov::op::v8::Softmax>(node.get())) {
                        sm = node; continue;
                    }
                } else if (!mm1) {
                    if (auto m = ov::as_type_ptr<const ov::op::v0::MatMul>(node)) { mm1 = m; break; }
                }
            }
            if (!mm0 || !sm || !mm1) {
                force_cpu_fallback(m_original_model);
                return;
            }

            auto expand_to_3d = [](const ov::Shape& s) {
                OPENVINO_ASSERT(s.size() == 2 || s.size() == 3, "MatMul attention: only rank-2/3 inputs supported");
                if (s.size() == 3) return std::vector<int64_t>{static_cast<int64_t>(s[0]), static_cast<int64_t>(s[1]), static_cast<int64_t>(s[2])};
                return std::vector<int64_t>{1, static_cast<int64_t>(s[0]), static_cast<int64_t>(s[1])};
            };
            auto make_matmul_op = [&](const ov::Shape& a_shape,
                                      const ov::Shape& b_shape,
                                      KernelTensor* a_tensor,
                                      KernelTensor* b_tensor,
                                      KernelTensor* c_tensor) -> KernelOp {
                KernelOp op;
                op.kind = KernelOpKind::MatMul;
                op.input0 = a_tensor;
                op.input1 = b_tensor;
                op.output = c_tensor;
                auto a3 = expand_to_3d(a_shape);
                auto b3 = expand_to_3d(b_shape);
                const int64_t batch_a = a3[0];
                const int64_t batch_b = b3[0];
                const int64_t batch = std::max(batch_a, batch_b);
                OPENVINO_ASSERT(batch_a == batch_b || batch_a == 1 || batch_b == 1, "MatMul attention: batch broadcast mismatch");
                const int64_t M = a3[1];
                const int64_t K_a = a3[2];
                int64_t K_b = b3[1];
                int64_t N = b3[2];
                bool b_is_nk = false;
                if (K_b != K_a && b3[2] == K_a) {
                    K_b = b3[2];
                    N = b3[1];
                    b_is_nk = true;
                }
                OPENVINO_ASSERT(K_a == K_b, "MatMul attention: K mismatch");
                op.M = M;
                op.N = N;
                op.K = K_a;
                op.batch = batch;
                op.batch_a = batch_a;
                op.batch_b = batch_b;
                op.b_is_nk_layout = b_is_nk;
                return op;
            };

            const auto a0_shape = mm0->get_input_shape(0);
            const auto b0_shape = mm0->get_input_shape(1);
            const auto out0_shape = mm0->get_output_shape(0);
            const auto sm_shape = out0_shape;
            const auto a1_shape = mm1->get_input_shape(0);
            const auto b1_shape = mm1->get_input_shape(1);
            const auto out1_shape = mm1->get_output_shape(0);

            KernelTensor t_in{"in", {a0_shape.begin(), a0_shape.end()}};
            KernelTensor t_w0{"w0", {b0_shape.begin(), b0_shape.end()}};
            KernelTensor t_tmp1{"tmp1", {out0_shape.begin(), out0_shape.end()}};
            KernelTensor t_tmp2{"tmp2", {sm_shape.begin(), sm_shape.end()}};
            KernelTensor t_w1{"w1", {b1_shape.begin(), b1_shape.end()}};
            KernelTensor t_out{"out", {out1_shape.begin(), out1_shape.end()}};
            auto set_dtype = [](KernelTensor& t, const ov::element::Type& et) {
                try {
                    t.dtype = resolve_metal_dtype(et);
                } catch (const ov::Exception&) {
                    // leave dynamic; fallback will handle unsupported types
                }
            };
            set_dtype(t_in, mm0->get_input_element_type(0));
            set_dtype(t_tmp1, mm0->get_output_element_type(0));
            set_dtype(t_tmp2, mm1->get_input_element_type(0));
            set_dtype(t_out, mm1->get_output_element_type(0));

            auto prepare_const = [&](KernelTensor& t,
                                     const std::shared_ptr<const ov::op::v0::Constant>& c,
                                     int64_t K,
                                     int64_t N,
                                     bool store_nk) {
                if (!c)
                    return;
                t.from_constant = true;
                t.source_node = c.get();
                set_dtype(t, c->get_element_type());
                auto data = c->cast_vector<float>();
                const auto& shape = c->get_shape();
                if (store_nk && shape.size() >= 2) {
                    const size_t R = shape[0];
                    const size_t C = shape[1];
                    const size_t batches = shape.size() == 3 ? shape[0] : 1;
                    const size_t rows = shape.size() == 3 ? shape[1] : R;
                    const size_t cols = shape.size() == 3 ? shape[2] : C;
                    const size_t expect_K = static_cast<size_t>(K);
                    const size_t expect_N = static_cast<size_t>(N);
                    std::vector<float> dst(static_cast<size_t>(batches * expect_K * expect_N), 0.f);
                    auto read = [&](size_t b, size_t r, size_t c) -> float {
                        if (shape.size() == 3) {
                            if (b >= shape[0] || r >= shape[1] || c >= shape[2]) return 0.f;
                            return data[(b * shape[1] + r) * shape[2] + c];
                        }
                        if (r >= shape[0] || c >= shape[1]) return 0.f;
                        return data[r * shape[1] + c];
                    };
                    for (size_t b = 0; b < batches; ++b) {
                        for (size_t k = 0; k < expect_K; ++k) {
                            for (size_t n = 0; n < expect_N; ++n) {
                                // store as [N, K] (n-major) when requested
                                float v = 0.f;
                                if (rows == expect_K && cols == expect_N) {
                                    v = read(b, k, n);
                                } else if (rows == expect_N && cols == expect_K) {
                                    v = read(b, n, k);
                                }
                                dst[(b * expect_N + n) * expect_K + k] = v;
                            }
                        }
                    }
                    data.swap(dst);
                }
                t.const_data = std::move(data);
                if (std::getenv("METAL_DEBUG_DUMP_OUTPUT") && !t.const_data.empty()) {
                    fprintf(stderr, "[METAL][dbg] const tensor %s shape=", t.name.c_str());
                    for (size_t i = 0; i < shape.size(); ++i) fprintf(stderr, "%zu%s", shape[i], i + 1 == shape.size() ? "" : ",");
                    fprintf(stderr, " first4:");
                    for (size_t i = 0; i < std::min<size_t>(4, t.const_data.size()); ++i)
                        fprintf(stderr, " %f", t.const_data[i]);
                    fprintf(stderr, " store_nk=%d\n", store_nk ? 1 : 0);
                }
            };

            MetalKernelIR ir;
            ir.tensors = {t_in, t_w0, t_tmp1, t_tmp2, t_w1, t_out};

            KernelOp op0 = make_matmul_op(a0_shape, b0_shape, &ir.tensors[0], &ir.tensors[1], &ir.tensors[2]);
            KernelOp op1;
            op1.kind = KernelOpKind::Softmax;
            op1.input0 = &ir.tensors[2];
            op1.output = &ir.tensors[3];
            size_t elems = 1;
            for (auto d : sm_shape) elems *= d;
            const size_t cols = sm_shape.back();
            const size_t rows = elems / cols;
            op1.rows = static_cast<int64_t>(rows);
            op1.cols = static_cast<int64_t>(cols);
            op1.inner = 1;
            int64_t softmax_axis = -1;
            if (auto s1 = ov::as_type_ptr<const ov::op::v1::Softmax>(sm)) softmax_axis = s1->get_axis();
            else if (auto s8 = ov::as_type_ptr<const ov::op::v8::Softmax>(sm)) softmax_axis = s8->get_axis();
            op1.softmax_axis = softmax_axis;
            // If axis is not the last dimension, derive rows/cols/inner to match runtime path.
            if (sm_shape.size() >= 1) {
                auto rank = static_cast<int64_t>(sm_shape.size());
                int64_t ax = softmax_axis;
                if (ax < 0) ax += rank;
                if (ax >= 0 && ax < rank) {
                    int64_t c = static_cast<int64_t>(sm_shape[static_cast<size_t>(ax)]);
                    int64_t inner = 1, outer = 1;
                    for (int64_t i = 0; i < ax; ++i) outer *= static_cast<int64_t>(sm_shape[static_cast<size_t>(i)]);
                    for (size_t i = static_cast<size_t>(ax) + 1; i < sm_shape.size(); ++i)
                        inner *= static_cast<int64_t>(sm_shape[i]);
                    op1.cols = c;
                    op1.rows = outer * inner;
                    op1.inner = inner;
                }
            }
            KernelOp op2 = make_matmul_op(a1_shape, b1_shape, &ir.tensors[3], &ir.tensors[4], &ir.tensors[5]);
            // Attach constant data (NK layout) to weight tensors so runtime binds them via MetalBufferManager.
            prepare_const(ir.tensors[1], std::dynamic_pointer_cast<const ov::op::v0::Constant>(mm0->get_input_node_shared_ptr(1)),
                          op0.K, op0.N, op0.b_is_nk_layout);
            prepare_const(ir.tensors[4], std::dynamic_pointer_cast<const ov::op::v0::Constant>(mm1->get_input_node_shared_ptr(1)),
                          op2.K, op2.N, op2.b_is_nk_layout);

            ir.ops = {op0, op1, op2};
            m_ir = std::move(ir);
            set_ops_from_ir();

            m_has_const_mm0 = false; m_has_const_mm1 = false;
            if (auto c = std::dynamic_pointer_cast<const ov::op::v0::Constant>(mm0->get_input_node_shared_ptr(1))) {
                m_const_mm0 = c->cast_vector<float>();
                m_has_const_mm0 = true;
            }
            if (auto c = std::dynamic_pointer_cast<const ov::op::v0::Constant>(mm1->get_input_node_shared_ptr(1))) {
                m_const_mm1 = c->cast_vector<float>();
                m_has_const_mm1 = true;
            }

            m_pipelines.clear();
            m_pipelines.push_back(compiler.compile_matmul_kernel(m_ops[0], log));
            m_pipelines.push_back(compiler.compile_softmax_kernel(m_ops[1], log));
            m_pipelines.push_back(compiler.compile_matmul_kernel(m_ops[2], log));
            return;
        }

        if ((has_add || has_add_broadcast || analysis.has_unary || analysis.has_mul || analysis.has_div || analysis.has_pow ||
             analysis.has_mod || analysis.has_floor_mod || analysis.has_squared_diff) && !has_matmul && !analysis.has_softmax) {
            if (try_build_eltwise(model, analysis, compiler, log))
                return;
        }

        if (has_mul && !analysis.has_div && !analysis.has_mod && !analysis.has_floor_mod &&
            analysis.compute_ops == 1 && !has_add && !has_add_broadcast && !analysis.has_unary &&
            !analysis.has_softmax && !has_maxpool && !has_avgpool && !has_conv2d && !has_batchnorm) {
            // Single multiply (broadcast or equal shape)
            MetalKernelIR ir = build_kernel_ir_for_broadcast_mul(model);
            m_ir = std::move(ir);
            set_ops_from_ir();
            for (const auto& node : model->get_ordered_ops()) {
                if (auto mul = ov::as_type_ptr<const ov::op::v1::Multiply>(node)) {
                    if (auto c = std::dynamic_pointer_cast<const ov::op::v0::Constant>(mul->get_input_node_shared_ptr(1))) {
                        m_const_mul = c->cast_vector<float>();
                        m_has_const_mul = true;
                    }
                    break;
                }
            }
            m_pipelines.clear();
            m_pipelines.push_back(compiler.compile_mul_kernel(m_ops[0], log));
            return;
        }

        if (analysis.has_softmax && !has_matmul && !has_add && !has_add_broadcast && !analysis.has_unary) {
            if (try_build_softmax(model, analysis, compiler, log))
                return;
        }


        if (has_mul && (has_add || has_add_broadcast) && analysis.compute_ops == 2) {
            // Pattern: per-channel affine y = scale * x + shift (BatchNorm decomposed by front-end).
            // Re-map to BatchNorm2D kernel with mean=0, var=1, eps=0 so that
            // y = gamma * (x - 0) / sqrt(1 + 0) + beta = scale * x + shift.
            std::vector<float> mul_const, add_const;
            bool mul_const_found = false, add_const_found = false;
            for (const auto& node : model->get_ordered_ops()) {
                if (auto mul = ov::as_type_ptr<const ov::op::v1::Multiply>(node)) {
                    if (auto c = std::dynamic_pointer_cast<const ov::op::v0::Constant>(mul->get_input_node_shared_ptr(1))) {
                        mul_const = c->cast_vector<float>();
                        mul_const_found = true;
                    }
                } else if (auto add = ov::as_type_ptr<const ov::op::v1::Add>(node)) {
                    if (auto c = std::dynamic_pointer_cast<const ov::op::v0::Constant>(add->get_input_node_shared_ptr(1))) {
                        add_const = c->cast_vector<float>();
                        add_const_found = true;
                    }
                }
            }
            if (!mul_const_found || !add_const_found) {
                force_cpu_fallback(m_original_model);
                return;
            }

            MetalKernelIR ir;
            // Infer shapes from multiply input and add output
            std::shared_ptr<const ov::Node> mul_node;
            std::shared_ptr<const ov::Node> add_node;
            for (const auto& node : model->get_ordered_ops()) {
                if (!mul_node && ov::as_type_ptr<const ov::op::v1::Multiply>(node)) mul_node = node;
                if (!add_node && ov::as_type_ptr<const ov::op::v1::Add>(node)) add_node = node;
            }
            if (!mul_node || !add_node) {
                force_cpu_fallback(m_original_model);
                return;
            }
            const auto in_shape = mul_node->get_input_shape(0);  // assume NCHW
            const auto out_shape = add_node->get_output_shape(0);
            KernelTensor in_t{"in", {in_shape.begin(), in_shape.end()}};
            KernelTensor out_t{"out", {out_shape.begin(), out_shape.end()}};
            ir.tensors.push_back(in_t);
            ir.tensors.push_back(out_t);

            KernelOp bn_op;
            bn_op.kind = KernelOpKind::BatchNorm2D;
            bn_op.input0 = &ir.tensors[0];
            bn_op.output = &ir.tensors[1];
            OPENVINO_ASSERT(in_shape.size() == 4, "Mul+Add BN pattern expects rank-4 input");
            const size_t C = in_shape[1];
            OPENVINO_ASSERT(mul_const.size() == C || mul_const.size() == 1,
                            "Mul constant size must match channels or be scalar");
            OPENVINO_ASSERT(add_const.size() == C || add_const.size() == 1,
                            "Add constant size must match channels or be scalar");
            bn_op.batchnorm.N = static_cast<uint32_t>(in_shape[0]);
            bn_op.batchnorm.C = static_cast<uint32_t>(C);
            bn_op.batchnorm.H = static_cast<uint32_t>(in_shape[2]);
            bn_op.batchnorm.W = static_cast<uint32_t>(in_shape[3]);
            bn_op.batchnorm.eps = 0.f;

            bn_op.bn_params.assign(4 * C + 1, 0.f);
            auto fill_broadcast = [&](const std::vector<float>& src, size_t offset) {
                if (src.size() == 1) {
                    std::fill(bn_op.bn_params.begin() + offset, bn_op.bn_params.begin() + offset + C, src[0]);
                } else {
                    std::copy(src.begin(), src.end(), bn_op.bn_params.begin() + offset);
                }
            };
            fill_broadcast(mul_const, 0 * C);  // gamma/scale
            fill_broadcast(add_const, 1 * C);  // beta/shift
            std::fill(bn_op.bn_params.begin() + 2 * C, bn_op.bn_params.begin() + 3 * C, 0.f);  // mean=0
            std::fill(bn_op.bn_params.begin() + 3 * C, bn_op.bn_params.begin() + 4 * C, 1.f);  // var=1
            bn_op.bn_params[4 * C] = 0.f;  // eps

            ir.ops.push_back(bn_op);
            m_ir = std::move(ir);
            set_ops_from_ir();
            m_ops_from_flat_segment = true;
            if (!m_ops.empty()) {
                m_const_bn = m_ops[0].bn_params;
            }
            m_pipelines.clear();
            m_pipelines.push_back(compiler.compile_batchnorm2d_kernel(m_ops[0], log));
            return;
        }

        if (has_conv2d && !has_add && !has_add_broadcast && !analysis.has_unary && !analysis.has_softmax &&
            !has_maxpool && !has_avgpool && !has_batchnorm && !analysis.has_conv3d) {
            bool const_w = false;
            try {
                mlir::MLIRContext ctx;
                auto module = build_mlir_conv2d_from_model(model, ctx);
                run_mlir_pipeline(module);
            } catch (const std::exception& e) {
                METAL_LOGSTREAM_TRACE("mlir") << "[MlirBackend] Conv2D MLIR build failed, fallback: " << e.what() << "\n";
                force_cpu_fallback(m_original_model);
                return;
            }
            MetalKernelIR ir = build_kernel_ir_for_conv2d(model, const_w);
            m_ir = std::move(ir);
            set_ops_from_ir();
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
                if (conv_node) {
                    if (auto c = std::dynamic_pointer_cast<const ov::op::v0::Constant>(conv_node->get_input_node_shared_ptr(1))) {
                        m_const_w = c->cast_vector<float>();
                        m_has_const_w = true;
                        METAL_LOG_DEBUG("mlir", "[METAL MLIR] Conv2D const weights size=" + std::to_string(m_const_w.size()));

                        if (!m_const_w.empty()) {
                            const auto* wptr = m_const_w.data();
                            std::ostringstream oss;
                            oss << "[METAL MLIR] Conv2D W first: ";
                            for (size_t i = 0; i < std::min<size_t>(m_const_w.size(), 8); ++i) oss << wptr[i] << " ";
                            METAL_LOG_DEBUG("mlir", oss.str());
                        }

                    }
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
                    auto source = generate_msl_from_mlir(module, desc);
                    m_pipelines.push_back(compiler.compile_msl_from_source(source, "conv2d_kernel", log));
                } catch (const std::exception& e) {
                    METAL_LOG_DEBUG("mlir", std::string("[METAL MLIR] Conv2D MLIR→MSL failed, fallback to hand-written kernel: ") + e.what());
                    m_pipelines.push_back(compiler.compile_conv2d_kernel(m_ops[0], log));
                }
            }
            return;
        }

        if (has_conv2d || analysis.has_conv3d) {
            if (try_build_conv(model, analysis, compiler, log))
                return;
        }

        if (has_conv2d && analysis.has_unary && analysis.compute_ops == 2) {
            bool const_w = false;
            try {
                mlir::MLIRContext ctx;
                auto module = build_mlir_conv2d_from_model(model, ctx);
                run_mlir_pipeline(module);
            } catch (const std::exception& e) {
                METAL_LOGSTREAM_TRACE("mlir") << "[MlirBackend] Conv2D+Unary MLIR build failed, fallback: " << e.what() << "\n";
                force_cpu_fallback(m_original_model);
                return;
            }
            MetalKernelIR ir = build_kernel_ir_for_conv2d(model, const_w);
            // Append unary op
            KernelOp unary_op;
            unary_op.kind = KernelOpKind::Unary;
            unary_op.activation = analysis.unary_kind;
            unary_op.alpha = analysis.unary_alpha;
            ir.ops.push_back(unary_op);
            // Wire tensors: reuse conv output as unary input/output
            ir.ops[1].input0 = ir.ops[0].output;
            ir.ops[1].output = ir.ops[0].output;
            m_ir = std::move(ir);
            set_ops_from_ir();
            m_ops_from_flat_segment = true;

            if (const_w && !m_ops.empty()) {
                std::shared_ptr<const ov::Node> conv_node;
                for (const auto& node : model->get_ordered_ops()) {
                    if (auto c = ov::as_type_ptr<const ov::op::v1::Convolution>(node)) { conv_node = c; break; }
                    if (auto g = ov::as_type_ptr<const ov::op::v1::GroupConvolution>(node)) { conv_node = g; break; }
                }
                if (conv_node) {
                    if (auto c = std::dynamic_pointer_cast<const ov::op::v0::Constant>(conv_node->get_input_node_shared_ptr(1))) {
                        m_const_w = c->cast_vector<float>();
                        m_has_const_w = true;
                    }
                }
            }

            m_pipelines.clear();
            m_pipelines.push_back(compiler.compile_conv2d_kernel(m_ops[0], log));
            m_pipelines.push_back(compiler.compile_unary_kernel(m_ops[1], log));
            return;
        }

        if (has_conv2d && has_batchnorm && analysis.has_unary && analysis.compute_ops == 3) {
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
                force_cpu_fallback(m_original_model);
                return;
            }

            MetalKernelIR conv_ir = build_kernel_ir_for_conv2d(model, const_w);
            MetalKernelIR bn_ir = build_kernel_ir_for_batchnorm(model, const_bn);
            if (!const_bn) {
                force_cpu_fallback(m_original_model);
                return;
            }

            MetalKernelIR ir;
            // tensors: input, tmp1 (conv out), tmp2 (bn out)
            ir.tensors.push_back(conv_ir.tensors[0]);  // input
            KernelTensor tmp1 = {"tmp1", conv_ir.ops[0].output ? conv_ir.ops[0].output->shape : conv_ir.tensors[0].shape};
            KernelTensor tmp2 = {"tmp2", bn_ir.ops[0].output ? bn_ir.ops[0].output->shape : conv_ir.tensors[0].shape};
            ir.tensors.push_back(tmp1);
            ir.tensors.push_back(tmp2);

            KernelOp conv_op = conv_ir.ops[0];
            conv_op.input0 = &ir.tensors[0];
            conv_op.output = &ir.tensors[1];

            KernelOp bn_op = bn_ir.ops[0];
            bn_op.input0 = &ir.tensors[1];
            bn_op.output = &ir.tensors[2];

            KernelOp unary_op;
            unary_op.kind = KernelOpKind::Unary;
            unary_op.activation = analysis.unary_kind;
            unary_op.alpha = analysis.unary_alpha;
            unary_op.input0 = &ir.tensors[2];
            unary_op.output = &ir.tensors[2];

            ir.ops.push_back(conv_op);
            ir.ops.push_back(bn_op);
            ir.ops.push_back(unary_op);
            m_ir = std::move(ir);
            set_ops_from_ir();
            m_ops_from_flat_segment = true;

            if (const_w && !conv_ir.ops.empty()) {
                std::shared_ptr<const ov::Node> conv_node;
                for (const auto& node : model->get_ordered_ops()) {
                    if (auto c = ov::as_type_ptr<const ov::op::v1::Convolution>(node)) { conv_node = c; break; }
                    if (auto g = ov::as_type_ptr<const ov::op::v1::GroupConvolution>(node)) { conv_node = g; break; }
                }
                if (conv_node) {
                    if (auto c = std::dynamic_pointer_cast<const ov::op::v0::Constant>(conv_node->get_input_node_shared_ptr(1))) {
                        m_const_w = c->cast_vector<float>();
                        m_has_const_w = true;
                    }
                }
            }
            if (const_bn && !bn_ir.ops.empty()) {
                m_const_bn = bn_ir.ops[0].bn_params;
            }

            m_pipelines.clear();
            m_pipelines.push_back(compiler.compile_conv2d_kernel(m_ops[0], log));
            m_pipelines.push_back(compiler.compile_batchnorm2d_kernel(m_ops[1], log));
            m_pipelines.push_back(compiler.compile_unary_kernel(m_ops[2], log));
            return;
        }

#if OV_HAS_LAYER_NORM
#endif

        if (analysis.has_maxpool || analysis.has_avgpool) {
            if (try_build_pool(model, analysis, compiler, log))
                return;
        }

        if (analysis.has_batchnorm) {
            if (try_build_batchnorm(model, analysis, compiler, log))
                return;
        }

        if (analysis.has_unary && analysis.compute_ops == 1) {
            if (try_build_unary(model, analysis, compiler, log))
                return;
        }

        if (m_ops.empty()) {
            set_ops_from_ir();
            if (!m_ops.empty())
                return;
            if (analysis.compute_ops == 0) {
                METAL_LOG_DEBUG("mlir", "[MlirBackend] shape-only graph; no compute ops to lower");
                // Leave m_ops empty so run() can handle reshape/transpose host path without fallback.
                return;
            }
            METAL_LOG_DEBUG("mlir", "[MlirBackend] Fallback: no supported template patterns matched (no IR ops)");
            force_cpu_fallback(m_original_model);
            return;
        }
        OPENVINO_THROW("MlirBackend: no supported ops found for METAL backend");
    }
