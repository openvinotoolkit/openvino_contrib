void MlirBackend::Impl::run(const std::vector<ov::Tensor>& orig_inputs, std::vector<ov::Tensor>& outputs) {
        m_last_profiling.clear();
        // Ensure all inputs/outputs have storage (_impl) to avoid runtime null checks.
        auto ensure_tensor_init = [](const ov::Tensor& t, const ov::PartialShape& ps, ov::element::Type et) -> ov::Tensor {
            if (t && t.data())
                return t;
            ov::Shape sh;
            if (ps.is_static())
                sh = ps.to_shape();
            else
                sh = ov::Shape{1};
            return ov::Tensor{et, sh};
        };
        std::vector<ov::Tensor> safe_inputs = orig_inputs;
        const auto model_inputs = m_original_model->inputs();
        for (size_t i = 0; i < safe_inputs.size() && i < model_inputs.size(); ++i) {
            auto et = model_inputs[i].get_element_type();
            safe_inputs[i] = ensure_tensor_init(safe_inputs[i], model_inputs[i].get_partial_shape(), et);
        }
        if (outputs.empty())
            outputs.resize(m_original_model->outputs().size());
        const auto model_outputs = m_original_model->outputs();
        for (size_t i = 0; i < outputs.size() && i < model_outputs.size(); ++i) {
            auto et = model_outputs[i].get_element_type();
            outputs[i] = ensure_tensor_init(outputs[i], model_outputs[i].get_partial_shape(), et);
        }
        for (size_t i = 0; i < safe_inputs.size(); ++i) {
            OPENVINO_ASSERT(safe_inputs[i].data(), "METAL backend: input tensor ", i, " is uninitialized");
        }
        for (size_t i = 0; i < outputs.size(); ++i) {
            OPENVINO_ASSERT(outputs[i].data(), "METAL backend: output tensor ", i, " is uninitialized at entry");
        }
        const std::vector<ov::Tensor>& inputs = safe_inputs;  // use sanitized inputs below
        m_last_inputs = inputs;

        MetalBufferManager* mgr = m_active_mgr;
        if (!mgr) {
            m_owned_mgr = std::make_shared<MetalBufferManager>(m_device);
            mgr = m_owned_mgr.get();
        }
        OPENVINO_ASSERT(mgr, "MetalBufferManager is not available");

        const bool safe_debug = metal_safe_debug_enabled();
        auto cpu_fallback_error = [&]() {
            OPENVINO_THROW("METAL: CPU fallback is disabled");
        };
        const auto& run_inputs = safe_inputs;

        auto run_fallback = [&](const char* reason) {
            METAL_LOG_ERROR("fallback", std::string("CPU fallback disabled: ") + (reason ? reason : "unknown"));
            OPENVINO_THROW("METAL: CPU fallback disabled");
        };

        METAL_LOG_DEBUG("mlir", std::string("[METAL MLIR] run: force_fallback=") + (m_force_fallback ? "true" : "false"));
        if (m_softmax_dynamic_only) {
            if (run_host_softmax(run_inputs, outputs))
                return;
        }
        if (m_force_fallback) {
            run_fallback("force_fallback flag");
            return;
        }

        // Patch missing dtype info (common for dynamic-typed test graphs).
        for (auto& op : m_ops) {
            if (op.kind == KernelOpKind::Softmax) {
                if (op.dtype.ov_type == ov::element::dynamic) {
                    try {
                        op.dtype = resolve_metal_dtype(m_inference_precision);
                    } catch (const ov::Exception&) {
                        // leave as dynamic; will fallback
                    }
                }
                auto ty = static_cast<ov::element::Type_t>(op.element_type);
                if (ty == ov::element::Type_t::dynamic) {
                    op.element_type = static_cast<uint32_t>(static_cast<ov::element::Type_t>(m_inference_precision));
                }
            }
        }

        // Prefer previously built flat/template segment; otherwise CPU fallback.
        if (m_fallback_request) {
            run_fallback("fallback request already prepared");
            return;
        }

        const auto& ops = m_ops;
        if (ops.empty() || m_segments.empty()) {
            OPENVINO_THROW("METAL: no GPU segments available (CPU fallback disabled)");
        }

        // Execute first segment (may be full flat IR with many ops)
        const Segment& seg = m_segments.front();
        if (seg.op_count == 0 || seg.first_op_index + seg.op_count > ops.size()) {
            METAL_LOG_DEBUG("mlir", "[METAL MLIR] Segment guard failed, falling back to CPU");
            run_fallback("segment guard failed");
            return;
        }
        if (m_pipelines.empty() || m_pipelines.size() < seg.first_op_index + seg.op_count) {
            METAL_LOG_DEBUG("mlir", "[METAL MLIR] Missing pipelines for segment, fallback to CPU");
            run_fallback("pipelines missing for segment");
            return;
        }

        auto seg_op = [&](size_t i) -> const KernelOp& { return ops[seg.first_op_index + i]; };
        auto seg_pipe = [&](size_t i) -> id<MTLComputePipelineState> { return m_pipelines[seg.first_op_index + i]; };

        size_t expected_inputs = inputs.size();  // allow variable inputs for flat execution
        const auto& ops_ref = ops;  // alias to suppress shadowing warnings
        auto first_kind = seg_op(0).kind;

        // Split/Slice special-cases
        if (try_run_split_segment(
                seg, ops, inputs, outputs, *mgr, [&](const std::string& msg){ cpu_fallback_error(); }))
            return;

        if (outputs.size() != 1) {
            bool allow_multi_output = false;
            for (size_t i = 0; i < seg.op_count; ++i) {
                const auto kind = seg_op(i).kind;
                if (kind == KernelOpKind::Split || kind == KernelOpKind::Slice) {
                    allow_multi_output = true;
                    break;
                }
            }
            OPENVINO_ASSERT(allow_multi_output, "MlirBackend: expected 1 output");
        }

        auto inputs_ok = [](const std::vector<ov::Tensor>& ts) {
            for (const auto& t : ts) {
                if (t.get_element_type() != ov::element::f32)
                    continue;
                const float* p = t.data<const float>();
                for (size_t i = 0; i < t.get_size(); ++i) {
                    if (!std::isfinite(p[i])) {
                        return false;
                    }
                }
            }
            return true;
        };
        if (!inputs_ok(inputs)) {
            METAL_LOG_DEBUG("mlir", "[MlirBackend] Inputs non-finite; attempting CPU fallback");
            if (m_allow_partial_offload) {
                ensure_fallback(m_original_model);
                run_fallback("inputs non-finite");
                return;
            } else {
                cpu_fallback_error();
            }
        }

        // Ensure outputs container/tensor exist before any allocation logic below.
        if (outputs.empty()) outputs.resize(1);
        if (!outputs[0].data()) {
            auto ot = m_original_model->output(0).get_element_type();
            ov::Shape osh;
            if (m_original_model->output(0).get_partial_shape().is_static())
                osh = m_original_model->output(0).get_shape();
            if (osh.empty()) osh = {1};  // scalar placeholder
            outputs[0] = ov::Tensor{ot, osh};
        }

        auto resize_output = [&](const std::vector<size_t>& out_shape, ov::element::Type et) {
            if (outputs.empty())
                return;
            ov::Shape osh(out_shape.begin(), out_shape.end());
            if (outputs[0].get_shape() != osh || outputs[0].get_element_type() != et) {
                outputs[0] = ov::Tensor{et, osh};
            }
        };
        auto resize_output_i64 = [&](const std::vector<int64_t>& out_shape, ov::element::Type et) {
            std::vector<size_t> tmp;
            tmp.reserve(out_shape.size());
            for (auto d : out_shape) tmp.push_back(static_cast<size_t>(d));
            resize_output(tmp, et);
        };

        auto shape_from_tensor = [](const ov::Tensor& t) {
            const auto& s = t.get_shape();
            return std::vector<int64_t>(s.begin(), s.end());
        };

        // Runtime-shape aware op parameters (all segment lengths).
        std::vector<KernelOp> runtime_ops;
        runtime_ops.reserve(seg.op_count);
        bool runtime_shapes_ok = true;
        std::string runtime_fail_reason;
        std::unordered_map<const KernelTensor*, uint64_t> concat_axis_offset;

        auto update_output_shape = [](KernelOp& op, const std::vector<int64_t>& shp) {
            if (op.output) {
                op.output->shape = shp;
            }
            if (op.kind == KernelOpKind::ElementwiseAdd || op.kind == KernelOpKind::ElementwiseSub ||
                op.kind == KernelOpKind::ElementwiseMul || op.kind == KernelOpKind::ElementwiseDiv ||
                op.kind == KernelOpKind::ElementwisePow || op.kind == KernelOpKind::ElementwiseMod ||
                op.kind == KernelOpKind::ElementwiseFloorMod) {
                op.out_shape = shp;
            }
        };

        auto compute_conv2d_out_shape = [](int64_t H,
                                           int64_t W,
                                           const KernelOp& op) -> std::pair<int64_t, int64_t> {
            const auto& c = op.conv2d;
            const int64_t one = 1;
            const int64_t kh_minus_1 = static_cast<int64_t>(c.kernelH) - one;
            const int64_t kw_minus_1 = static_cast<int64_t>(c.kernelW) - one;
            const int64_t eff_filterH = static_cast<int64_t>(c.dilationH) * kh_minus_1;
            const int64_t eff_filterW = static_cast<int64_t>(c.dilationW) * kw_minus_1;
            const int64_t H_pad = H + static_cast<int64_t>(c.padTop) + static_cast<int64_t>(c.padBottom);
            const int64_t W_pad = W + static_cast<int64_t>(c.padLeft) + static_cast<int64_t>(c.padRight);
            const int64_t outH = (H_pad - eff_filterH - one) / static_cast<int64_t>(c.strideH) + one;
            const int64_t outW = (W_pad - eff_filterW - one) / static_cast<int64_t>(c.strideW) + one;
            return {outH, outW};
        };
        auto compute_pool2d_out_shape = [](int64_t H,
                                           int64_t W,
                                           const KernelOp& op) -> std::pair<int64_t, int64_t> {
            const auto& p = op.pool;
            const int64_t one = 1;
            const int64_t kh_minus_1 = static_cast<int64_t>(p.kernelH) - one;
            const int64_t kw_minus_1 = static_cast<int64_t>(p.kernelW) - one;
            const int64_t eff_filterH = static_cast<int64_t>(p.dilationH) * kh_minus_1;
            const int64_t eff_filterW = static_cast<int64_t>(p.dilationW) * kw_minus_1;
            const int64_t H_pad = H + static_cast<int64_t>(p.padTop) + static_cast<int64_t>(p.padBottom);
            const int64_t W_pad = W + static_cast<int64_t>(p.padLeft) + static_cast<int64_t>(p.padRight);
            const int64_t outH = (H_pad - eff_filterH - one) / static_cast<int64_t>(p.strideH) + one;
            const int64_t outW = (W_pad - eff_filterW - one) / static_cast<int64_t>(p.strideW) + one;
            return {outH, outW};
        };

        int64_t sm_rows = 0, sm_cols = 0, sm_inner = 0;
        bool sm_params_ok = false;

        auto softmax_params_from_shape = [&](const KernelOp& op,
                                             const std::vector<int64_t>& shape) -> std::tuple<int64_t, int64_t, int64_t, bool> {
            if (shape.empty() || op.softmax_axis == -1)
                return {0, 0, 0, false};
            int64_t rank = static_cast<int64_t>(shape.size());
            int64_t axis = op.softmax_axis;
            if (axis < 0) axis += rank;
            if (axis < 0 || axis >= rank)
                return {0, 0, 0, false};
            int64_t cols = shape[static_cast<size_t>(axis)];
            int64_t inner = 1;
            for (int64_t i = axis + 1; i < rank; ++i) inner *= shape[static_cast<size_t>(i)];
            int64_t outer = 1;
            for (int64_t i = 0; i < axis; ++i) outer *= shape[static_cast<size_t>(i)];
            int64_t rows = outer * inner;
            return {rows, cols, inner, true};
        };

        auto tensor_shape = [](const KernelTensor* t) -> std::vector<int64_t> {
            if (!t) return {};
            return t->shape;
        };

        // Map parameter nodes to incoming tensors to keep binding stable even if the first op does not consume input0.
        std::unordered_map<const ov::Node*, ov::Tensor> param_tensors;
        const auto seg_model_inputs = m_model->inputs();
        for (size_t i = 0; i < inputs.size() && i < seg_model_inputs.size(); ++i) {
            param_tensors[seg_model_inputs[i].get_node()] = inputs[i];
        }

        auto sync_param_shape = [&](KernelTensor* kt) {
            if (!kt || !kt->from_parameter || !kt->source_node)
                return;
            auto it = param_tensors.find(static_cast<const ov::Node*>(kt->source_node));
            if (it == param_tensors.end())
                return;
            kt->shape = shape_from_tensor(it->second);
        };
        sync_param_shape(seg_op(0).input0);
        sync_param_shape(seg_op(0).input1);

        for (size_t idx = 0; idx < seg.op_count; ++idx) {
            KernelOp op_rt = seg_op(idx);
            const auto cur_in0 = tensor_shape(op_rt.input0);
            const auto cur_in1 = tensor_shape(op_rt.input1);

            switch (op_rt.kind) {
                case KernelOpKind::MatMul: {
                    // Support batched MatMul: A[..., M, K] x B[..., K, N]
                    auto collapse = [](const std::vector<int64_t>& s) {
                        int64_t batch = 1;
                        for (size_t i = 0; i + 2 < s.size(); ++i) batch *= s[i];
                        return batch;
                    };
                    if (cur_in0.size() < 2 || (!cur_in1.empty() && cur_in1.size() < 2)) {
                        runtime_shapes_ok = false;
                        runtime_fail_reason = "MatMul rank <2";
                        break;
                    }
                    bool ta = op_rt.a_transpose;
                    bool tb = op_rt.b_transpose;
                    int64_t M = ta ? cur_in0.back() : cur_in0[cur_in0.size() - 2];
                    int64_t K = ta ? cur_in0[cur_in0.size() - 2] : cur_in0.back();
                    int64_t N = op_rt.N;
                    if (!cur_in1.empty()) {
                        int64_t kb = tb ? cur_in1.back() : cur_in1[cur_in1.size() - 2];
                        int64_t nb = tb ? cur_in1[cur_in1.size() - 2] : cur_in1.back();
                        if (kb != K) {
                            METAL_LOG_WARN("mlir", "[METAL MLIR] MatMul K mismatch: input K=" + std::to_string(K) +
                                                   " rhs K=" + std::to_string(kb) + " — using rhs K");
                            K = kb;
                        }
                        N = nb;
                    } else {
                        if (op_rt.K != 0 && op_rt.K != K) {
                            METAL_LOG_WARN("mlir", "[METAL MLIR] MatMul preload K mismatch: compiled K=" +
                                                   std::to_string(op_rt.K) + " runtime K=" + std::to_string(K));
                        }
                    }
                    op_rt.batch_a = collapse(cur_in0);
                    op_rt.batch_b = cur_in1.empty() ? 1 : collapse(cur_in1);
                    op_rt.batch = std::max(op_rt.batch_a, op_rt.batch_b);
                    op_rt.M = M;
                    op_rt.K = K;
                    op_rt.N = N;
                    std::vector<int64_t> out_shape;
                    if (op_rt.batch == 1) {
                        out_shape = {M, N};
                    } else {
                        out_shape = {op_rt.batch, M, N};
                    }
                    update_output_shape(op_rt, out_shape);
                    break;
                }
                case KernelOpKind::Conv2D: {
                    if (cur_in0.size() != 4) {
                        runtime_shapes_ok = false;
                        runtime_fail_reason = "Conv2D rank!=4";
                        break;
                    }
                    int64_t N = cur_in0[0];
                    int64_t C = cur_in0[1];
                    int64_t H = cur_in0[2];
                    int64_t W = cur_in0[3];
                    auto& c = op_rt.conv2d;
                    if (C != static_cast<int64_t>(c.C_in)) {
                        METAL_LOG_WARN("mlir", "[METAL MLIR] Conv2D runtime C mismatch: input C=" +
                                                  std::to_string(C) + " expected=" + std::to_string(c.C_in) +
                                                  " — proceeding with compiled channels");
                        C = static_cast<int64_t>(c.C_in);
                    }
                    const auto [outH, outW] = compute_conv2d_out_shape(H, W, op_rt);
                    if (outH <= 0 || outW <= 0) { runtime_shapes_ok = false; runtime_fail_reason = "Conv2D invalid output shape"; break; }
                    c.N = static_cast<uint32_t>(N);
                    c.H = static_cast<uint32_t>(H);
                    c.W = static_cast<uint32_t>(W);
                    c.outH = static_cast<uint32_t>(outH);
                    c.outW = static_cast<uint32_t>(outW);
                    op_rt.conv2d = c;
                    std::vector<int64_t> out_shape{N, static_cast<int64_t>(c.C_out), outH, outW};
                    update_output_shape(op_rt, out_shape);
                    break;
                }
                case KernelOpKind::Conv3D: {
                    if (cur_in0.size() != 5) {
                        runtime_shapes_ok = false;
                        runtime_fail_reason = "Conv3D rank!=5";
                        break;
                    }
                    int64_t N = cur_in0[0];
                    int64_t C = cur_in0[1];
                    int64_t D = cur_in0[2];
                    int64_t H = cur_in0[3];
                    int64_t W = cur_in0[4];
                    auto& c = op_rt.conv3d;
                    if (C != static_cast<int64_t>(c.C_in)) { runtime_shapes_ok = false; runtime_fail_reason = "Conv3D C mismatch"; break; }
                    int64_t outD = (D + c.padFront + c.padBack - static_cast<int64_t>(c.dilationD) * (c.kernelD - 1) - 1) /
                                   static_cast<int64_t>(c.strideD) + 1;
                    int64_t outH = (H + c.padTop + c.padBottom - static_cast<int64_t>(c.dilationH) * (c.kernelH - 1) - 1) /
                                   static_cast<int64_t>(c.strideH) + 1;
                    int64_t outW = (W + c.padLeft + c.padRight - static_cast<int64_t>(c.dilationW) * (c.kernelW - 1) - 1) /
                                   static_cast<int64_t>(c.strideW) + 1;
                    if (outD <= 0 || outH <= 0 || outW <= 0) { runtime_shapes_ok = false; runtime_fail_reason = "Conv3D invalid output shape"; break; }
                    c.N = static_cast<uint32_t>(N);
                    c.D = static_cast<uint32_t>(D);
                    c.H = static_cast<uint32_t>(H);
                    c.W = static_cast<uint32_t>(W);
                    c.outD = static_cast<uint32_t>(outD);
                    c.outH = static_cast<uint32_t>(outH);
                    c.outW = static_cast<uint32_t>(outW);
                    op_rt.conv3d = c;
                    std::vector<int64_t> out_shape{N, static_cast<int64_t>(c.C_out), outD, outH, outW};
                    update_output_shape(op_rt, out_shape);
                    break;
                }
                case KernelOpKind::ElementwiseAdd:
                case KernelOpKind::ElementwiseSub:
                case KernelOpKind::ElementwiseMul:
                case KernelOpKind::ElementwiseDiv:
                case KernelOpKind::ElementwisePow:
                case KernelOpKind::ElementwiseMod:
                case KernelOpKind::ElementwiseFloorMod: {
                    auto to_shape = [](const std::vector<int64_t>& v) -> ov::Shape {
                        if (v.empty()) return {};
                        ov::Shape r;
                        r.reserve(v.size());
                        for (auto d : v) r.push_back(static_cast<size_t>(d < 0 ? 1 : d));
                        return r;
                    };
                    auto br_rt = compute_broadcast(to_shape(cur_in0), to_shape(cur_in1));
                    if (!br_rt.success) {
                        runtime_shapes_ok = false;
                        runtime_fail_reason = "Eltwise broadcast mismatch";
                        break;
                    }
                    std::vector<int64_t> out_shape(br_rt.out_shape.begin(), br_rt.out_shape.end());
                    if (out_shape.empty()) out_shape.push_back(1);  // scalar as [1] for storage/strides
                    op_rt.out_shape = out_shape;
                    op_rt.stride0 = br_rt.stride0;
                    op_rt.stride1 = br_rt.stride1;
                    op_rt.is_broadcast = !(cur_in0 == out_shape && cur_in1 == out_shape);
                    update_output_shape(op_rt, out_shape);
                    break;
                }
                case KernelOpKind::Unary: {
                    if (cur_in0.empty()) {
                        runtime_shapes_ok = false;
                        runtime_fail_reason = "Unary missing input shape";
                        break;
                    }
                    update_output_shape(op_rt, cur_in0);
                    break;
                }
                case KernelOpKind::Softmax: {
                    if (std::getenv("METAL_MLIR_DEBUG")) {
                        std::ostringstream oss;
                        oss << "[METAL MLIR] softmax runtime axis=" << op_rt.softmax_axis << " shape=[";
                        for (size_t i = 0; i < cur_in0.size(); ++i) {
                            if (i) oss << ",";
                            oss << cur_in0[i];
                        }
                        oss << "]";
                        fprintf(stderr, "%s\n", oss.str().c_str());
                    }
                    auto params = softmax_params_from_shape(op_rt, cur_in0);
                    sm_rows = std::get<0>(params);
                    sm_cols = std::get<1>(params);
                    sm_inner = std::get<2>(params);
                    sm_params_ok = std::get<3>(params);
                    if (!sm_params_ok || sm_rows <= 0 || sm_cols <= 0) {
                        METAL_LOG_WARN("mlir", "[METAL MLIR] Softmax params not resolved at runtime; using compiled values");
                    }
                    op_rt.rows = sm_rows;
                    op_rt.cols = sm_cols;
                    op_rt.inner = sm_inner;
                    update_output_shape(op_rt, cur_in0);
                    break;
                }
                case KernelOpKind::Concat: {
                    if (cur_in0.empty() || !op_rt.output) {
                        runtime_shapes_ok = false;
                        runtime_fail_reason = "Concat missing shapes";
                        break;
                    }
                    const auto axis = op_rt.concat.axis;
                    if (axis < 0 || static_cast<size_t>(axis) >= cur_in0.size()) {
                        runtime_shapes_ok = false;
                        runtime_fail_reason = "Concat axis out of range";
                        break;
                    }
                    uint64_t axis_len = static_cast<uint64_t>(cur_in0[static_cast<size_t>(axis)]);
                    uint64_t axis_off = concat_axis_offset[op_rt.output];
                    concat_axis_offset[op_rt.output] = axis_off + axis_len;

                    uint64_t outer = 1, inner = 1;
                    for (size_t i = 0; i < static_cast<size_t>(axis); ++i) outer *= static_cast<uint64_t>(cur_in0[i]);
                    for (size_t i = static_cast<size_t>(axis) + 1; i < cur_in0.size(); ++i) inner *= static_cast<uint64_t>(cur_in0[i]);
                    op_rt.concat.outer = outer;
                    op_rt.concat.inner = inner;
                    if (op_rt.concat.axis_sizes.empty()) op_rt.concat.axis_sizes.resize(1);
                    if (op_rt.concat.axis_offsets.empty()) op_rt.concat.axis_offsets.resize(1);
                    op_rt.concat.axis_sizes[0] = axis_len;
                    op_rt.concat.axis_offsets[0] = axis_off;
                    if (op_rt.output && !op_rt.output->shape.empty() &&
                        static_cast<size_t>(axis) < op_rt.output->shape.size()) {
                        const int64_t total = op_rt.output->shape[static_cast<size_t>(axis)];
                        if (total > 0)
                            op_rt.concat.axis_total = static_cast<uint64_t>(total);
                    }
                    break;
                }
                case KernelOpKind::MaxPool2D:
                case KernelOpKind::AvgPool2D: {
                    if (cur_in0.size() != 4) {
                        runtime_shapes_ok = false;
                        runtime_fail_reason = "Pool rank!=4";
                        break;
                    }
                    int64_t N = cur_in0[0];
                    int64_t C = cur_in0[1];
                    int64_t H = cur_in0[2];
                    int64_t W = cur_in0[3];
                    auto& p = op_rt.pool;
                    const auto [outH, outW] = compute_pool2d_out_shape(H, W, op_rt);
                    if (outH <= 0 || outW <= 0) {
                        runtime_shapes_ok = false;
                        runtime_fail_reason = "Pool invalid output shape";
                        break;
                    }
                    p.N = static_cast<uint32_t>(N);
                    p.C = static_cast<uint32_t>(C);
                    p.H = static_cast<uint32_t>(H);
                    p.W = static_cast<uint32_t>(W);
                    p.outH = static_cast<uint32_t>(outH);
                    p.outW = static_cast<uint32_t>(outW);
                    op_rt.pool = p;
                    update_output_shape(op_rt, {N, C, outH, outW});
                    break;
                }
                case KernelOpKind::BatchNorm2D: {
                    if (cur_in0.size() != 4) {
                        runtime_shapes_ok = false;
                        runtime_fail_reason = "BatchNorm rank!=4";
                        break;
                    }
                    int64_t N = cur_in0[0];
                    int64_t C = cur_in0[1];
                    int64_t H = cur_in0[2];
                    int64_t W = cur_in0[3];
                    op_rt.batchnorm.N = static_cast<uint32_t>(N);
                    op_rt.batchnorm.C = static_cast<uint32_t>(C);
                    op_rt.batchnorm.H = static_cast<uint32_t>(H);
                    op_rt.batchnorm.W = static_cast<uint32_t>(W);
                    update_output_shape(op_rt, cur_in0);
                    break;
                }
                case KernelOpKind::Interpolate: {
                    if (cur_in0.size() != 4 || !op_rt.output) {
                        runtime_shapes_ok = false;
                        runtime_fail_reason = "Interpolate rank!=4";
                        break;
                    }
                    op_rt.interpolate.N = static_cast<uint32_t>(cur_in0[0]);
                    op_rt.interpolate.C = static_cast<uint32_t>(cur_in0[1]);
                    op_rt.interpolate.H_in = static_cast<uint32_t>(cur_in0[2]);
                    op_rt.interpolate.W_in = static_cast<uint32_t>(cur_in0[3]);
                    if (!op_rt.output->shape.empty() && op_rt.output->shape.size() == 4) {
                        op_rt.interpolate.H_out = static_cast<uint32_t>(op_rt.output->shape[2]);
                        op_rt.interpolate.W_out = static_cast<uint32_t>(op_rt.output->shape[3]);
                    }
                    if (op_rt.interpolate.H_in > 0)
                        op_rt.interpolate.scale_h = static_cast<float>(op_rt.interpolate.H_out) / op_rt.interpolate.H_in;
                    if (op_rt.interpolate.W_in > 0)
                        op_rt.interpolate.scale_w = static_cast<float>(op_rt.interpolate.W_out) / op_rt.interpolate.W_in;
                    update_output_shape(op_rt, op_rt.output->shape);
                    break;
                }
            }
            runtime_ops.push_back(op_rt);
            // Propagate updated runtime info back into m_ops for execution
            if (op_rt.kind == KernelOpKind::MatMul) {
                m_ops[seg.first_op_index + idx] = op_rt;
            }
            if (!runtime_shapes_ok) break;
        }

        if (!runtime_shapes_ok) {
            METAL_LOG_WARN("mlir", "[METAL MLIR] runtime shape inference incomplete: " + runtime_fail_reason);
            OPENVINO_THROW("METAL: runtime shape inference failed: ", runtime_fail_reason);
        }

        // Commit runtime-updated ops back to storage
        for (size_t i = 0; i < runtime_ops.size(); ++i) {
            m_ops[seg.first_op_index + i] = runtime_ops[i];
        }

        // Final output shape: use last op's output tensor if present.
        if (seg.op_count > 0) {
            const auto& last = m_ops[seg.first_op_index + seg.op_count - 1];
            if (last.output) {
                if (outputs.empty()) outputs.resize(1);
                auto final_shape = last.output->shape;
                if (final_shape.empty()) final_shape.push_back(1);  // scalar -> [1]
                ov::element::Type et = outputs[0].get_element_type();
                if (et == ov::element::dynamic)
                    et = last.output->dtype.ov_type;
                resize_output_i64(final_shape, et);
                last.output->shape = final_shape;  // keep IR consistent
                // Ensure OV tensor is allocated (resize_output_i64 may no-op if impl missing)
                if (!outputs[0].data()) {
                    outputs[0] = ov::Tensor(et, ov::Shape(final_shape.begin(), final_shape.end()));
                }
            }
        }

        if (!ops_ref.empty() && seg_op(0).kind == KernelOpKind::MatMul &&
            inputs[0].get_element_type() == ov::element::f32 &&
            inputs.size() > 1 && inputs[1].get_element_type() == ov::element::f32) {
            auto log_tensor = [](const char* tag, const ov::Tensor& t) {
                METAL_LOGSTREAM_TRACE("mlir") << "[MlirBackend run] " << tag << " size=" << t.get_size() << " first:";
                const float* p = t.data<const float>();
                for (size_t i = 0; i < std::min<size_t>(t.get_size(), 8); ++i) METAL_LOGSTREAM_TRACE("mlir") << " " << p[i];
                METAL_LOGSTREAM_TRACE("mlir") << "\n";
            };
            log_tensor("A", inputs[0]);
            if (inputs.size() > 1) log_tensor("B", inputs[1]);
        }

        const auto& pipelines = m_pipelines;
        if (pipelines.empty()) {
            run_fallback("pipelines empty before dispatch");
            return;
        }

        METAL_LOGSTREAM_TRACE("mlir") << "[MlirBackend run] segment ops=" << seg.op_count << " kinds:";
        for (size_t i = 0; i < seg.op_count; ++i) METAL_LOGSTREAM_TRACE("mlir") << " " << static_cast<int>(seg_op(i).kind);
        METAL_LOGSTREAM_TRACE("mlir") << "\n";

        auto tensor_num_elems = [](const KernelTensor* t) -> size_t {
            if (!t) return 0;
            if (t->shape.empty()) return 1;
            size_t elems = 1;
            for (auto d : t->shape) {
                if (d < 0)
                    continue;  // dynamic dim: treat as 1 for buffer sizing
                elems *= static_cast<size_t>(d);
            }
            return elems;
        };

        auto compute_size = [](const MetalDType& dt) -> size_t {
            switch (dt.compute) {
                case MetalDType::ComputeType::F32: return sizeof(float);
                case MetalDType::ComputeType::I32: return sizeof(int32_t);
                case MetalDType::ComputeType::I64: return sizeof(int64_t);
                default: return sizeof(float);
            }
        };
        auto element_size = [&](const KernelTensor* t) -> size_t {
            if (!t) return sizeof(float);
            if (t->dtype.ov_type == ov::element::dynamic)
                return sizeof(float);
            const size_t storage = storage_size(t->dtype);
            const size_t compute = compute_size(t->dtype);
            return std::max(storage, compute);
        };
        auto storage_elem_size = [&](const KernelTensor* t) -> size_t {
            if (!t) return sizeof(float);
            if (t->dtype.ov_type == ov::element::dynamic)
                return sizeof(float);
            return storage_size(t->dtype);
        };
        auto tensor_bytes = [&](const KernelTensor* t) -> size_t {
            return tensor_num_elems(t) * element_size(t);
        };

        auto dtype_to_element = [](const MetalDType& dt) -> ov::element::Type {
            switch (dt.storage) {
                case MetalDType::StorageType::I32: return ov::element::i32;
                case MetalDType::StorageType::I64: return ov::element::i64;
                case MetalDType::StorageType::F16: return ov::element::f16;
                case MetalDType::StorageType::F32:
                default: return ov::element::f32;
            }
        };

        auto to_mtl = [](const MetalBuffer& b) -> id<MTLBuffer> {
            return static_cast<id<MTLBuffer>>(b.buffer);
        };

        struct ResultKey {
            const ov::Node* node = nullptr;
            size_t index = 0;
            bool operator==(const ResultKey& other) const {
                return node == other.node && index == other.index;
            }
        };
        struct ResultKeyHash {
            size_t operator()(const ResultKey& k) const noexcept {
                return std::hash<const void*>()(k.node) ^ (k.index * 0x9e3779b97f4a7c15ULL);
            }
        };
        std::unordered_set<ResultKey, ResultKeyHash> result_keys;
        for (const auto& out : m_original_model->outputs()) {
            auto src = out.get_node()->input_value(0);
            result_keys.insert({src.get_node(), src.get_index()});
        }
        auto is_result_tensor = [&](const KernelTensor* t) -> bool {
            if (!t || !t->source_node)
                return false;
            ResultKey k{static_cast<const ov::Node*>(t->source_node), t->source_index};
            return result_keys.find(k) != result_keys.end();
        };

        // Buffer mapping for flat execution (device-resident via MetalBufferManager).
        std::unordered_map<const KernelTensor*, MetalBuffer> buf_map;

        // Reset temp buffers each run to avoid leaking/reusing handles across infer calls with
        // different shapes, and also when the buffer manager changes.
        if (m_temp_mgr != mgr) {
            m_temp_buffer_handles.clear();
        } else {
            m_temp_buffer_handles.clear();
        }
        m_temp_mgr = mgr;

        auto make_const_buffer = [&](const KernelTensor* t) -> MetalBuffer {
            return ensure_const_buffer(t, *mgr);
        };

        auto load_const_input = [&](const KernelOp& op) -> MetalBuffer {
            if (!op.input1 || !op.input1->from_constant)
                return MetalBuffer{};
            MetalBuffer buf = make_const_buffer(op.input1);
            return buf;
        };

        auto make_buffer_from_tensor = [&](const ov::Tensor& t) -> MetalBuffer {
            ov::element::Type et = t.get_element_type();
            const ov::Tensor& tmp = t;
            MetalBuffer buf = mgr->allocate(tmp.get_byte_size(), et, /*persistent=*/false, /*storageModePrivate=*/false);
            if (buf.valid()) {
                mgr->upload(buf, tmp.data(), tmp.get_byte_size());
                if (std::getenv("METAL_DEBUG_DUMP_OUTPUT")) {
                    auto* p = static_cast<const float*>([static_cast<id<MTLBuffer>>(buf.buffer) contents]);
                    if (p) {
                        fprintf(stderr, "[METAL][dbg] uploaded src first=%f size=%zu\n", p[0], tmp.get_size());
                    }
                }
            }
            return buf;
        };
        const std::vector<ov::Tensor>* in_vec = inputs.empty() && !m_last_inputs.empty() ? &m_last_inputs : &inputs;
        auto bind_param_buffer = [&](KernelTensor* kt) {
            if (!kt || !kt->from_parameter || !kt->source_node || !in_vec)
                return;
            auto it = param_tensors.find(static_cast<const ov::Node*>(kt->source_node));
            if (it == param_tensors.end())
                return;
            buf_map[kt] = make_buffer_from_tensor(it->second);
        };
        if (!in_vec->empty() && std::getenv("OV_METAL_DEBUG")) {
            const auto& in0 = (*in_vec)[0];
            std::ostringstream os;
            os << "[METAL MLIR] input0 shape=";
            auto s = in0.get_shape();
            os << "[";
            for (size_t i = 0; i < s.size(); ++i) {
                if (i) os << ",";
                os << s[i];
            }
            os << "]";
            METAL_LOG_DEBUG("mlir", os.str());
        }
        // Bind all parameter tensors from the map so ops can consume them regardless of order.
        for (auto& kt : m_ir.tensors) {
            if (kt.from_parameter)
                bind_param_buffer(&kt);
        }
        for (auto& kt : m_flat_tensors) {
            if (kt.from_parameter)
                bind_param_buffer(&kt);
        }
        if (seg.op_count == 1 && seg_op(0).kind == KernelOpKind::ElementwiseSub) {
            auto dump_i32 = [](const ov::Tensor& t, const char* tag) {
                if (t.get_element_type() != ov::element::i32)
                    return;
                auto log = METAL_LOGSTREAM_TRACE("mlir");
                log << "[EltwiseSub] " << tag << " shape=";
                auto s = t.get_shape();
                log << "[";
                for (size_t i = 0; i < s.size(); ++i) { if (i) log << ","; log << s[i]; }
                log << "] first:";
                const int32_t* p = t.data<const int32_t>();
                for (size_t i = 0; i < std::min<size_t>(t.get_size(), 8); ++i) log << " " << p[i];
                log << "\n";
            };
            if (!inputs.empty()) dump_i32(inputs[0], "A");
            if (inputs.size() > 1) dump_i32(inputs[1], "B");
            // Always emit a minimal debug line to help triage i32 subtract issues.
        }
        // Pre-create buffers for constant tensors referenced in the segment (helps broadcast const exponents).
        for (size_t i = seg.first_op_index; i < seg.first_op_index + seg.op_count; ++i) {
            const auto& op = m_ops[i];
            auto try_bind_const = [&](const KernelTensor* t) {
                if (!t || !t->from_constant || t->const_data.empty())
                    return;
                if (buf_map.find(t) != buf_map.end())
                    return;
                if (std::getenv("METAL_DEBUG_DUMP_OUTPUT")) {
                    fprintf(stderr, "[METAL][dbg] const tensor %s dtype=%s size=%zu first4:",
                            t->name.c_str(),
                            t->dtype.ov_type.get_type_name().c_str(),
                            t->const_data.size());
                    for (size_t k = 0; k < std::min<size_t>(4, t->const_data.size()); ++k)
                        fprintf(stderr, " %f", t->const_data[k]);
                    fprintf(stderr, "\n");
                }
                MetalBuffer buf = make_const_buffer(t);
                if (std::getenv("METAL_DEBUG_DUMP_OUTPUT")) {
                    fprintf(stderr, "[METAL][dbg] const device buf size=%zu type=%s\n",
                            buf.size, t->dtype.ov_type.get_type_name().c_str());
                }
                buf_map[t] = buf;
            };
            try_bind_const(op.input0);
            try_bind_const(op.input1);
        }
        auto get_buffer = [&](const KernelTensor* t) -> id<MTLBuffer> {
            auto it = buf_map.find(t);
            return it == buf_map.end() ? nil : to_mtl(it->second);
        };

        id<MTLCommandQueue> queue = m_queue;
#ifdef __OBJC__
        bool temp_queue = false;
        if (!queue) {
            queue = [m_device newCommandQueue];
            temp_queue = queue != nil;
        }
#endif
        // Always create a command buffer; profiling (if enabled later) will use GPU counters or timestamps,
        // but a nullptr command buffer leads to invalid handle fallback.
        id<MTLCommandBuffer> cmd = [queue commandBuffer];
#ifdef __OBJC__
        if (cmd)
            [cmd retain];
#endif
        ov::Shape final_shape = (!outputs.empty() && outputs[0]) ? outputs[0].get_shape() : ov::Shape{};
        ov::element::Type final_logical =
            (!outputs.empty() ? outputs[0].get_element_type() : ov::element::dynamic);

        const bool dbg_logs = std::getenv("OV_METAL_DEBUG") || std::getenv("METAL_MLIR_DEBUG");

        auto alloc_out_buffer = [&](const KernelOp& op, size_t slot, size_t fallback_bytes) -> id<MTLBuffer> {
            const KernelTensor* out_tensor = op.output ? op.output : op.input0;
            if (!out_tensor)
                return nil;
            auto it_existing = buf_map.find(out_tensor);
            if (it_existing != buf_map.end()) return to_mtl(it_existing->second);
            size_t bytes = tensor_bytes(out_tensor);
            if (bytes == 0) bytes = fallback_bytes;
            if (bytes == 0) bytes = 1;  // scalar
            ov::element::Type et = dtype_to_element(out_tensor->dtype);
            TempBufferKey key{op.kind, slot, et};
            auto& handle = m_temp_buffer_handles[key];
            const bool use_private = false;  // keep outputs host-visible (no CPU copies allowed)
            MetalBuffer buf = mgr->allocate_dynamic(bytes, et, handle, /*persistent=*/false, use_private);
            buf_map[out_tensor] = buf;
            return to_mtl(buf);
        };

        auto dump_shape = [&](const KernelTensor* t) -> std::string {
            if (!t) return "<null>";
            std::string s = "[";
            for (size_t i = 0; i < t->shape.size(); ++i) {
                if (i) s += ",";
                s += std::to_string(t->shape[i]);
            }
            s += "]";
            return s;
        };

        auto buf_len = [&](id<MTLBuffer> b) -> size_t {
#ifdef __OBJC__
            return b ? static_cast<size_t>([b length]) : 0;
#else
            (void)b;
            return 0;
#endif
        };

        auto find_buf = [&](const KernelTensor* t) -> MetalBuffer {
            auto it = buf_map.find(t);
            if (it != buf_map.end())
                return it->second;
            return MetalBuffer{};
        };

                auto dispatch_op = [&](const KernelOp& op,
                               id<MTLComputePipelineState> pipeline,
                               id<MTLBuffer> src0,
                               id<MTLBuffer> src1,
                               id<MTLBuffer> dst,
                               size_t src0_bytes,
                               const MetalBuffer& src0_buf,
                               id<MTLCommandBuffer> cmdBuf) {
            auto assert_capacity = [&](id<MTLBuffer> buf, size_t need, const char* tag) {
                if (!buf) {
                    OPENVINO_THROW(std::string("METAL: missing buffer for ") + tag);
                }
                const size_t cap = static_cast<size_t>([buf length]);
                OPENVINO_ASSERT(cap >= need, "METAL: buffer too small for ", tag, " need=", need, " have=", cap);
            };
            if (!cmdBuf || reinterpret_cast<uintptr_t>(cmdBuf) < 0x1000) {
                run_fallback("run_segment: invalid command buffer");
                return;
            }
            id<MTLComputeCommandEncoder> enc = [cmdBuf computeCommandEncoder];
#ifdef __OBJC__
            if (enc)
                [enc retain];
#endif
            if (!enc || reinterpret_cast<uintptr_t>(enc) < 0x1000) {
                if (enc) [enc release];
                run_fallback("run_segment: invalid compute encoder");
                return;
            }
            if (std::getenv("METAL_MLIR_DEBUG")) {
                fprintf(stderr, "[METAL MLIR] dispatch kind=%d cmd=%p enc=%p pipeline=%p\n",
                        static_cast<int>(op.kind), cmdBuf, enc, pipeline);
            }
            if (!pipeline) {
                METAL_LOG_DEBUG("mlir", "[METAL MLIR] Null pipeline in run_segment dispatch -> fallback");
                [enc endEncoding];
#ifdef __OBJC__
                if (enc) [enc release];
#endif
                m_force_fallback = true;
                ensure_fallback(m_original_model);
                return;
            }
            if (reinterpret_cast<uintptr_t>(pipeline) < 0x1000) {
                [enc endEncoding];
#ifdef __OBJC__
                if (enc) [enc release];
#endif
                run_fallback("run_segment: invalid pipeline handle");
                return;
            }
#ifdef __OBJC__
            if (pipeline && ![(id)pipeline respondsToSelector:@selector(device)]) {
                [enc endEncoding];
                [enc release];
                run_fallback("run_segment: pipeline is not an ObjC object");
                return;
            }
#endif
            [enc setComputePipelineState:pipeline];
            const NSUInteger max_tg = pipeline.maxTotalThreadsPerThreadgroup;
            if (dbg_logs) {
                const size_t have_in0 = src0_buf.valid() ? src0_buf.size : buf_len(src0);
                MetalBuffer src1_mb = find_buf(op.input1);
                const size_t have_in1 = src1_mb.valid() ? src1_mb.size : buf_len(src1);
                MetalBuffer dst_mb = find_buf(op.output ? op.output : op.input0);
                const size_t have_out = dst_mb.valid() ? dst_mb.size : buf_len(dst);
                fprintf(stderr,
                        "[METAL] Op=%s in0_shape=%s in1_shape=%s out_shape=%s buf_in0=%zu buf_in1=%zu buf_out=%zu\n",
                        kind_to_string(op.kind).c_str(),
                        dump_shape(op.input0).c_str(),
                        dump_shape(op.input1).c_str(),
                        dump_shape(op.output).c_str(),
                        have_in0, have_in1, have_out);
            }
            switch (op.kind) {
                case KernelOpKind::MatMul: {
                    if (!src0 || !src1 || !dst) {
                        METAL_LOG_ERROR("mlir", "[METAL MLIR] MatMul missing buffer — abort (CPU fallback disabled)");
                        [enc endEncoding];
#ifdef __OBJC__
                        if (enc) [enc release];
#endif
                        OPENVINO_THROW("METAL: missing device buffer for MatMul and CPU fallback is disabled");
                    }
                    const size_t elem_sz_out = storage_elem_size(op.output ? op.output : op.input0);
                    const size_t elem_sz_a = storage_elem_size(op.input0);
                    const size_t elem_sz_b = storage_elem_size(op.input1);
                    const size_t total = static_cast<size_t>(op.M * op.N * op.batch);
                    const size_t need_dst = total * elem_sz_out;
                    assert_capacity(dst, need_dst, "MatMul dst");
                    assert_capacity(src0, static_cast<size_t>(op.M * op.K * op.batch) * elem_sz_a, "MatMul A");
                    assert_capacity(src1, static_cast<size_t>(op.K * op.N * op.batch) * elem_sz_b, "MatMul B");
                    if (std::getenv("METAL_MLIR_DEBUG")) {
                        fprintf(stderr,
                                "[METAL MLIR] matmul dispatch M=%lld N=%lld K=%lld batch=%lld b_is_nk=%d a_t=%d b_t=%d\n",
                                static_cast<long long>(op.M),
                                static_cast<long long>(op.N),
                                static_cast<long long>(op.K),
                                static_cast<long long>(op.batch),
                                op.b_is_nk_layout ? 1 : 0,
                                op.a_transpose ? 1 : 0,
                                op.b_transpose ? 1 : 0);
                    }
                    [enc setBuffer:src0 offset:0 atIndex:0];
                    [enc setBuffer:src1 offset:0 atIndex:1];
                    [enc setBuffer:dst offset:0 atIndex:2];
                    const NSUInteger totalThreads = static_cast<NSUInteger>(op.M * op.N * op.batch);
                    const NSUInteger threads_per_tg = 128;
                    MTLSize grid = MTLSizeMake(totalThreads, 1, 1);
                    MTLSize tg = MTLSizeMake(std::min<NSUInteger>(threads_per_tg, max_tg), 1, 1);
                    if (dbg_logs) {
                        const size_t have_in0 = src0_buf.valid() ? src0_buf.size : buf_len(src0);
                        const MetalBuffer src1_mb = find_buf(op.input1);
                        const size_t have_in1 = src1_mb.valid() ? src1_mb.size : buf_len(src1);
                        MetalBuffer dst_mb = find_buf(op.output ? op.output : op.input0);
                        const size_t have_out = dst_mb.valid() ? dst_mb.size : buf_len(dst);
                        const uint64_t need_a = static_cast<uint64_t>(op.M) * op.K * op.batch * elem_sz_a;
                        const uint64_t need_b = static_cast<uint64_t>(op.K) * op.N * op.batch * elem_sz_b;
                        fprintf(stderr,
                                "[METAL DBG] MatMul M=%lld N=%lld K=%lld batch=%lld bytes A/B/out=%llu/%llu/%llu "
                                "buf=%zu/%zu/%zu grid=%lu tg=%lu\n",
                                static_cast<long long>(op.M),
                                static_cast<long long>(op.N),
                                static_cast<long long>(op.K),
                                static_cast<long long>(op.batch),
                                static_cast<unsigned long long>(need_a),
                                static_cast<unsigned long long>(need_b),
                                static_cast<unsigned long long>(need_dst),
                                have_in0, have_in1, have_out,
                                static_cast<unsigned long>(grid.width),
                                static_cast<unsigned long>(tg.width));
                    }
                    [enc dispatchThreads:grid threadsPerThreadgroup:tg];
                    break;
                }
                case KernelOpKind::Conv3D: {
                    METAL_LOGSTREAM_TRACE("mlir") << "[METAL MLIR] entering Conv3D case\n";
                    if (!src0 || !src1 || !dst) {
                        METAL_LOG_ERROR("mlir", "[METAL MLIR] Conv3D missing buffer — abort (CPU fallback disabled)");
                        [enc endEncoding];
#ifdef __OBJC__
                        if (enc) [enc release];
#endif
                        OPENVINO_THROW("METAL: missing device buffer for Conv3D and CPU fallback is disabled");
                    }
                    const auto& d = op.conv3d;
                    const size_t need_dst = static_cast<size_t>(d.outD) * d.outH * d.outW * d.C_out * d.N * sizeof(float);
                    assert_capacity(dst, need_dst, "Conv3D dst");
                    struct Conv3DParams {
                        uint32_t N, C_in, D, H, W;
                        uint32_t C_out;
                        uint32_t kD, kH, kW;
                        uint32_t strideD, strideH, strideW;
                        uint32_t dilationD, dilationH, dilationW;
                        uint32_t padFront, padTop, padLeft, padBack, padBottom, padRight;
                        uint32_t outD, outH, outW;
                    } params;
                    params.N = op.conv3d.N;
                    params.C_in = op.conv3d.C_in;
                    params.D = op.conv3d.D;
                    params.H = op.conv3d.H;
                    params.W = op.conv3d.W;
                    params.C_out = op.conv3d.C_out;
                    params.kD = op.conv3d.kernelD;
                    params.kH = op.conv3d.kernelH;
                    params.kW = op.conv3d.kernelW;
                    params.strideD = op.conv3d.strideD;
                    params.strideH = op.conv3d.strideH;
                    params.strideW = op.conv3d.strideW;
                    params.dilationD = op.conv3d.dilationD;
                    params.dilationH = op.conv3d.dilationH;
                    params.dilationW = op.conv3d.dilationW;
                    params.padFront = op.conv3d.padFront;
                    params.padTop = op.conv3d.padTop;
                    params.padLeft = op.conv3d.padLeft;
                    params.padBack = op.conv3d.padBack;
                    params.padBottom = op.conv3d.padBottom;
                    params.padRight = op.conv3d.padRight;
                    params.outD = op.conv3d.outD;
                    params.outH = op.conv3d.outH;
                    params.outW = op.conv3d.outW;
                    // Buffer capacity guards.
                    const size_t in_elem = storage_elem_size(op.input0);
                    const size_t w_elem = storage_elem_size(op.input1);
                    const size_t out_elem = storage_elem_size(op.output ? op.output : op.input0);
                    const uint64_t need_in = static_cast<uint64_t>(params.N) *
                                             params.C_in * params.D * params.H * params.W *
                                             static_cast<uint64_t>(in_elem);
                    const uint64_t need_w = static_cast<uint64_t>(params.C_out) *
                                            params.kD * params.kH * params.kW *
                                            params.C_in * static_cast<uint64_t>(w_elem);
                    const uint64_t need_out = static_cast<uint64_t>(params.N) *
                                              params.C_out * params.outD * params.outH * params.outW *
                                              static_cast<uint64_t>(out_elem);
                    const size_t have_in = src0_buf.valid() ? src0_buf.size : buf_len(src0);
                    const MetalBuffer w_mb = find_buf(op.input1);
                    const size_t have_w = w_mb.valid() ? w_mb.size : buf_len(src1);
                    MetalBuffer dst_mb = find_buf(op.output ? op.output : op.input0);
                    const size_t have_out = dst_mb.valid() ? dst_mb.size : buf_len(dst);
                    OPENVINO_ASSERT(need_in <= have_in,
                                    "METAL: Conv3D input buffer overflow: need=", need_in, " have=", have_in);
                    OPENVINO_ASSERT(need_w <= have_w,
                                    "METAL: Conv3D weights buffer overflow: need=", need_w, " have=", have_w);
                    OPENVINO_ASSERT(need_out <= have_out,
                                    "METAL: Conv3D output buffer overflow: need=", need_out, " have=", have_out,
                                    " out_shape=", dump_shape(op.output));
                    if (std::getenv("METAL_MLIR_DEBUG")) {
                        METAL_LOGSTREAM_TRACE("mlir") << "[METAL MLIR] conv3d params N=" << params.N
                                  << " C_in=" << params.C_in << " D/H/W=" << params.D << "/" << params.H << "/" << params.W
                                  << " C_out=" << params.C_out << " k=" << params.kD << "x" << params.kH << "x" << params.kW
                                  << " stride=" << params.strideD << "," << params.strideH << "," << params.strideW
                                  << " dilation=" << params.dilationD << "," << params.dilationH << "," << params.dilationW
                                  << " padF/T/L/B/Bot/R=" << params.padFront << "," << params.padTop << "," << params.padLeft
                                  << "/" << params.padBack << "," << params.padBottom << "," << params.padRight
                                  << " out=" << params.outD << "x" << params.outH << "x" << params.outW << "\n";
                    }
                    [enc setBuffer:src0 offset:0 atIndex:0];
                    [enc setBuffer:src1 offset:0 atIndex:1];
                    [enc setBuffer:dst offset:0 atIndex:2];
                    [enc setBytes:&params length:sizeof(params) atIndex:3];
                    MTLSize grid = MTLSizeMake(params.C_out, params.N, 1);
                    MTLSize tg = MTLSizeMake(1, 1, 1);
                    if (dbg_logs) {
                        fprintf(stderr, "[METAL DBG] Conv3D grid=(%u,%u,1) tg=(1,1,1)\n",
                                params.C_out, params.N);
                    }
                    [enc dispatchThreads:grid threadsPerThreadgroup:tg];
                    break;
                }
                case KernelOpKind::ElementwiseAdd:
                case KernelOpKind::ElementwiseSub:
                case KernelOpKind::ElementwiseMul:
                case KernelOpKind::ElementwiseDiv:
                case KernelOpKind::ElementwisePow:
                case KernelOpKind::ElementwiseMod:
                case KernelOpKind::ElementwiseFloorMod: {
                    if (!src0 || !src1 || !dst) {
                        METAL_LOG_ERROR("mlir", "[METAL MLIR] Elementwise op missing buffer — abort (CPU fallback disabled)");
                        [enc endEncoding];
                        OPENVINO_THROW("METAL: missing device buffer for elementwise op and CPU fallback is disabled");
                    }
                    size_t elem_count = tensor_num_elems(op.output);
                    if (elem_count == 0)
                        elem_count = tensor_num_elems(op.input0);
                    if (elem_count == 0 && src0_bytes > 0) {
                        size_t elem_sz = op.input0 ? element_size(op.input0) : sizeof(float);
                        if (elem_sz == 0) elem_sz = sizeof(float);
                        elem_count = std::max<size_t>(1, src0_bytes / elem_sz);
                    }
                    // Kernel signature: A[[0]], B[[1]], C[[2]], NUM_ELEMS[[3]], RANK[[4]], out_dims[[5]], stride0[[6]], stride1[[7]]
                    [enc setBuffer:src0 offset:0 atIndex:0];
                    [enc setBuffer:src1 offset:0 atIndex:1];
                    [enc setBuffer:dst offset:0 atIndex:2];
                    uint32_t num_elems = static_cast<uint32_t>(elem_count);
                    uint32_t rank = 1;
                    std::vector<int> out_dims;
                    std::vector<int> stride0;
                    std::vector<int> stride1;
                    if (op.output && !op.output->shape.empty()) {
                        rank = static_cast<uint32_t>(op.output->shape.size());
                        out_dims.reserve(rank);
                        stride0.reserve(rank);
                        stride1.reserve(rank);
                        for (auto d : op.output->shape) out_dims.push_back(static_cast<int>(d));
                        if (!op.stride0.empty() && op.stride0.size() == rank &&
                            !op.stride1.empty() && op.stride1.size() == rank) {
                            for (size_t k = 0; k < rank; ++k) {
                                stride0.push_back(static_cast<int>(op.stride0[k]));
                                stride1.push_back(static_cast<int>(op.stride1[k]));
                            }
                        } else {
                            // Non-broadcast case: contiguous row-major strides.
                            stride0.resize(rank, 1);
                            stride1.resize(rank, 1);
                            for (int i = static_cast<int>(rank) - 2; i >= 0; --i) {
                                stride0[static_cast<size_t>(i)] = stride0[static_cast<size_t>(i + 1)] * out_dims[static_cast<size_t>(i + 1)];
                                stride1[static_cast<size_t>(i)] = stride0[static_cast<size_t>(i)];
                            }
                        }
                    } else {
                        out_dims = {static_cast<int>(elem_count)};
                        stride0 = {1};
                        stride1 = {1};
                    }
                    [enc setBytes:&num_elems length:sizeof(num_elems) atIndex:3];
                    [enc setBytes:&rank length:sizeof(rank) atIndex:4];
                    [enc setBytes:out_dims.data() length:out_dims.size() * sizeof(int) atIndex:5];
                    [enc setBytes:stride0.data() length:stride0.size() * sizeof(int) atIndex:6];
                    [enc setBytes:stride1.data() length:stride1.size() * sizeof(int) atIndex:7];
                    // Buffer capacity checks: use tensor shapes when available, otherwise fall back to raw buffer lengths.
                    const size_t out_elem = storage_elem_size(op.output ? op.output : op.input0);
                    const uint64_t need_out = static_cast<uint64_t>(elem_count) * static_cast<uint64_t>(out_elem);
                    const size_t have_out = [&]() {
                        MetalBuffer dst_mb = find_buf(op.output ? op.output : op.input0);
                        return dst_mb.valid() ? dst_mb.size : buf_len(dst);
                    }();
                    auto max_offset = [&](const std::vector<int>& dims,
                                          const std::vector<int>& strides) -> uint64_t {
                        if (dims.empty() || strides.empty())
                            return elem_count ? static_cast<uint64_t>(elem_count - 1) : 0ULL;
                        uint64_t off = 0;
                        for (size_t k = 0; k < dims.size(); ++k) {
                            int d = dims[k] < 0 ? 0 : dims[k];
                            int s = (k < strides.size()) ? strides[k] : 0;
                            if (d > 0 && s > 0) {
                                off += static_cast<uint64_t>(d - 1) * static_cast<uint64_t>(s);
                            }
                        }
                        return off;
                    };
                    const uint64_t max0 = max_offset(out_dims, stride0);
                    const uint64_t max1 = max_offset(out_dims, stride1);
                    const size_t have_in0 = src0_buf.valid() ? src0_buf.size : buf_len(src0);
                    const MetalBuffer src1_buf = find_buf(op.input1);
                    const size_t have_in1 = src1_buf.valid() ? src1_buf.size : buf_len(src1);
                    const size_t in0_elem = storage_elem_size(op.input0);
                    const size_t in1_elem = storage_elem_size(op.input1);
                    const uint64_t need_in0 = (max0 + 1) * static_cast<uint64_t>(in0_elem);
                    const uint64_t need_in1 = (max1 + 1) * static_cast<uint64_t>(in1_elem);
                    OPENVINO_ASSERT(need_in0 <= have_in0,
                                    "METAL: Elementwise input0 overflow need=", need_in0, " have=", have_in0);
                    OPENVINO_ASSERT(need_in1 <= have_in1,
                                    "METAL: Elementwise input1 overflow need=", need_in1, " have=", have_in1);
                    OPENVINO_ASSERT(need_out <= have_out,
                                    "METAL: Elementwise output overflow need=", need_out, " have=", have_out,
                                    " out_shape=", dump_shape(op.output));
                    if (dbg_logs) {
                        fprintf(stderr,
                                "[METAL DBG] Eltwise kind=%d elems=%zu rank=%u out_bytes=%llu buf_out=%zu "
                                "in0_bytes=%llu/%zu in1_bytes=%llu/%zu\n",
                                static_cast<int>(op.kind),
                                elem_count,
                                rank,
                                static_cast<unsigned long long>(need_out),
                                have_out,
                                static_cast<unsigned long long>(need_in0), have_in0,
                                static_cast<unsigned long long>(need_in1), have_in1);
                    }
                    const NSUInteger elems = static_cast<NSUInteger>(elem_count);
                    const NSUInteger threads_per_tg = 64;
                    MTLSize grid = MTLSizeMake(elems, 1, 1);
                    MTLSize tg = MTLSizeMake(std::min<NSUInteger>(threads_per_tg, max_tg), 1, 1);
                    if (dbg_logs) {
                        fprintf(stderr, "[METAL DBG] Eltwise grid=%lu tg=%lu\n",
                                static_cast<unsigned long>(elems),
                                static_cast<unsigned long>(tg.width));
                    }
                    [enc dispatchThreads:grid threadsPerThreadgroup:tg];
                    break;
                }
                case KernelOpKind::Unary: {
                    if (!src0 || !dst) {
                        METAL_LOG_ERROR("mlir", "[METAL MLIR] Unary missing buffer — abort (CPU fallback disabled)");
                        [enc endEncoding];
                        OPENVINO_THROW("METAL: missing device buffer for unary op and CPU fallback is disabled");
                    }
                    const size_t in_elems = tensor_num_elems(op.input0);
                    const size_t out_elems = tensor_num_elems(op.output ? op.output : op.input0);
                    const size_t elem_in = storage_elem_size(op.input0);
                    const size_t elem_out = storage_elem_size(op.output ? op.output : op.input0);
                    const uint64_t need_in = static_cast<uint64_t>(in_elems) * static_cast<uint64_t>(elem_in);
                    const uint64_t need_out = static_cast<uint64_t>(out_elems) * static_cast<uint64_t>(elem_out);
                    const size_t have_in = src0_buf.valid() ? src0_buf.size : buf_len(src0);
                    MetalBuffer dst_mb = find_buf(op.output ? op.output : op.input0);
                    const size_t have_out = dst_mb.valid() ? dst_mb.size : buf_len(dst);
                    OPENVINO_ASSERT(need_in <= have_in,
                                    "METAL: Unary input buffer overflow need=", need_in, " have=", have_in);
                    OPENVINO_ASSERT(need_out <= have_out,
                                    "METAL: Unary output buffer overflow need=", need_out, " have=", have_out,
                                    " out_shape=", dump_shape(op.output));
                    [enc setBuffer:src0 offset:0 atIndex:0];
                    [enc setBuffer:dst offset:0 atIndex:1];
                    const NSUInteger elems = static_cast<NSUInteger>(tensor_num_elems(op.output));
                    OPENVINO_ASSERT(elems <= std::numeric_limits<uint32_t>::max(),
                                    "METAL: Unary element count exceeds uint32 range");
                    const uint32_t num_elems = static_cast<uint32_t>(elems);
                    [enc setBytes:&num_elems length:sizeof(num_elems) atIndex:2];
                    const NSUInteger threads_per_tg = 64;
                    MTLSize grid = MTLSizeMake(elems, 1, 1);
                    MTLSize tg = MTLSizeMake(std::min<NSUInteger>(threads_per_tg, max_tg), 1, 1);
                    if (dbg_logs) {
                        fprintf(stderr,
                                "[METAL DBG] Unary elems=%lu in_bytes=%llu out_bytes=%llu buf_in/out=%zu/%zu grid=%lu tg=%lu\n",
                                static_cast<unsigned long>(elems),
                                static_cast<unsigned long long>(need_in),
                                static_cast<unsigned long long>(need_out),
                                have_in, have_out,
                                static_cast<unsigned long>(grid.width),
                                static_cast<unsigned long>(tg.width));
                    }
                    [enc dispatchThreads:grid threadsPerThreadgroup:tg];
                    break;
                }
                case KernelOpKind::Convert: {
                    if (!src0 || !dst) {
                        METAL_LOG_ERROR("mlir", "[METAL MLIR] Convert missing buffer — abort (CPU fallback disabled)");
                        [enc endEncoding];
                        OPENVINO_THROW("METAL: missing device buffer for Convert and CPU fallback is disabled");
                    }
                    const size_t in_elems = tensor_num_elems(op.input0);
                    const size_t out_elems = tensor_num_elems(op.output ? op.output : op.input0);
                    const size_t elem_in = storage_elem_size(op.input0);
                    const size_t elem_out = storage_elem_size(op.output ? op.output : op.input0);
                    const uint64_t need_in = static_cast<uint64_t>(in_elems) * static_cast<uint64_t>(elem_in);
                    const uint64_t need_out = static_cast<uint64_t>(out_elems) * static_cast<uint64_t>(elem_out);
                    const size_t have_in = src0_buf.valid() ? src0_buf.size : buf_len(src0);
                    MetalBuffer dst_mb = find_buf(op.output ? op.output : op.input0);
                    const size_t have_out = dst_mb.valid() ? dst_mb.size : buf_len(dst);
                    OPENVINO_ASSERT(need_in <= have_in,
                                    "METAL: Convert input buffer overflow need=", need_in, " have=", have_in);
                    OPENVINO_ASSERT(need_out <= have_out,
                                    "METAL: Convert output buffer overflow need=", need_out, " have=", have_out,
                                    " out_shape=", dump_shape(op.output));
                    [enc setBuffer:src0 offset:0 atIndex:0];
                    [enc setBuffer:dst offset:0 atIndex:1];
                    const NSUInteger elems = static_cast<NSUInteger>(tensor_num_elems(op.output));
                    OPENVINO_ASSERT(elems <= std::numeric_limits<uint32_t>::max(),
                                    "METAL: Convert element count exceeds uint32 range");
                    const uint32_t num_elems = static_cast<uint32_t>(elems);
                    [enc setBytes:&num_elems length:sizeof(num_elems) atIndex:2];
                    const NSUInteger threads_per_tg = 256;
                    MTLSize grid = MTLSizeMake(elems, 1, 1);
                    MTLSize tg = MTLSizeMake(std::min<NSUInteger>(threads_per_tg, max_tg), 1, 1);
                    if (dbg_logs) {
                        fprintf(stderr,
                                "[METAL DBG] Convert elems=%lu in_bytes=%llu out_bytes=%llu buf_in/out=%zu/%zu grid=%lu tg=%lu\n",
                                static_cast<unsigned long>(elems),
                                static_cast<unsigned long long>(need_in),
                                static_cast<unsigned long long>(need_out),
                                have_in, have_out,
                                static_cast<unsigned long>(grid.width),
                                static_cast<unsigned long>(tg.width));
                    }
                    [enc dispatchThreads:grid threadsPerThreadgroup:tg];
                    break;
                }
                case KernelOpKind::Softmax: {
                    if (!src0 || !dst) {
                        METAL_LOG_ERROR("mlir", "[METAL MLIR] Softmax missing buffer — abort (CPU fallback disabled)");
                        [enc endEncoding];
                        OPENVINO_THROW("METAL: missing device buffer for Softmax and CPU fallback is disabled");
                    }
                    size_t rows = static_cast<size_t>(op.rows);
                    size_t cols = static_cast<size_t>(op.cols);
                    size_t inner = static_cast<size_t>(op.inner);
                    auto infer_rt_softmax_dims = [&]() {
                        auto recompute_from_shape = [&](const std::vector<int64_t>& shp) {
                            if (shp.empty())
                                return;
                            int64_t axis = op.softmax_axis;
                            const int64_t rank = static_cast<int64_t>(shp.size());
                            if (axis < 0)
                                axis += rank;
                            if (axis < 0 || axis >= rank)
                                return;
                            size_t outer_rt = 1;
                            for (int64_t k = 0; k < axis; ++k)
                                outer_rt *= static_cast<size_t>(std::max<int64_t>(1, shp[static_cast<size_t>(k)]));
                            size_t inner_rt = 1;
                            for (int64_t k = axis + 1; k < rank; ++k)
                                inner_rt *= static_cast<size_t>(std::max<int64_t>(1, shp[static_cast<size_t>(k)]));
                            size_t cols_rt = static_cast<size_t>(std::max<int64_t>(1, shp[static_cast<size_t>(axis)]));
                            rows = outer_rt * inner_rt;
                            cols = cols_rt;
                            inner = inner_rt;
                        };
                        if (op.input0 && !op.input0->shape.empty()) {
                            recompute_from_shape(op.input0->shape);
                            return;
                        }
                        if (in_vec && !in_vec->empty() && (*in_vec)[0]) {
                            const auto& shp = (*in_vec)[0].get_shape();
                            std::vector<int64_t> tmp(shp.begin(), shp.end());
                            recompute_from_shape(tmp);
                        }
                    };
                    infer_rt_softmax_dims();
                    if (rows == 0 || cols == 0) {
                        rows = static_cast<size_t>(op.rows);
                        cols = static_cast<size_t>(op.cols);
                        inner = static_cast<size_t>(op.inner);
                    }
                    if (rows == 0 || cols == 0) {
                        [enc endEncoding];
                        OPENVINO_THROW("METAL: Softmax runtime dims unresolved (rows/cols == 0)");
                    }
                    // Ensure output buffer is large enough for dynamic shapes.
                    size_t softmax_elems = rows * cols;
                    size_t elem_sz_dyn = element_size(op.output ? op.output : op.input0);
                    if (elem_sz_dyn == 0) elem_sz_dyn = sizeof(float);
                    size_t need_bytes_dyn = softmax_elems * elem_sz_dyn;
                    auto it_dst = buf_map.find(op.output ? op.output : op.input0);
                    if (it_dst == buf_map.end() || it_dst->second.size < need_bytes_dyn) {
                        ov::element::Type et = dtype_to_element(op.output ? op.output->dtype : op.input0->dtype);
                        size_t slot = reinterpret_cast<size_t>(op.output ? op.output : op.input0);
                        TempBufferKey key{KernelOpKind::Softmax, slot, et};
                        auto& handle = m_temp_buffer_handles[key];
                        MetalBuffer new_dst =
                            mgr->allocate_dynamic(need_bytes_dyn, et, handle, /*persistent=*/false,
                                                  /*storageModePrivate=*/false);
                        if (op.output)
                            buf_map[op.output] = new_dst;
                        dst = to_mtl(new_dst);
                    } else {
                        dst = to_mtl(it_dst->second);
                    }
                    if (std::getenv("METAL_MLIR_DEBUG")) {
                        fprintf(stderr, "[METAL MLIR] softmax dispatch rows=%zu cols=%zu inner=%zu total=%zu dst_bytes=%zu need=%zu\n",
                                rows, cols, inner, rows * cols,
                                buf_map[op.output ? op.output : op.input0].size,
                                need_bytes_dyn);
                    }
                    METAL_LOGSTREAM_TRACE("mlir")
                        << "[METAL MLIR] softmax runtime rows=" << rows << " cols=" << cols << " inner=" << inner << "\n";
                    METAL_LOGSTREAM_TRACE("mlir")
                        << "[METAL MLIR] softmax runtime rows=" << rows << " cols=" << cols << " inner=" << inner << "\n";
#ifdef __OBJC__
                    if (metal_log_trace_enabled() &&
                        src0_buf.storage_mode == static_cast<uint32_t>(MTLStorageModeShared) &&
                        src0 && [src0 contents]) {
                        const float* p = static_cast<const float*>([src0 contents]);
                        size_t n = std::min<size_t>(8, tensor_num_elems(op.input0));
                        auto log = METAL_LOGSTREAM_TRACE("mlir");
                        log << "[METAL MLIR] softmax src first:";
                        for (size_t i = 0; i < n; ++i) log << " " << p[i];
                        log << "\n";
                    }
#endif
                    // input first as generated by MSL
                    [enc setBuffer:src0 offset:0 atIndex:0];
                    [enc setBuffer:dst offset:0 atIndex:1];
                    struct SoftmaxParams {
                        uint32_t rows;
                        uint32_t cols;
                        uint32_t inner;
                    } params;
                    params.rows = static_cast<uint32_t>(rows);
                    params.cols = static_cast<uint32_t>(cols);
                    params.inner = static_cast<uint32_t>(inner);
                    // Buffer capacity checks.
                    const size_t elem_sz = storage_elem_size(op.output ? op.output : op.input0);
                    const uint64_t need_bytes = static_cast<uint64_t>(rows) *
                                                static_cast<uint64_t>(cols) *
                                                static_cast<uint64_t>(elem_sz);
                    const size_t have_in = src0_buf.valid() ? src0_buf.size : buf_len(src0);
                    MetalBuffer dst_mb = find_buf(op.output ? op.output : op.input0);
                    const size_t have_out = dst_mb.valid() ? dst_mb.size : buf_len(dst);
                    OPENVINO_ASSERT(need_bytes <= have_in,
                                    "METAL: Softmax input buffer overflow: need=", need_bytes, " have=", have_in);
                    OPENVINO_ASSERT(need_bytes <= have_out,
                                    "METAL: Softmax output buffer overflow: need=", need_bytes, " have=", have_out,
                                    " out_shape=", dump_shape(op.output));
                    if (dbg_logs) {
                        fprintf(stderr,
                                "[METAL DBG] Softmax rows=%zu cols=%zu inner=%zu bytes=%llu buf in/out=%zu/%zu\n",
                                rows, cols, inner,
                                static_cast<unsigned long long>(need_bytes),
                                have_in, have_out);
                    }
                    [enc setBytes:&params length:sizeof(params) atIndex:2];
                    const NSUInteger total = static_cast<NSUInteger>(rows * cols);
                    const NSUInteger threads_per_tg = 64;
                    MTLSize grid = MTLSizeMake(total, 1, 1);
                    MTLSize tg = MTLSizeMake(std::min<NSUInteger>(threads_per_tg, max_tg), 1, 1);
                    if (dbg_logs) {
                        fprintf(stderr, "[METAL DBG] Softmax grid=%lu tg=%lu\n",
                                static_cast<unsigned long>(total),
                                static_cast<unsigned long>(tg.width));
                    }
                    [enc dispatchThreads:grid threadsPerThreadgroup:tg];
                    break;
                }
                case KernelOpKind::Slice: {
                    if (!src0 || !dst) {
                        METAL_LOG_ERROR("mlir", "[METAL MLIR] Slice missing buffer — abort (CPU fallback disabled)");
                        [enc endEncoding];
                        OPENVINO_THROW("METAL: missing device buffer for Slice and CPU fallback is disabled");
                    }
                    const auto& sd = op.slice;
                    std::vector<int64_t> out_shape = sd.out_shape;
                    if (out_shape.empty() && op.output)
                        out_shape = op.output->shape;
                    size_t rank = out_shape.size();
                    if (rank == 0) {
                        out_shape = {1};
                        rank = 1;
                    }
                    std::vector<uint32_t> out_shape_u(rank, 1);
                    std::vector<uint32_t> in_stride_u(rank, 1);
                    std::vector<int32_t> starts_i(rank, 0);
                    std::vector<uint32_t> steps_u(rank, 1);
                    for (size_t d = 0; d < rank; ++d) {
                        out_shape_u[d] = static_cast<uint32_t>(std::max<int64_t>(1, out_shape[d]));
                        if (d < sd.in_strides.size())
                            in_stride_u[d] = static_cast<uint32_t>(std::max<int64_t>(1, sd.in_strides[d]));
                        if (d < sd.starts.size())
                            starts_i[d] = static_cast<int32_t>(sd.starts[d]);
                        if (d < sd.steps.size())
                            steps_u[d] = static_cast<uint32_t>(std::max<int64_t>(1, sd.steps[d]));
                    }
                    uint64_t total = 1;
                    for (auto d : out_shape_u) total *= static_cast<uint64_t>(d);
                    uint64_t max_off = 0;
                    for (size_t d = 0; d < rank; ++d) {
                        const uint64_t coord = out_shape_u[d] > 0 ? (out_shape_u[d] - 1) : 0;
                        const int64_t step = sd.steps.size() > d ? sd.steps[d] : 1;
                        OPENVINO_ASSERT(step > 0, "METAL: Slice step must be positive");
                        const int64_t start = sd.starts.size() > d ? sd.starts[d] : 0;
                        const int64_t idx = start + static_cast<int64_t>(coord) * step;
                        OPENVINO_ASSERT(idx >= 0, "METAL: Slice index negative");
                        max_off += static_cast<uint64_t>(idx) * static_cast<uint64_t>(in_stride_u[d]);
                    }
                    const size_t elem_sz = storage_elem_size(op.output ? op.output : op.input0);
                    const uint64_t need_in = (max_off + 1) * static_cast<uint64_t>(elem_sz);
                    const uint64_t need_out = total * static_cast<uint64_t>(elem_sz);
                    const size_t have_in = src0_buf.valid() ? src0_buf.size : buf_len(src0);
                    MetalBuffer dst_mb = find_buf(op.output ? op.output : op.input0);
                    const size_t have_out = dst_mb.valid() ? dst_mb.size : buf_len(dst);
                    OPENVINO_ASSERT(need_in <= have_in,
                                    "METAL: Slice input buffer overflow need=", need_in, " have=", have_in);
                    OPENVINO_ASSERT(need_out <= have_out,
                                    "METAL: Slice output buffer overflow need=", need_out, " have=", have_out,
                                    " out_shape=", dump_shape(op.output));
                    [enc setBuffer:src0 offset:0 atIndex:0];
                    [enc setBuffer:dst offset:0 atIndex:1];
                    const uint32_t total_u = static_cast<uint32_t>(total);
                    const uint32_t rank_u = static_cast<uint32_t>(rank);
                    [enc setBytes:&total_u length:sizeof(total_u) atIndex:2];
                    [enc setBytes:&rank_u length:sizeof(rank_u) atIndex:3];
                    [enc setBytes:out_shape_u.data() length:out_shape_u.size() * sizeof(uint32_t) atIndex:4];
                    [enc setBytes:in_stride_u.data() length:in_stride_u.size() * sizeof(uint32_t) atIndex:5];
                    [enc setBytes:starts_i.data() length:starts_i.size() * sizeof(int32_t) atIndex:6];
                    [enc setBytes:steps_u.data() length:steps_u.size() * sizeof(uint32_t) atIndex:7];
                    const NSUInteger elems = static_cast<NSUInteger>(total);
                    const NSUInteger threads_per_tg = 64;
                    MTLSize grid = MTLSizeMake(elems, 1, 1);
                    MTLSize tg = MTLSizeMake(std::min<NSUInteger>(threads_per_tg, max_tg), 1, 1);
                    if (dbg_logs) {
                        fprintf(stderr,
                                "[METAL DBG] Slice rank=%zu total=%llu in_bytes=%llu out_bytes=%llu buf_in/out=%zu/%zu "
                                "grid=%lu tg=%lu\n",
                                rank,
                                static_cast<unsigned long long>(total),
                                static_cast<unsigned long long>(need_in),
                                static_cast<unsigned long long>(need_out),
                                have_in, have_out,
                                static_cast<unsigned long>(grid.width),
                                static_cast<unsigned long>(tg.width));
                    }
                    [enc dispatchThreads:grid threadsPerThreadgroup:tg];
                    break;
                }
                case KernelOpKind::BatchNorm2D: {
                    if (!src0 || !dst) {
                        METAL_LOG_ERROR("mlir", "[METAL MLIR] BatchNorm missing buffer — abort (CPU fallback disabled)");
                        [enc endEncoding];
                        OPENVINO_THROW("METAL: missing device buffer for BatchNorm and CPU fallback is disabled");
                    }
                    const uint32_t N = op.batchnorm.N;
                    const uint32_t C = op.batchnorm.C;
                    const uint32_t H = op.batchnorm.H;
                    const uint32_t W = op.batchnorm.W;
                    OPENVINO_ASSERT(N && C && H && W, "METAL: BatchNorm dims missing");
                    OPENVINO_ASSERT(op.bn_params.size() == static_cast<size_t>(4 * C + 1),
                                    "METAL: BatchNorm params size mismatch");
                    const size_t elem_sz = storage_elem_size(op.output ? op.output : op.input0);
                    const uint64_t need_in = static_cast<uint64_t>(N) * C * H * W * elem_sz;
                    const uint64_t need_out = need_in;
                    const size_t have_in = src0_buf.valid() ? src0_buf.size : buf_len(src0);
                    MetalBuffer dst_mb = find_buf(op.output ? op.output : op.input0);
                    const size_t have_out = dst_mb.valid() ? dst_mb.size : buf_len(dst);
                    OPENVINO_ASSERT(need_in <= have_in,
                                    "METAL: BatchNorm input buffer overflow need=", need_in, " have=", have_in);
                    OPENVINO_ASSERT(need_out <= have_out,
                                    "METAL: BatchNorm output buffer overflow need=", need_out, " have=", have_out,
                                    " out_shape=", dump_shape(op.output));
                    const size_t params_bytes = op.bn_params.size() * sizeof(float);
                    MetalBuffer bn_buf = mgr->allocate(params_bytes,
                                                       ov::element::f32,
                                                       /*persistent=*/false,
                                                       /*storageModePrivate=*/false);
                    mgr->upload(bn_buf, op.bn_params.data(), params_bytes);
                    struct BNParams { uint32_t N, C, H, W; } params{N, C, H, W};
                    [enc setBuffer:src0 offset:0 atIndex:0];
                    [enc setBuffer:to_mtl(bn_buf) offset:0 atIndex:1];
                    [enc setBuffer:dst offset:0 atIndex:2];
                    [enc setBytes:&params length:sizeof(params) atIndex:3];
                    const NSUInteger elems = static_cast<NSUInteger>(N) *
                                             static_cast<NSUInteger>(C) *
                                             static_cast<NSUInteger>(H) *
                                             static_cast<NSUInteger>(W);
                    const NSUInteger threads_per_tg = 64;
                    MTLSize grid = MTLSizeMake(elems, 1, 1);
                    MTLSize tg = MTLSizeMake(std::min<NSUInteger>(threads_per_tg, max_tg), 1, 1);
                    if (dbg_logs) {
                        fprintf(stderr, "[METAL DBG] BatchNorm N=%u C=%u H=%u W=%u grid=%lu tg=%lu\n",
                                N, C, H, W,
                                static_cast<unsigned long>(grid.width),
                                static_cast<unsigned long>(tg.width));
                    }
                    [enc dispatchThreads:grid threadsPerThreadgroup:tg];
                    break;
                }
                case KernelOpKind::Interpolate: {
                    if (!src0 || !dst) {
                        METAL_LOG_ERROR("mlir", "[METAL MLIR] Interpolate missing buffer — abort (CPU fallback disabled)");
                        [enc endEncoding];
                        OPENVINO_THROW("METAL: missing device buffer for Interpolate and CPU fallback is disabled");
                    }
                    struct InterpolateParams {
                        uint32_t N;
                        uint32_t C;
                        uint32_t H_in;
                        uint32_t W_in;
                        uint32_t H_out;
                        uint32_t W_out;
                        float scale_h;
                        float scale_w;
                        uint32_t align_corners;
                    } params{};
                    params.N = op.interpolate.N;
                    params.C = op.interpolate.C;
                    params.H_in = op.interpolate.H_in;
                    params.W_in = op.interpolate.W_in;
                    params.H_out = op.interpolate.H_out;
                    params.W_out = op.interpolate.W_out;
                    params.scale_h = op.interpolate.scale_h;
                    params.scale_w = op.interpolate.scale_w;
                    params.align_corners = op.interpolate.align_corners ? 1u : 0u;
                    OPENVINO_ASSERT(params.N && params.C && params.H_in && params.W_in &&
                                    params.H_out && params.W_out,
                                    "METAL: Interpolate dims missing");
                    const size_t elem_sz = storage_elem_size(op.output ? op.output : op.input0);
                    const uint64_t need_in = static_cast<uint64_t>(params.N) *
                                             params.C * params.H_in * params.W_in * elem_sz;
                    const uint64_t need_out = static_cast<uint64_t>(params.N) *
                                              params.C * params.H_out * params.W_out * elem_sz;
                    const size_t have_in = src0_buf.valid() ? src0_buf.size : buf_len(src0);
                    MetalBuffer dst_mb = find_buf(op.output ? op.output : op.input0);
                    const size_t have_out = dst_mb.valid() ? dst_mb.size : buf_len(dst);
                    OPENVINO_ASSERT(need_in <= have_in,
                                    "METAL: Interpolate input buffer overflow need=", need_in, " have=", have_in);
                    OPENVINO_ASSERT(need_out <= have_out,
                                    "METAL: Interpolate output buffer overflow need=", need_out, " have=", have_out,
                                    " out_shape=", dump_shape(op.output));
                    [enc setBuffer:src0 offset:0 atIndex:0];
                    [enc setBuffer:dst offset:0 atIndex:1];
                    [enc setBytes:&params length:sizeof(params) atIndex:2];
                    const NSUInteger total = static_cast<NSUInteger>(params.N) *
                                             static_cast<NSUInteger>(params.C) *
                                             static_cast<NSUInteger>(params.H_out) *
                                             static_cast<NSUInteger>(params.W_out);
                    const NSUInteger threads_per_tg = 64;
                    MTLSize grid = MTLSizeMake(total, 1, 1);
                    MTLSize tg = MTLSizeMake(std::min<NSUInteger>(threads_per_tg, max_tg), 1, 1);
                    if (dbg_logs) {
                        fprintf(stderr, "[METAL DBG] Interpolate N=%u C=%u H_out=%u W_out=%u grid=%lu tg=%lu\n",
                                params.N, params.C, params.H_out, params.W_out,
                                static_cast<unsigned long>(grid.width),
                                static_cast<unsigned long>(tg.width));
                    }
                    [enc dispatchThreads:grid threadsPerThreadgroup:tg];
                    break;
                }
                case KernelOpKind::Concat: {
                    if (!src0 || !dst) {
                        METAL_LOG_ERROR("mlir", "[METAL MLIR] Concat missing buffer — abort (CPU fallback disabled)");
                        [enc endEncoding];
                        OPENVINO_THROW("METAL: missing device buffer for Concat and CPU fallback is disabled");
                    }
                    const uint64_t outer = op.concat.outer;
                    const uint64_t inner = op.concat.inner;
                    uint64_t axis_len = !op.concat.axis_sizes.empty() ? op.concat.axis_sizes[0] : 0;
                    uint64_t axis_off = !op.concat.axis_offsets.empty() ? op.concat.axis_offsets[0] : 0;
                    uint64_t axis_total = op.concat.axis_total;
                    const int64_t axis = op.concat.axis;
                    if (axis_total == 0 && op.output && axis >= 0 &&
                        static_cast<size_t>(axis) < op.output->shape.size()) {
                        const int64_t d = op.output->shape[static_cast<size_t>(axis)];
                        axis_total = static_cast<uint64_t>(d > 0 ? d : 0);
                    }
                    if (axis_len == 0 && op.input0 && axis >= 0 &&
                        static_cast<size_t>(axis) < op.input0->shape.size()) {
                        const int64_t d = op.input0->shape[static_cast<size_t>(axis)];
                        axis_len = static_cast<uint64_t>(d > 0 ? d : 0);
                    }
                    if (axis_total == 0)
                        axis_total = axis_off + axis_len;
                    OPENVINO_ASSERT(axis_off + axis_len <= axis_total,
                                    "METAL: Concat axis range invalid offset=", axis_off,
                                    " len=", axis_len, " total=", axis_total);
                    const size_t elem_sz = storage_elem_size(op.output ? op.output : op.input0);
                    const uint64_t need_in = outer * axis_len * inner * static_cast<uint64_t>(elem_sz);
                    const uint64_t need_out = outer * axis_total * inner * static_cast<uint64_t>(elem_sz);
                    const size_t have_in = src0_buf.valid() ? src0_buf.size : buf_len(src0);
                    MetalBuffer dst_mb = find_buf(op.output ? op.output : op.input0);
                    const size_t have_out = dst_mb.valid() ? dst_mb.size : buf_len(dst);
                    OPENVINO_ASSERT(need_in <= have_in,
                                    "METAL: Concat input buffer overflow need=", need_in, " have=", have_in);
                    OPENVINO_ASSERT(need_out <= have_out,
                                    "METAL: Concat output buffer overflow need=", need_out, " have=", have_out,
                                    " out_shape=", dump_shape(op.output));
                    struct ConcatParams {
                        uint32_t outer;
                        uint32_t inner;
                        uint32_t axis_offset;
                        uint32_t axis_len;
                        uint32_t axis_total;
                    } params;
                    params.outer = static_cast<uint32_t>(outer);
                    params.inner = static_cast<uint32_t>(inner);
                    params.axis_offset = static_cast<uint32_t>(axis_off);
                    params.axis_len = static_cast<uint32_t>(axis_len);
                    params.axis_total = static_cast<uint32_t>(axis_total);
                    [enc setBuffer:src0 offset:0 atIndex:0];
                    [enc setBuffer:dst offset:0 atIndex:1];
                    [enc setBytes:&params length:sizeof(params) atIndex:2];
                    const NSUInteger total = static_cast<NSUInteger>(outer * axis_len * inner);
                    const NSUInteger threads_per_tg = 64;
                    MTLSize grid = MTLSizeMake(total, 1, 1);
                    MTLSize tg = MTLSizeMake(std::min<NSUInteger>(threads_per_tg, max_tg), 1, 1);
                    if (dbg_logs) {
                        fprintf(stderr,
                                "[METAL DBG] Concat outer=%llu inner=%llu off=%llu len=%llu total=%llu grid=%lu tg=%lu\n",
                                static_cast<unsigned long long>(outer),
                                static_cast<unsigned long long>(inner),
                                static_cast<unsigned long long>(axis_off),
                                static_cast<unsigned long long>(axis_len),
                                static_cast<unsigned long long>(axis_total),
                                static_cast<unsigned long>(grid.width),
                                static_cast<unsigned long>(tg.width));
                    }
                    [enc dispatchThreads:grid threadsPerThreadgroup:tg];
                    break;
                }
                case KernelOpKind::Conv2D: {
                    if (!src0 || !src1 || !dst) {
            METAL_LOG_ERROR("mlir", "[METAL MLIR] Conv2D missing buffer — abort (CPU fallback disabled)");
            [enc endEncoding];
            OPENVINO_THROW("METAL: missing device buffer for Conv2D and CPU fallback is disabled");
                    }
                    if (std::getenv("METAL_MLIR_DEBUG")) {
                        fprintf(stderr, "[METAL MLIR] conv2d src0=%p src1=%p dst=%p N=%u C=%u H=%u W=%u outH=%u outW=%u C_out=%u\n",
                                src0, src1, dst,
                                op.conv2d.N, op.conv2d.C_in, op.conv2d.H, op.conv2d.W,
                                op.conv2d.outH, op.conv2d.outW, op.conv2d.C_out);
                    }
                    struct ConvParams {
                        uint32_t N, C_in, H, W;
                        uint32_t C_out;
                        uint32_t groups;
                        uint32_t C_in_pg;
                        uint32_t C_out_pg;
                        uint32_t kH, kW;
                        uint32_t strideH, strideW;
                        uint32_t dilationH, dilationW;
                        uint32_t padTop, padLeft;
                        uint32_t padBottom, padRight;
                        uint32_t outH, outW;
                        uint32_t has_bias;
                        uint32_t has_bn;
                        uint32_t activation;
                        float alpha;
                        float epsilon;
                        float clamp_min;
                        float clamp_max;
                    } params{};
                    params.N = op.conv2d.N;
                    params.C_in = op.conv2d.C_in;
                    params.H = op.conv2d.H;
                    params.W = op.conv2d.W;
                    params.C_out = op.conv2d.C_out;
                    params.groups = op.conv2d.groups;
                    params.C_in_pg = op.conv2d.C_in_per_group;
                    params.C_out_pg = op.conv2d.C_out_per_group;
                    params.kH = op.conv2d.kernelH;
                    params.kW = op.conv2d.kernelW;
                    params.strideH = op.conv2d.strideH;
                    params.strideW = op.conv2d.strideW;
                    params.dilationH = op.conv2d.dilationH;
                    params.dilationW = op.conv2d.dilationW;
                    params.padTop = op.conv2d.padTop;
                    params.padLeft = op.conv2d.padLeft;
                    params.padBottom = op.conv2d.padBottom;
                    params.padRight = op.conv2d.padRight;
                    params.outH = op.conv2d.outH;
                    params.outW = op.conv2d.outW;
                    params.has_bias = op.conv2d.has_bias ? 1u : 0u;
                    params.has_bn = op.conv2d.has_bn ? 1u : 0u;
                    params.activation = static_cast<uint32_t>(op.conv2d.activation);
                    params.alpha = op.conv2d.alpha;
                    params.epsilon = op.conv2d.epsilon;
                    params.clamp_min = op.conv2d.clamp_min;
                    params.clamp_max = op.conv2d.clamp_max;
                    // Buffer capacity guards to catch overruns before dispatch.
                    const size_t in_elem = storage_elem_size(op.input0);
                    const size_t w_elem = storage_elem_size(op.input1);
                    const size_t out_elem = storage_elem_size(op.output ? op.output : op.input0);
                    const uint64_t need_in = static_cast<uint64_t>(params.N) *
                                             params.C_in * params.H * params.W * static_cast<uint64_t>(in_elem);
                    const uint64_t need_w = static_cast<uint64_t>(params.C_out) *
                                            params.C_in_pg * params.kH * params.kW * static_cast<uint64_t>(w_elem);
                    const uint64_t need_out = static_cast<uint64_t>(params.N) *
                                              params.C_out * params.outH * params.outW * static_cast<uint64_t>(out_elem);
                    const size_t have_in = src0_buf.valid() ? src0_buf.size : buf_len(src0);
                    const MetalBuffer w_mb = find_buf(op.input1);
                    const size_t have_w = w_mb.valid() ? w_mb.size : buf_len(src1);
                    MetalBuffer dst_mb = find_buf(op.output ? op.output : op.input0);
                    const size_t have_out = dst_mb.valid() ? dst_mb.size : buf_len(dst);
                    OPENVINO_ASSERT(need_in <= have_in,
                                    "METAL: Conv2D input buffer overflow: need=", need_in,
                                    " have=", have_in);
                    OPENVINO_ASSERT(need_w <= have_w,
                                    "METAL: Conv2D weights buffer overflow: need=", need_w,
                                    " have=", have_w);
                    OPENVINO_ASSERT(need_out <= have_out,
                                    "METAL: Conv2D output buffer overflow: need=", need_out,
                                    " have=", have_out,
                                    " out_shape=", dump_shape(op.output));
                    if (dbg_logs) {
                        fprintf(stderr,
                                "[METAL DBG] Conv2D NCHW=(%u,%u,%u,%u) k=%u x %u stride=%u x %u pad=%u,%u,%u,%u "
                                "out=%u x %u C_out=%u bytes in/w/out=%llu/%llu/%llu buf in/w/out=%zu/%zu/%zu\n",
                                params.N, params.C_in, params.H, params.W,
                                params.kH, params.kW, params.strideH, params.strideW,
                                params.padTop, params.padLeft, params.padBottom, params.padRight,
                                params.outH, params.outW, params.C_out,
                                static_cast<unsigned long long>(need_in),
                                static_cast<unsigned long long>(need_w),
                                static_cast<unsigned long long>(need_out),
                                have_in, have_w, have_out);
                        const std::string in_ty = op.input0 ? op.input0->dtype.ov_type.get_type_name() : "<null>";
                        const std::string w_ty = op.input1 ? op.input1->dtype.ov_type.get_type_name() : "<null>";
                        const std::string out_ty = op.output ? op.output->dtype.ov_type.get_type_name() : "<null>";
                        fprintf(stderr, "[METAL DBG] Conv2D dtypes in=%s w=%s out=%s elem_sz in/w/out=%zu/%zu/%zu\n",
                                in_ty.c_str(), w_ty.c_str(), out_ty.c_str(), in_elem, w_elem, out_elem);
                    }
                    auto make_zero = [&](size_t count) -> id<MTLBuffer> {
                        const size_t elems = std::max<size_t>(1, count);
                        MetalBuffer b = mgr->allocate(elems * sizeof(float),
                                                      ov::element::f32,
                                                      /*persistent=*/false,
                                                      /*storageModePrivate=*/false);
                        OPENVINO_ASSERT(b.valid(), "METAL: Conv2D zero buffer allocation failed");
                        auto* ptr = [to_mtl(b) contents];
                        OPENVINO_ASSERT(ptr, "METAL: Conv2D zero buffer has no CPU mapping");
                        std::memset(ptr, 0, elems * sizeof(float));
                        return to_mtl(b);
                    };
                    id<MTLBuffer> bias = make_zero(params.C_out);
                    id<MTLBuffer> gamma = make_zero(params.C_out);
                    id<MTLBuffer> beta = make_zero(params.C_out);
                    id<MTLBuffer> mean = make_zero(params.C_out);
                    id<MTLBuffer> var = make_zero(params.C_out);

                    [enc setBuffer:src0 offset:0 atIndex:0];
                    [enc setBuffer:src1 offset:0 atIndex:1];
                    [enc setBuffer:bias offset:0 atIndex:2];
                    [enc setBuffer:gamma offset:0 atIndex:3];
                    [enc setBuffer:beta offset:0 atIndex:4];
                    [enc setBuffer:mean offset:0 atIndex:5];
                    [enc setBuffer:var offset:0 atIndex:6];
                    [enc setBuffer:dst offset:0 atIndex:7];
                    [enc setBytes:&params length:sizeof(params) atIndex:8];
                    const NSUInteger total = static_cast<NSUInteger>(params.N) *
                                             static_cast<NSUInteger>(params.outH) *
                                             static_cast<NSUInteger>(params.outW) *
                                             static_cast<NSUInteger>(params.C_out);
                    const NSUInteger threads_per_tg = 64;
                    MTLSize grid = MTLSizeMake(total, 1, 1);
                    MTLSize tg = MTLSizeMake(std::min<NSUInteger>(threads_per_tg, max_tg), 1, 1);
                    if (dbg_logs) {
                        fprintf(stderr, "[METAL DBG] Conv2D grid=%lu tg=%lu\n",
                                static_cast<unsigned long>(total),
                                static_cast<unsigned long>(tg.width));
                    }
                    [enc dispatchThreads:grid threadsPerThreadgroup:tg];
                    if (std::getenv("METAL_MLIR_DEBUG")) {
                        fprintf(stderr, "[METAL MLIR] conv2d dispatched grid=%lu\n", static_cast<unsigned long>(total));
                    }
                    break;
                }
                case KernelOpKind::MaxPool2D:
                case KernelOpKind::AvgPool2D: {
                    if (!src0 || !dst) {
                        METAL_LOG_ERROR("mlir", "[METAL MLIR] Pool missing buffer — abort (CPU fallback disabled)");
                        [enc endEncoding];
                        OPENVINO_THROW("METAL: missing device buffer for Pool and CPU fallback is disabled");
                    }
                    struct PoolParams {
                        uint32_t N, C, H, W;
                        uint32_t kH, kW;
                        uint32_t strideH, strideW;
                        uint32_t dilationH, dilationW;
                        uint32_t padTop, padLeft, padBottom, padRight;
                        uint32_t outH, outW;
                        uint32_t is_avg;
                        uint32_t exclude_pad;
                    } params{};
                    params.N = op.pool.N;
                    params.C = op.pool.C;
                    params.H = op.pool.H;
                    params.W = op.pool.W;
                    params.kH = op.pool.kernelH;
                    params.kW = op.pool.kernelW;
                    params.strideH = op.pool.strideH;
                    params.strideW = op.pool.strideW;
                    params.dilationH = 1;
                    params.dilationW = 1;
                    params.padTop = op.pool.padTop;
                    params.padLeft = op.pool.padLeft;
                    params.padBottom = op.pool.padBottom;
                    params.padRight = op.pool.padRight;
                    params.outH = op.pool.outH;
                    params.outW = op.pool.outW;
                    params.is_avg = op.kind == KernelOpKind::AvgPool2D ? 1u : 0u;
                    params.exclude_pad = op.kind == KernelOpKind::AvgPool2D ? (op.pool.exclude_pad ? 1u : 0u) : 0u;
                    // Buffer capacity checks.
                    const size_t in_elem = storage_elem_size(op.input0);
                    const size_t out_elem = storage_elem_size(op.output ? op.output : op.input0);
                    const uint64_t need_in = static_cast<uint64_t>(params.N) *
                                             params.C * params.H * params.W * static_cast<uint64_t>(in_elem);
                    const uint64_t need_out = static_cast<uint64_t>(params.N) *
                                              params.C * params.outH * params.outW * static_cast<uint64_t>(out_elem);
                    MetalBuffer dst_mb = find_buf(op.output ? op.output : op.input0);
                    const size_t have_in = src0_buf.valid() ? src0_buf.size : buf_len(src0);
                    const size_t have_out = dst_mb.valid() ? dst_mb.size : buf_len(dst);
                    OPENVINO_ASSERT(need_in <= have_in,
                                    "METAL: Pool input buffer overflow: need=", need_in, " have=", have_in);
                    OPENVINO_ASSERT(need_out <= have_out,
                                    "METAL: Pool output buffer overflow: need=", need_out, " have=", have_out,
                                    " out_shape=", dump_shape(op.output));
                    if (dbg_logs) {
                        fprintf(stderr,
                                "[METAL DBG] Pool kind=%s N=%u C=%u HxW=%u x %u out=%u x %u k=%u x %u "
                                "stride=%u x %u in/out bytes=%llu/%llu buf=%zu/%zu\n",
                                (op.kind == KernelOpKind::AvgPool2D ? "Avg" : "Max"),
                                params.N, params.C, params.H, params.W,
                                params.outH, params.outW,
                                params.kH, params.kW,
                                params.strideH, params.strideW,
                                static_cast<unsigned long long>(need_in),
                                static_cast<unsigned long long>(need_out),
                                have_in, have_out);
                    }
                    [enc setBuffer:src0 offset:0 atIndex:0];
                    [enc setBuffer:dst offset:0 atIndex:1];
                    [enc setBytes:&params length:sizeof(params) atIndex:2];
                    const NSUInteger total = static_cast<NSUInteger>(op.pool.N) *
                                             static_cast<NSUInteger>(op.pool.C);
                    const NSUInteger threads_per_tg = 64;
                    MTLSize grid = MTLSizeMake(total, 1, 1);
                    MTLSize tg = MTLSizeMake(std::min<NSUInteger>(threads_per_tg, max_tg), 1, 1);
                    if (dbg_logs) {
                        fprintf(stderr, "[METAL DBG] Pool grid=%lu tg=%lu\n",
                                static_cast<unsigned long>(total),
                                static_cast<unsigned long>(tg.width));
                    }
                    [enc dispatchThreads:grid threadsPerThreadgroup:tg];
                    break;
                }
            }
            [enc endEncoding];
#ifdef __OBJC__
            if (enc) [enc release];
#endif
        };

        id<MTLBuffer> final_buf = nil;
        MetalBuffer final_dev;
        for (size_t i = 0; i < seg.op_count; ++i) {
            const auto& op = seg_op(i);
            if (seg.first_op_index + i >= m_pipelines.size()) {
                run_fallback("run_segment: pipelines missing for op");
                return;
            }
            if (std::getenv("METAL_DEBUG_DUMP_OUTPUT")) {
                auto dump_shape = [](const KernelTensor* t) {
                    if (!t) return std::string("<null>");
                    std::string s = "[";
                    for (size_t k = 0; k < t->shape.size(); ++k) {
                        if (k) s += ",";
                        s += std::to_string(t->shape[k]);
                    }
                    s += "]";
                    return s;
                };
                auto tensor_name = [](const KernelTensor* t) -> std::string {
                    if (!t || !t->source_node) return "<null>";
                    return static_cast<const ov::Node*>(t->source_node)->get_friendly_name();
                };
                fprintf(stderr,
                        "[METAL][dbg] op%zu kind=%d out=%s ptr=%p shape=%s in0=%s/%p shape0=%s in1=%s/%p shape1=%s\n",
                        i,
                        static_cast<int>(op.kind),
                        tensor_name(op.output).c_str(),
                        static_cast<const void*>(op.output),
                        dump_shape(op.output).c_str(),
                        tensor_name(op.input0).c_str(),
                        static_cast<const void*>(op.input0),
                        dump_shape(op.input0).c_str(),
                        tensor_name(op.input1).c_str(),
                        static_cast<const void*>(op.input1),
                        dump_shape(op.input1).c_str());
            }
            MetalBuffer src0_mb;
            if (op.input0) {
                auto it0 = buf_map.find(op.input0);
                if (it0 != buf_map.end()) src0_mb = it0->second;
                // If input0 is a constant and not yet materialized, create it now.
                if (!src0_mb.valid() && op.input0->from_constant) {
                    src0_mb = make_const_buffer(op.input0);
                    if (src0_mb.valid())
                        buf_map[op.input0] = src0_mb;
                }
            }
            const size_t src0_bytes = src0_mb.size;
            id<MTLBuffer> src0 = to_mtl(src0_mb);
            if (!src0 && inputs.size() > 0 && i == 0 && op.input0) {
                MetalBuffer b = make_buffer_from_tensor(inputs[0]);
                buf_map[op.input0] = b;
                src0 = to_mtl(b);
            }
            if (op.kind == KernelOpKind::Split) {
                const auto& desc = op.split;
                if (!src0) {
                    const char* in0_name = op.input0 ? op.input0->name.c_str() : "(null)";
                    fprintf(stderr, "[METAL][ERR] Split missing src0 idx=%zu name=%s\n", i, in0_name);
                    run_fallback("run_segment: Split missing src0 buffer");
                    return;
                }
                // Compute copy geometry
                const size_t axis = static_cast<size_t>(desc.axis >= 0 ? desc.axis : 0);
                size_t outer = 1;
                for (size_t k = 0; k < axis && k < desc.input_shape.size(); ++k)
                    outer *= static_cast<size_t>(desc.input_shape[k]);
                const size_t axis_len = (axis < desc.input_shape.size())
                                            ? static_cast<size_t>(desc.input_shape[axis])
                                            : 1;
                const size_t inner = desc.inner > 0 ? static_cast<size_t>(desc.inner) : 1;
                const size_t elem_sz = src0_mb.type.size() ? src0_mb.type.size() : element_size(op.input0);
                const size_t src_need = outer * axis_len * inner * elem_sz;
                const size_t src_have = src0_mb.valid() ? src0_mb.size : buf_len(src0);
                OPENVINO_ASSERT(src_need <= src_have,
                                "METAL: Split input buffer overflow need=", src_need, " have=", src_have);
                if (dbg_logs) {
                    fprintf(stderr,
                            "[METAL DBG] Split axis=%zu outer=%zu inner=%zu axis_len=%zu bytes=%zu buf_in=%zu\n",
                            axis, outer, inner, axis_len, src_need, src_have);
                }

                // Allocate outputs (one per split slice) via buffer manager with dynamic growth.
                for (size_t out_idx = 0; out_idx < op.outputs.size(); ++out_idx) {
                    auto* out_t = op.outputs[out_idx];
                    size_t split = (out_idx < desc.split_sizes.size()) ? desc.split_sizes[out_idx] : 1;
                    size_t bytes = outer * split * inner * elem_sz;
                    ov::element::Type et = out_t ? dtype_to_element(out_t->dtype) : src0_mb.type;
                    TempBufferKey key{KernelOpKind::Split,
                                      (seg.first_op_index + i) * 16 + out_idx,
                                      et};
                    auto& handle = m_temp_buffer_handles[key];
                    const bool use_private = false;  // keep outputs host-visible (no CPU copies allowed)
                    MetalBuffer dst_mb =
                        mgr->allocate_dynamic(bytes, et, handle, /*persistent=*/false, use_private);
                    if (out_t)
                        buf_map[out_t] = dst_mb;
                    if (dst_mb.valid()) {
                        OPENVINO_ASSERT(bytes <= dst_mb.size,
                                        "METAL: Split output buffer overflow need=", bytes, " have=", dst_mb.size);
                        if (dbg_logs) {
                            const size_t axis_off =
                                (out_idx < desc.axis_offsets.size()) ? static_cast<size_t>(desc.axis_offsets[out_idx]) : 0;
                            fprintf(stderr,
                                    "[METAL DBG] Split out%zu off=%zu len=%zu bytes=%zu buf_out=%zu\n",
                                    out_idx, axis_off, split, bytes, dst_mb.size);
                        }
                    }
                }

#ifdef __OBJC__
                id<MTLBlitCommandEncoder> blit = [cmd blitCommandEncoder];
                for (size_t out_idx = 0; out_idx < op.outputs.size(); ++out_idx) {
                    auto* out_t = op.outputs[out_idx];
                    auto it_out = buf_map.find(out_t);
                    if (it_out == buf_map.end() || !it_out->second.valid())
                        continue;
                    id<MTLBuffer> dst = to_mtl(it_out->second);
                    const size_t split = (out_idx < desc.split_sizes.size()) ? desc.split_sizes[out_idx] : 1;
                    const size_t axis_off =
                        (out_idx < desc.axis_offsets.size()) ? static_cast<size_t>(desc.axis_offsets[out_idx]) : 0;
                    for (size_t o = 0; o < outer; ++o) {
                        size_t src_off = (o * axis_len * inner + axis_off * inner) * elem_sz;
                        size_t dst_off = (o * split * inner) * elem_sz;
                        size_t bytes = split * inner * elem_sz;
                        OPENVINO_ASSERT(src_off + bytes <= src_have,
                                        "METAL: Split src OOB (offset+bytes > size)");
                        OPENVINO_ASSERT(dst_off + bytes <= it_out->second.size,
                                        "METAL: Split dst OOB (offset+bytes > size)");
                        [blit copyFromBuffer:src0 sourceOffset:src_off toBuffer:dst destinationOffset:dst_off size:bytes];
                    }
                }
                [blit endEncoding];
#endif
                // No compute pipeline needed; continue to next op.
                continue;
            }
            if (op.kind == KernelOpKind::Transpose) {
                if (!src0) {
                    run_fallback("run_segment: Transpose missing src0 buffer");
                    return;
                }
                if (!op.output || op.transpose.perm.empty()) {
                    run_fallback("run_segment: Transpose missing metadata");
                    return;
                }
                std::vector<int64_t> in_shape = op.transpose.in_shape;
                if (in_shape.empty() && op.input0 && !op.input0->shape.empty())
                    in_shape = op.input0->shape;
                std::vector<int64_t> out_shape = op.transpose.out_shape;
                if (out_shape.empty() && op.output && !op.output->shape.empty())
                    out_shape = op.output->shape;
                if (in_shape.empty() || out_shape.empty()) {
                    run_fallback("run_segment: Transpose missing shapes");
                    return;
                }
                ov::element::Type et = dtype_to_element(op.output->dtype);
                if (et == ov::element::dynamic && op.input0)
                    et = dtype_to_element(op.input0->dtype);
                if (et == ov::element::dynamic)
                    et = ov::element::f32;
                const size_t elem_sz = et.size();
                auto shape_elems = [&](const std::vector<int64_t>& shp) -> size_t {
                    size_t elems = 1;
                    for (auto d : shp) {
                        elems *= static_cast<size_t>(d < 0 ? 1 : d);
                    }
                    return elems;
                };
                const size_t total_elems = shape_elems(out_shape);
                const size_t bytes = total_elems * elem_sz;
                TempBufferKey key{KernelOpKind::Transpose, seg.first_op_index + i, et};
                auto& handle = m_temp_buffer_handles[key];
                const bool use_private = false;  // keep outputs host-visible (no CPU copies allowed)
                MetalBuffer dst_mb =
                    mgr->allocate_dynamic(bytes, et, handle, /*persistent=*/false, use_private);
                buf_map[op.output] = dst_mb;
#ifdef __OBJC__
                id<MTLComputePipelineState> tp = seg_pipe(i);
                if (!tp) { run_fallback("run_segment: transpose pipeline missing"); return; }
                id<MTLComputeCommandEncoder> enc = [cmd computeCommandEncoder];
                if (!enc) { run_fallback("run_segment: transpose encoder null"); return; }
                const size_t need_src = shape_elems(in_shape) * elem_sz;
                const size_t need_dst = total_elems * elem_sz;
                OPENVINO_ASSERT(src0_mb.size >= need_src, "METAL: Transpose src buffer too small");
                OPENVINO_ASSERT(dst_mb.size >= need_dst, "METAL: Transpose dst buffer too small");
                std::vector<uint32_t> in_stride(in_shape.size(), 1);
                for (int s = static_cast<int>(in_shape.size()) - 2; s >= 0; --s) {
                    in_stride[static_cast<size_t>(s)] =
                        in_stride[static_cast<size_t>(s + 1)] * static_cast<uint32_t>(in_shape[static_cast<size_t>(s + 1)]);
                }
                std::vector<uint32_t> out_shape_u(out_shape.begin(), out_shape.end());
                std::vector<uint32_t> perm_u;
                perm_u.reserve(op.transpose.perm.size());
                for (auto p : op.transpose.perm)
                    perm_u.push_back(static_cast<uint32_t>(p));
                const uint32_t num_elems = static_cast<uint32_t>(total_elems);
                const uint32_t rank = static_cast<uint32_t>(out_shape_u.size());
                [enc setComputePipelineState:tp];
                [enc setBuffer:src0 offset:0 atIndex:0];
                [enc setBuffer:to_mtl(dst_mb) offset:0 atIndex:1];
                [enc setBytes:&num_elems length:sizeof(num_elems) atIndex:2];
                [enc setBytes:&rank length:sizeof(rank) atIndex:3];
                [enc setBytes:out_shape_u.data() length:out_shape_u.size() * sizeof(uint32_t) atIndex:4];
                [enc setBytes:perm_u.data() length:perm_u.size() * sizeof(uint32_t) atIndex:5];
                [enc setBytes:in_stride.data() length:in_stride.size() * sizeof(uint32_t) atIndex:6];
                MTLSize grid = MTLSizeMake(total_elems, 1, 1);
                const NSUInteger max_tp = tp.maxTotalThreadsPerThreadgroup;
                MTLSize tg = MTLSizeMake(std::min<NSUInteger>(64, max_tp), 1, 1);
                if (dbg_logs) {
                    fprintf(stderr,
                            "[METAL DBG] Transpose elems=%zu bytes=%zu buf_in/out=%zu/%zu grid=%lu tg=%lu\n",
                            total_elems, need_dst,
                            src0_mb.size, dst_mb.size,
                            static_cast<unsigned long>(grid.width),
                            static_cast<unsigned long>(tg.width));
                }
                [enc dispatchThreads:grid threadsPerThreadgroup:tg];
                [enc endEncoding];
#endif
                final_buf = to_mtl(dst_mb);
                final_dev = dst_mb;
                final_shape.assign(out_shape.begin(), out_shape.end());
                final_logical = et;
                continue;
            }
            if (op.kind == KernelOpKind::Reshape) {
                if (!src0) {
                    run_fallback("run_segment: Reshape missing src0 buffer");
                    return;
                }
                if (op.output) {
                    buf_map[op.output] = src0_mb;
                    final_dev = src0_mb;
                    final_buf = src0;
                    if (!op.reshape.out_shape.empty())
                        final_shape.assign(op.reshape.out_shape.begin(), op.reshape.out_shape.end());
                }
                continue;
            }
            if (!src0) {
                size_t host_in = inputs.size();
                size_t mapped_in = in_vec ? in_vec->size() : 0;
                const char* in0_name = op.input0 ? op.input0->name.c_str() : "(null)";
                const char* op_name = (op.output ? op.output->name.c_str() : "");
                fprintf(stderr,
                        "[METAL][ERR] missing src0 idx=%zu kind=%d op_out=%s in0=%s host_inputs=%zu mapped_inputs=%zu params=%zu\n",
                        i,
                        static_cast<int>(op.kind),
                        op_name,
                        in0_name,
                        host_in,
                        mapped_in,
                        param_tensors.size());
                run_fallback("run_segment: missing src0 buffer");
                return;
            }

            MetalBuffer src1_mb;
            if (op.input1) {
                auto it1 = buf_map.find(op.input1);
                if (it1 != buf_map.end()) src1_mb = it1->second;
            }
            id<MTLBuffer> src1 = to_mtl(src1_mb);
            if (std::getenv("METAL_DEBUG_DUMP_OUTPUT") && src1_mb.valid()) {
                fprintf(stderr, "[METAL][dbg] src1_mb size=%zu storage=%u\n",
                        src1_mb.size, src1_mb.storage_mode);
            }
            if ((op.kind == KernelOpKind::MatMul || op.kind == KernelOpKind::ElementwiseAdd ||
                 op.kind == KernelOpKind::ElementwiseSub || op.kind == KernelOpKind::ElementwiseMul ||
                 op.kind == KernelOpKind::ElementwiseDiv || op.kind == KernelOpKind::ElementwisePow ||
                 op.kind == KernelOpKind::ElementwiseMod || op.kind == KernelOpKind::ElementwiseFloorMod) &&
                !src1 && inputs.size() > 1) {
                MetalBuffer b1 = make_buffer_from_tensor(inputs[1]);
                if (op.input1) buf_map[op.input1] = b1;
                src1 = to_mtl(b1);
            }
            if (!src1) {
                MetalBuffer c = load_const_input(op);
                src1 = to_mtl(c);
                if (op.input1 && c.valid()) buf_map[op.input1] = c;
            }
            if ((op.kind == KernelOpKind::MatMul || op.kind == KernelOpKind::ElementwiseAdd ||
                 op.kind == KernelOpKind::ElementwiseSub || op.kind == KernelOpKind::ElementwiseMul ||
                 op.kind == KernelOpKind::ElementwiseDiv || op.kind == KernelOpKind::ElementwisePow ||
                 op.kind == KernelOpKind::ElementwiseMod || op.kind == KernelOpKind::ElementwiseFloorMod) &&
                !src1) { run_fallback("run_segment: missing src1 buffer"); return; }

            id<MTLBuffer> dst = alloc_out_buffer(op, seg.first_op_index + i, src0_bytes);
            if (!dst) { run_fallback("run_segment: alloc_out_buffer failed"); return; }
            auto pipe = seg_pipe(i);
            OPENVINO_ASSERT(pipe != nil, "run_segment: missing pipeline");
            dispatch_op(op, pipe, src0, src1, dst, src0_bytes, src0_mb, cmd);
            final_buf = dst;
            const KernelTensor* out_tensor = op.output ? op.output : op.input0;
            auto it_out = buf_map.find(out_tensor);
            if (it_out != buf_map.end()) {
                final_dev = it_out->second;
                const KernelTensor* shape_src = op.output ? op.output : op.input0;
                if (shape_src && !shape_src->shape.empty()) {
                    final_shape.assign(shape_src->shape.begin(), shape_src->shape.end());
                }
                if (shape_src && final_logical == ov::element::dynamic &&
                    shape_src->dtype.ov_type != ov::element::dynamic) {
                    final_logical = dtype_to_element(shape_src->dtype);
                }
            }
        }

        [cmd commit];
        [cmd waitUntilCompleted];
        if (cmd.error) {
            METAL_LOG_DEBUG("mlir", std::string("[METAL MLIR] Command buffer error: ") +
                      [[cmd.error localizedDescription] UTF8String]);
        }
#ifdef __OBJC__
        if (cmd)
            [cmd release];
        if (temp_queue && queue)
            [queue release];
#endif
        if (std::getenv("METAL_DEBUG_DUMP_OUTPUT")) {
            for (size_t i = 0; i < seg.op_count; ++i) {
                const auto& op = m_ops[seg.first_op_index + i];
                const KernelTensor* kt = op.output ? op.output : op.input0;
                if (!kt) continue;
                auto it = buf_map.find(kt);
                if (it == buf_map.end() || !it->second.valid())
                    continue;
                fprintf(stderr, "[METAL][dbg] op%zu out buf size=%zu type=%s shape=%s\n",
                        i,
                        it->second.size,
                        kt->dtype.ov_type.get_type_name().c_str(),
                        dump_shape(kt).c_str());
            }
        }

        auto bind_model_outputs = [&]() {
            if (!m_active_tensor_map)
                return;
            const auto& model_ref = m_model ? m_model : m_original_model;
            const auto model_outs = model_ref->outputs();
            for (size_t oi = 0; oi < model_outs.size(); ++oi) {
                auto src_val = model_outs[oi].get_node()->input_value(0);
                const ov::Node* src_node = src_val.get_node();
                const size_t src_port = src_val.get_index();
                bool bound = false;
                for (const auto& kv : buf_map) {
                    const KernelTensor* kt = kv.first;
                    if (!kt)
                        continue;
                    if (kt->source_node != src_node || kt->source_index != src_port)
                        continue;
                    MetalTensor mt;
                    mt.buf = kv.second;
                    bool shape_dynamic = std::any_of(kt->shape.begin(), kt->shape.end(), [](int64_t d) { return d < 0; });
                    if (!kt->shape.empty() && !shape_dynamic) {
                        mt.shape.assign(kt->shape.begin(), kt->shape.end());
                    } else if (model_outs[oi].get_partial_shape().is_static()) {
                        mt.shape = model_outs[oi].get_partial_shape().to_shape();
                    }
                    mt.expected_type = dtype_to_element(kt->dtype);
                    if (mt.expected_type == ov::element::dynamic)
                        mt.expected_type = model_outs[oi].get_element_type();
                    m_active_tensor_map->bind_output_device(oi, mt);
                    bound = true;
                    break;
                }
                if (!bound && final_dev.valid() && oi == 0) {
                    MetalTensor mt{final_dev, final_shape, final_logical};
                    if (mt.shape.empty() && model_outs[0].get_partial_shape().is_static()) {
                        mt.shape = model_outs[0].get_partial_shape().to_shape();
                    }
                    if (mt.expected_type == ov::element::dynamic)
                        mt.expected_type = model_outs[0].get_element_type();
                    m_active_tensor_map->bind_output_device(0, mt);
                }
            }
        };
        bind_model_outputs();

        if (final_buf) {
            if (final_shape.empty()) {
                if (!outputs.empty() && outputs[0])
                    final_shape = outputs[0].get_shape();
                else if (!m_original_model->outputs().empty() &&
                         m_original_model->output(0).get_partial_shape().is_static()) {
                    final_shape = m_original_model->output(0).get_partial_shape().to_shape();
                }
            }
            if (!m_original_model->outputs().empty()) {
                auto model_ps = m_original_model->output(0).get_partial_shape();
                if (model_ps.is_static()) {
                    const ov::Shape model_shape = model_ps.to_shape();
                    final_shape = model_shape;
                }
                final_logical = m_original_model->output(0).get_element_type();
            }
            MetalTensor mt{final_dev, final_shape, final_logical};
            if (m_active_tensor_map) {
                m_active_tensor_map->bind_output_device(0, mt);
                if (!outputs.empty()) {
                    outputs[0] = ov::Tensor{final_logical, final_shape};
                }
            } else {
                OPENVINO_THROW("METAL: host output copy disabled (no active tensor map)");
            }
        } else if (!outputs.empty() && !outputs[0].data()) {
            // As a last resort, ensure output tensor is initialized to avoid nullptr _impl.
            auto et = outputs[0].get_element_type();
            if (et == ov::element::dynamic)
                et = m_original_model->output(0).get_element_type();
            auto pshape = m_original_model->output(0).get_partial_shape();
            ov::Shape osh = pshape.is_static() ? pshape.to_shape() : ov::Shape{1};
            outputs[0] = ov::Tensor{et, osh};
        }

        return;
    }
