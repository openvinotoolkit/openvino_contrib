void MlirBackend::Impl::build_flat_ir(const std::shared_ptr<const ov::Model>& model) {
        m_flat_ops.clear();
        m_flat_tensors.clear();
        m_flat_segments.clear();
        // Reserve to keep KernelTensor pointers stable while push_back is used.
        m_flat_tensors.reserve(model->get_ordered_ops().size() * 2);

        struct OutputKey {
            const ov::Node* node = nullptr;
            size_t index = 0;
            bool operator==(const OutputKey& other) const {
                return node == other.node && index == other.index;
            }
        };
        struct OutputKeyHash {
            size_t operator()(const OutputKey& k) const noexcept {
                return std::hash<const void*>()(k.node) ^ (k.index * 0x9e3779b97f4a7c15ULL);
            }
        };
        std::unordered_map<OutputKey, size_t, OutputKeyHash> tensor_index;

        auto make_tensor = [&](const ov::Output<const ov::Node>& out) -> size_t {
            OutputKey key{out.get_node(), out.get_index()};
            auto it = tensor_index.find(key);
            if (it != tensor_index.end()) return it->second;
            const auto pshape = out.get_partial_shape();
            KernelTensor t;
            t.name = out.get_node()->get_friendly_name();
            t.from_parameter = ov::is_type<ov::op::v0::Parameter>(out.get_node());
            t.from_constant = ov::is_type<ov::op::v0::Constant>(out.get_node());
            if (t.from_constant) {
                if (auto c = std::dynamic_pointer_cast<const ov::op::v0::Constant>(out.get_node_shared_ptr())) {
                    // Store as float for simplicity; integers are converted to float.
                    t.const_data = c->cast_vector<float>();
                }
            }
            const auto ov_et = out.get_element_type();
            if (ov_et != ov::element::dynamic) {
                try {
                    t.dtype = resolve_metal_dtype(ov_et);
                } catch (const ov::Exception&) {
                    // Keep default dtype for unsupported types; fallback will handle later.
                }
            }
            t.source_node = out.get_node();
            if (pshape.is_static()) {
                const auto shape = pshape.to_shape();
                t.shape.assign(shape.begin(), shape.end());
            } else if (pshape.rank().is_static()) {
                t.shape.assign(static_cast<size_t>(pshape.rank().get_length()), -1);
            }
            m_flat_tensors.push_back(t);
            size_t idx = m_flat_tensors.size() - 1;
            tensor_index.emplace(key, idx);
            return idx;
        };

        const auto& ordered = model->get_ordered_ops();
        for (const auto& node : ordered) {
            std::ostringstream oss;
            oss << "[METAL MLIR] visiting node " << node->get_friendly_name()
                << " type=" << node->get_type_info().name;
            METAL_LOG_DEBUG("mlir", oss.str());
            // Ensure Parameters have corresponding tensors for input mapping
            if (ov::is_type<ov::op::v0::Parameter>(node.get())) {
                if (node->get_output_size() > 0)
                    make_tensor(node->output(0));
                continue;
            }
            // Skip Result/Constant early
            if (ov::is_type<ov::op::v0::Result>(node.get()) ||
                ov::is_type<ov::op::v0::Constant>(node.get())) {
                continue;
            }

            auto add_input_indices = [&](KernelOp& op) {
                if (node->get_input_size() > 0) {
                    size_t in0 = make_tensor(node->input_value(0));
                    op.input0 = &m_flat_tensors[in0];
                }
                if (node->get_input_size() > 1) {
                    size_t in1 = make_tensor(node->input_value(1));
                    op.input1 = &m_flat_tensors[in1];
                }
            };

            auto set_output = [&](KernelOp& op) {
                if (node->get_output_size() == 0) return;
                size_t out_idx = make_tensor(node->output(0));
                op.output = &m_flat_tensors[out_idx];
            };

            auto propagate_dtype = [&](KernelOp& op) {
                if (op.dtype.ov_type == ov::element::dynamic && node->get_output_size() > 0) {
                    auto et = node->get_output_element_type(0);
                    if (et != ov::element::dynamic) {
                        try {
                            op.dtype = resolve_metal_dtype(et);
                        } catch (const ov::Exception&) {
                            // unsupported dtype will be handled by fallback paths
                        }
                    }
                }
                switch (op.kind) {
                    case KernelOpKind::Split:
                        if (op.split.dtype.ov_type == ov::element::dynamic) {
                            op.split.dtype = op.dtype;
                        }
                        break;
                    case KernelOpKind::Conv2D:
                        if (op.conv2d.dtype.ov_type == ov::element::dynamic) {
                            op.conv2d.dtype = op.dtype;
                        }
                        break;
                    case KernelOpKind::Conv3D:
                        if (op.conv3d.dtype.ov_type == ov::element::dynamic) {
                            op.conv3d.dtype = op.dtype;
                        }
                        break;
                    case KernelOpKind::BatchNorm2D:
                        if (op.batchnorm.dtype.ov_type == ov::element::dynamic) {
                            op.batchnorm.dtype = op.dtype;
                        }
                        break;
                    default:
                        break;
                }
            };

            KernelOp op{};
            bool supported = false;

            // Collapse OpenVINO Mod/FloorMod decomposition (Sign/Abs/Divide/Sub/Mul) into a single eltwise op.
            if (auto mod_pat = match_mod_decomposition(node)) {
                auto [a_out, b_out, is_floor] = *mod_pat;
                if (a_out.get_partial_shape().is_static() && b_out.get_partial_shape().is_static()) {
                    auto a_shape_vec = a_out.get_shape();
                    auto b_shape_vec = b_out.get_shape();
                    auto br = compute_broadcast(a_shape_vec, b_shape_vec);
                    if (!br.success) {
                        METAL_LOG_DEBUG("mlir", "[METAL MLIR] Mod pattern broadcast mismatch → skipping");
                    } else {
                        op.kind = is_floor ? KernelOpKind::ElementwiseFloorMod : KernelOpKind::ElementwiseMod;
                        op.is_broadcast = (a_shape_vec != b_shape_vec);
                        op.out_shape.assign(br.out_shape.begin(), br.out_shape.end());
                        op.stride0 = br.stride0;
                        op.stride1 = br.stride1;
                        auto normalize_stride = [&](std::vector<int64_t>& st) {
                            if (st.size() < op.out_shape.size())
                                st.insert(st.begin(), op.out_shape.size() - st.size(), 0);
                        };
                        normalize_stride(op.stride0);
                        normalize_stride(op.stride1);
                        size_t in0_idx = make_tensor(a_out);
                        size_t in1_idx = make_tensor(b_out);
                        op.input0 = &m_flat_tensors[in0_idx];
                        op.input1 = &m_flat_tensors[in1_idx];
                        set_output(op);
                        if (op.output) op.output->shape = op.out_shape;
                        // If RHS is Constant and requires broadcast, pre-expand to out_shape
                        if (op.input1 && op.input1->from_constant && op.is_broadcast && !op.out_shape.empty()) {
                            const size_t out_elems = ov::shape_size(op.out_shape);
                            if (out_elems > 0 && !op.input1->const_data.empty()) {
                                const size_t rank = op.out_shape.size();
                                // normalize RHS shape to rank
                                std::vector<size_t> rhs_shape_norm(rank, 1);
                                {
                                    auto rhs_shape = op.input1->shape;
                                    size_t off = rank - rhs_shape.size();
                                    for (size_t i = 0; i < rhs_shape.size(); ++i)
                                        rhs_shape_norm[off + i] = static_cast<size_t>(rhs_shape[i]);
                                }
                                std::vector<size_t> rhs_stride(rank, 1);
                                for (int i = static_cast<int>(rank) - 2; i >= 0; --i) {
                                    rhs_stride[i] = rhs_stride[i + 1] * rhs_shape_norm[i + 1];
                                }
                                std::vector<int64_t> out_stride(rank, 1);
                                for (int i = static_cast<int>(rank) - 2; i >= 0; --i) {
                                    out_stride[i] = out_stride[i + 1] * static_cast<int64_t>(op.out_shape[i + 1]);
                                }
                                std::vector<float> broadcasted(out_elems, 0.f);
                                for (size_t idx = 0; idx < out_elems; ++idx) {
                                    size_t tmp = idx;
                                    size_t src_idx = 0;
                                    for (size_t d = 0; d < rank; ++d) {
                                        size_t coord = tmp / static_cast<size_t>(out_stride[d]);
                                        tmp -= coord * static_cast<size_t>(out_stride[d]);
                                        if (rhs_shape_norm[d] == 1) continue;
                                        src_idx += coord * rhs_stride[d];
                                    }
                                    broadcasted[idx] = op.input1->const_data[std::min(src_idx, op.input1->const_data.size() - 1)];
                                }
                                op.input1->const_data.swap(broadcasted);
                                op.input1->shape = op.out_shape;
                                const size_t rank2 = op.out_shape.size();
                                // LHS normalized shape/strides
                                std::vector<size_t> lhs_shape_norm(rank2, 1);
                                if (op.input0) {
                                    auto lhs_shape = op.input0->shape;
                                    size_t off_l = rank2 - lhs_shape.size();
                                    for (size_t i = 0; i < lhs_shape.size(); ++i)
                                        lhs_shape_norm[off_l + i] = static_cast<size_t>(lhs_shape[i]);
                                }
                                std::vector<size_t> lhs_stride(rank2, 1);
                                for (int i = static_cast<int>(rank2) - 2; i >= 0; --i) {
                                    lhs_stride[i] = lhs_stride[i + 1] * lhs_shape_norm[i + 1];
                                }
                                op.stride0.resize(rank2);
                                op.stride1.assign(rank2, 1);
                                for (int i = static_cast<int>(rank2) - 2; i >= 0; --i) {
                                    op.stride1[i] = op.stride1[i + 1] * op.out_shape[i + 1];
                                }
                                for (size_t k = 0; k < rank2; ++k) {
                                    op.stride0[k] = (lhs_shape_norm[k] == 1) ? 0
                                                                              : static_cast<int64_t>(lhs_stride[k]);
                                    if (rhs_shape_norm[k] == 1)
                                        op.stride1[k] = 0;
                                }
                                // Broadcast is still needed if lhs differs from out_shape
                                bool lhs_match = true;
                                for (size_t k = 0; k < rank2; ++k) {
                                    if (lhs_shape_norm[k] != static_cast<size_t>(op.out_shape[k])) {
                                        lhs_match = false;
                                        break;
                                    }
                                }
                                op.is_broadcast = !lhs_match;
                            }
                        }
                        supported = true;
                        std::ostringstream oss;
                        oss << "[METAL MLIR] Collapsed Mod pattern to single op. kind="
                            << (is_floor ? "FloorMod" : "Mod")
                            << " A=" << describe_shape(op.input0)
                            << " B=" << describe_shape(op.input1)
                            << " out=" << describe_shape(op.output);
                        METAL_LOG_DEBUG("mlir", oss.str());
                    }
                }
                if (supported) {
                    propagate_dtype(op);
                    m_flat_ops.push_back(op);
                    continue;
                }
            }

            if (auto split = ov::as_type_ptr<const ov::op::v1::Split>(node)) {
                const auto in_shape = split->get_input_shape(0);
                if (in_shape.empty()) continue;
                auto axis_const = ov::as_type_ptr<const ov::op::v0::Constant>(split->input_value(1).get_node_shared_ptr());
                if (!axis_const) {
                    METAL_LOG_DEBUG("mlir", "[METAL MLIR] Split axis not constant; skipping");
                    continue;
                }
                auto axis_vec = axis_const->cast_vector<int64_t>();
                if (axis_vec.empty()) continue;
                int64_t axis = axis_vec[0];
                int64_t rank = static_cast<int64_t>(in_shape.size());
                int64_t axis_norm = axis >= 0 ? axis : axis + rank;
                if (axis_norm < 0 || axis_norm >= rank) {
                    METAL_LOG_DEBUG("mlir", "[METAL MLIR] Split axis out of range; skipping");
                    continue;
                }
                size_t parts = split->get_num_splits();
                auto dim = in_shape[static_cast<size_t>(axis_norm)];
                if (dim % parts != 0) {
                    METAL_LOG_DEBUG("mlir", "[METAL MLIR] Split dim not divisible by parts; skipping");
                    continue;
                }
                op.kind = KernelOpKind::Split;
                op.split.axis = axis_norm;
                op.split.input_shape.assign(in_shape.begin(), in_shape.end());
                op.split.split_sizes.assign(parts, dim / parts);
                op.split.axis_offsets.resize(parts, 0);
                uint64_t inner = 1;
                for (size_t k = static_cast<size_t>(axis_norm + 1); k < in_shape.size(); ++k) inner *= static_cast<uint64_t>(in_shape[k]);
                op.split.inner = inner;
                for (size_t i = 1; i < parts; ++i) op.split.axis_offsets[i] = op.split.axis_offsets[i - 1] + op.split.split_sizes[i - 1];
                op.split.dtype = resolve_metal_dtype(node->get_output_element_type(0));
                op.split.element_type = static_cast<uint32_t>(static_cast<ov::element::Type_t>(node->get_output_element_type(0)));
                size_t in_idx = make_tensor(node->input_value(0));
                op.input0 = &m_flat_tensors[in_idx];
                for (size_t i = 0; i < parts; ++i) {
                    size_t out_idx = make_tensor(node->output(i));
                    op.outputs.push_back(&m_flat_tensors[out_idx]);
                }
                supported = true;
            } else if (auto vs = ov::as_type_ptr<const ov::op::v1::VariadicSplit>(node)) {
                const auto in_shape = vs->get_input_shape(0);
                if (in_shape.empty()) continue;
                auto axis_const = ov::as_type_ptr<const ov::op::v0::Constant>(vs->input_value(1).get_node_shared_ptr());
                auto len_const = ov::as_type_ptr<const ov::op::v0::Constant>(vs->input_value(2).get_node_shared_ptr());
                if (!axis_const || !len_const) {
                    METAL_LOG_DEBUG("mlir", "[METAL MLIR] VariadicSplit inputs not constant; skipping");
                    continue;
                }
                auto axis_vec = axis_const->cast_vector<int64_t>();
                if (axis_vec.empty()) continue;
                int64_t axis = axis_vec[0];
                int64_t rank = static_cast<int64_t>(in_shape.size());
                int64_t axis_norm = axis >= 0 ? axis : axis + rank;
                if (axis_norm < 0 || axis_norm >= rank) {
                    METAL_LOG_DEBUG("mlir", "[METAL MLIR] VariadicSplit axis out of range; skipping");
                    continue;
                }
                auto lengths = len_const->cast_vector<int64_t>();
                op.kind = KernelOpKind::Split;
                op.split.axis = axis_norm;
                op.split.input_shape.assign(in_shape.begin(), in_shape.end());
                op.split.split_sizes.clear();
                for (auto v : lengths) op.split.split_sizes.push_back(static_cast<size_t>(v));
                op.split.axis_offsets.resize(lengths.size(), 0);
                uint64_t inner = 1;
                for (size_t k = static_cast<size_t>(axis_norm + 1); k < in_shape.size(); ++k) inner *= static_cast<uint64_t>(in_shape[k]);
                op.split.inner = inner;
                for (size_t i = 1; i < lengths.size(); ++i) op.split.axis_offsets[i] = op.split.axis_offsets[i - 1] + op.split.split_sizes[i - 1];
                op.split.dtype = resolve_metal_dtype(node->get_output_element_type(0));
                op.split.element_type = static_cast<uint32_t>(static_cast<ov::element::Type_t>(node->get_output_element_type(0)));
                size_t in_idx = make_tensor(node->input_value(0));
                op.input0 = &m_flat_tensors[in_idx];
                for (size_t i = 0; i < lengths.size(); ++i) {
                    size_t out_idx = make_tensor(node->output(i));
                    op.outputs.push_back(&m_flat_tensors[out_idx]);
                }
                supported = true;
            } else if (ov::as_type_ptr<const ov::op::v0::Convert>(node)) {
                KernelOp cvt{};
                cvt.kind = KernelOpKind::Convert;
                add_input_indices(cvt);
                set_output(cvt);
                if (cvt.output && cvt.input0) {
                    cvt.output->shape = cvt.input0->shape;
                }
                const auto src_et = node->get_input_element_type(0);
                const auto dst_et = node->get_output_element_type(0);
                cvt.convert.src_dtype.ov_type = src_et;
                cvt.convert.dst_dtype.ov_type = dst_et;
                try {
                    cvt.dtype = resolve_metal_dtype(dst_et);
                } catch (const ov::Exception&) {
                    // Let fallback paths handle unsupported dtypes.
                }
                m_flat_ops.push_back(cvt);
                supported = true;
                continue;
            } else if (auto mm = ov::as_type_ptr<const ov::op::v0::MatMul>(node)) {
                // reuse existing helper for shapes
                auto a_shape = mm->get_input_shape(0);
                auto b_shape = mm->get_input_shape(1);
                auto out_shape = mm->get_output_shape(0);
#if METAL_MLIR_DEBUG
                METAL_LOG_DEBUG("mlir", "[METAL MLIR] MatMul build_flat_ir A shape=" + std::to_string(a_shape.size()) +
                          " dims0=" + (a_shape.size() > 0 ? std::to_string(a_shape[0]) : "0") +
                          " dims1=" + (a_shape.size() > 1 ? std::to_string(a_shape[1]) : "0") +
                          " B shape dims=" + (b_shape.size() > 0 ? std::to_string(b_shape[0]) : "0") + "," +
                          (b_shape.size() > 1 ? std::to_string(b_shape[1]) : "0"));
#endif
                auto to3d = [](const ov::Shape& s) {
                    if (s.size() == 2) return std::vector<int64_t>{1, (int64_t)s[0], (int64_t)s[1]};
                    return std::vector<int64_t>{(int64_t)s[0], (int64_t)s[1], (int64_t)s[2]};
                };
                auto a3 = to3d(a_shape);
                auto b3 = to3d(b_shape);
                op.kind = KernelOpKind::MatMul;
                // Trust output shape for M/N; trust A for K to avoid backend-inserted transposes on weights.
                op.M = (out_shape.size() >= 2) ? static_cast<int64_t>(out_shape[out_shape.size() - 2])
                                               : a3[1];
                op.N = (out_shape.size() >= 1) ? static_cast<int64_t>(out_shape.back()) : b3[2];
                op.K = (a_shape.size() >= 1) ? static_cast<int64_t>(a_shape.back()) : a3[2];
                op.a_transpose = mm->get_transpose_a();
                op.b_transpose = mm->get_transpose_b();
                op.b_is_nk_layout = op.b_transpose;
                op.batch_a = a3[0];
                op.batch_b = b3[0];
                op.batch = std::max(op.batch_a, op.batch_b);
                add_input_indices(op);
                set_output(op);
                // Heuristic: some upstream passes fold Transpose(Constant) into a new constant with a "Transpose" name.
                // When that happens the data is already in [N, K] order; mark NK layout so the kernel indexes correctly.
                auto rhs_node = node->input_value(1).get_node_shared_ptr();
                bool rhs_is_folded_transpose = rhs_node &&
                                               (rhs_node->get_friendly_name().find("Transpose") != std::string::npos ||
                                                rhs_node->get_type_name() == std::string("Transpose"));
                if (op.input1 && op.input1->from_constant &&
                    op.input1->name.find("Transpose") != std::string::npos) {
                    rhs_is_folded_transpose = true;
                }
                if (rhs_is_folded_transpose) {
                    op.b_is_nk_layout = true;
                    op.b_transpose = true;
                }
                supported = true;
            } else if (auto sq = ov::as_type_ptr<const ov::op::v0::SquaredDifference>(node)) {
                // Decompose SqDiff into Sub then Mul(sub, sub)
                auto a_shape_vec = node->get_input_shape(0);
                auto b_shape_vec = node->get_input_shape(1);
                auto br = compute_broadcast(a_shape_vec, b_shape_vec);
                if (!br.success) {
                    METAL_LOG_DEBUG("mlir", "[METAL MLIR] SqDiff broadcast mismatch → unsupported");
                    continue;
                }
                // Create temp tensor for sub output
                KernelTensor tmp;
                tmp.name = node->get_friendly_name() + "_sqdiff_tmp";
                tmp.shape.assign(br.out_shape.begin(), br.out_shape.end());
                try {
                    tmp.dtype = resolve_metal_dtype(node->get_output_element_type(0));
                } catch (const ov::Exception&) {
                    // leave dynamic; fallback will handle if unsupported
                }
                m_flat_tensors.push_back(tmp);
                KernelTensor* tmp_ptr = &m_flat_tensors.back();

                // Sub op
                KernelOp sub_op{};
                sub_op.kind = KernelOpKind::ElementwiseSub;
                sub_op.is_broadcast = (a_shape_vec != b_shape_vec);
                sub_op.out_shape.assign(br.out_shape.begin(), br.out_shape.end());
                sub_op.stride0 = br.stride0;
                sub_op.stride1 = br.stride1;
                auto norm = [&](std::vector<int64_t>& st) {
                    if (st.size() < sub_op.out_shape.size())
                        st.insert(st.begin(), sub_op.out_shape.size() - st.size(), 0);
                };
                norm(sub_op.stride0);
                norm(sub_op.stride1);
                size_t in0_idx = make_tensor(node->input_value(0));
                size_t in1_idx = make_tensor(node->input_value(1));
                sub_op.input0 = &m_flat_tensors[in0_idx];
                sub_op.input1 = &m_flat_tensors[in1_idx];
                sub_op.output = tmp_ptr;
                sub_op.output->shape = sub_op.out_shape;
                // If RHS is a Constant and broadcast is needed, pre-expand to full out_shape
                if (sub_op.input1 && sub_op.input1->from_constant && sub_op.is_broadcast && !sub_op.out_shape.empty()) {
                    std::vector<float> broadcasted;
                    const size_t out_elems = ov::shape_size(sub_op.out_shape);
                    if (out_elems > 0 && !sub_op.input1->const_data.empty()) {
                        const size_t rank = sub_op.out_shape.size();
                        // row-major strides for output
                        std::vector<size_t> out_stride(rank, 1);
                        for (int i = static_cast<int>(rank) - 2; i >= 0; --i) {
                            out_stride[i] = out_stride[i + 1] * static_cast<size_t>(sub_op.out_shape[i + 1]);
                        }
                        // normalize RHS shape to out rank
                        std::vector<size_t> rhs_shape(rank, 1);
                        size_t off = rank - sub_op.input1->shape.size();
                        for (size_t i = 0; i < sub_op.input1->shape.size(); ++i)
                            rhs_shape[off + i] = static_cast<size_t>(sub_op.input1->shape[i]);
                        std::vector<size_t> rhs_stride(rank, 1);
                        for (int i = static_cast<int>(rank) - 2; i >= 0; --i) {
                            rhs_stride[i] = rhs_stride[i + 1] * rhs_shape[i + 1];
                        }
                        broadcasted.assign(out_elems, 0.f);
                        for (size_t idx = 0; idx < out_elems; ++idx) {
                            size_t tmp = idx;
                            size_t src_idx = 0;
                            for (size_t d = 0; d < rank; ++d) {
                                size_t coord = tmp / out_stride[d];
                                tmp -= coord * out_stride[d];
                                if (rhs_shape[d] == 1) continue;
                                src_idx += coord * rhs_stride[d];
                            }
                            broadcasted[idx] = sub_op.input1->const_data[std::min(src_idx, sub_op.input1->const_data.size() - 1)];
                        }
                        sub_op.input1->const_data.swap(broadcasted);
                        sub_op.input1->shape = sub_op.out_shape;
                        sub_op.stride1.assign(rank, 1);
                        for (int i = static_cast<int>(rank) - 2; i >= 0; --i) {
                            sub_op.stride1[i] = sub_op.stride1[i + 1] * sub_op.out_shape[i + 1];
                        }
                    }
                } else if (sub_op.is_broadcast && sub_op.input1 && sub_op.input1->shape.size() == 1 && sub_op.out_shape.size() > 1) {
                    // Scalar RHS that wasn’t constant-expanded: make stride1 all zeros except last
                    sub_op.input1->shape.assign(sub_op.out_shape.size(), 1);
                    sub_op.stride1.assign(sub_op.out_shape.size(), 0);
                }

                // Mul op: tmp * tmp -> original output
                KernelOp mul_op{};
                mul_op.kind = KernelOpKind::ElementwiseMul;
                mul_op.is_broadcast = false;
                mul_op.out_shape = sub_op.out_shape;
                mul_op.stride0.assign(sub_op.out_shape.size(), 1);
                mul_op.stride1.assign(sub_op.out_shape.size(), 1);
                for (int i = static_cast<int>(mul_op.out_shape.size()) - 2; i >= 0; --i) {
                    mul_op.stride0[i] = mul_op.stride0[i + 1] * mul_op.out_shape[i + 1];
                    mul_op.stride1[i] = mul_op.stride0[i];
                }
                mul_op.input0 = tmp_ptr;
                mul_op.input1 = tmp_ptr;
                size_t out_idx = make_tensor(node->output(0));
                mul_op.output = &m_flat_tensors[out_idx];
                mul_op.output->shape = mul_op.out_shape;
                if (mul_op.output->shape.empty()) {
                    mul_op.output->shape.assign(mul_op.out_shape.begin(), mul_op.out_shape.end());
                }
                // Propagate dtype to temporary/output tensors explicitly
                try {
                    auto out_dtype = resolve_metal_dtype(node->get_output_element_type(0));
                    tmp_ptr->dtype = out_dtype;
                    mul_op.output->dtype = out_dtype;
                    sub_op.dtype = out_dtype;
                    mul_op.dtype = out_dtype;
                } catch (const ov::Exception&) {
                }

                propagate_dtype(sub_op);
                propagate_dtype(mul_op);
                m_flat_ops.push_back(sub_op);
                m_flat_ops.push_back(mul_op);
                supported = true;
                continue;  // already emitted decomposed ops

            } else if (ov::as_type_ptr<const ov::op::v1::Add>(node) ||
                       ov::as_type_ptr<const ov::op::v1::Subtract>(node) ||
                       ov::as_type_ptr<const ov::op::v1::Multiply>(node) ||
                       ov::as_type_ptr<const ov::op::v1::Power>(node) ||
                       ov::is_type<ov::op::v1::Divide>(node.get()) ||
                       ov::as_type_ptr<const ov::op::v1::Mod>(node) ||
                       ov::as_type_ptr<const ov::op::v1::FloorMod>(node)) {
                auto et0 = node->get_input_element_type(0);
                auto et1 = node->get_input_element_type(1);
                auto supported_num = [](const ov::element::Type& t) {
                    return t.is_real() || t == ov::element::i32 || t == ov::element::i64;
                };
                if (!supported_num(et0) || !supported_num(et1)) {
                    OPENVINO_THROW("METAL eltwise supports only f16/f32/i32/i64/u8 inputs");
                }
                // Unified binary eltwise handling (Add/Sub/Mul/Div/Pow/Mod/FloorMod) with proper NumPy broadcast
                const bool is_mul_node = static_cast<bool>(ov::as_type_ptr<const ov::op::v1::Multiply>(node));
                const bool name_says_div = node->get_friendly_name().find("Divide") != std::string::npos;

                // Detect Multiply + Power(x, -1) pattern that represents true division by parameter.
                auto is_neg_one_const = [](const std::shared_ptr<const ov::Node>& n) {
                    auto c = std::dynamic_pointer_cast<const ov::op::v0::Constant>(n);
                    if (!c) return false;
                    auto vals = c->cast_vector<float>();
                    if (vals.empty()) return false;
                    for (float v : vals) {
                        if (std::fabs(v + 1.0f) > 1e-6f) return false;
                    }
                    return true;
                };

                ov::Output<const ov::Node> lhs_value = node->input_value(0);
                ov::Output<const ov::Node> rhs_value = node->input_value(1);
                auto a_shape_vec = node->get_input_shape(0);
                auto b_shape_vec = node->get_input_shape(1);

                bool div_pow_rhs = false;
                bool div_pow_lhs = false;

                if (is_mul_node) {
                    if (auto pow_rhs = ov::as_type_ptr<const ov::op::v1::Power>(rhs_value.get_node_shared_ptr())) {
                        if (is_neg_one_const(pow_rhs->get_input_node_shared_ptr(1))) {
                            div_pow_rhs = true;
                            rhs_value = pow_rhs->input_value(0);  // true divisor
                            b_shape_vec = pow_rhs->get_input_shape(0);
                        }
                    }
                    if (!div_pow_rhs) {
                        if (auto pow_lhs = ov::as_type_ptr<const ov::op::v1::Power>(lhs_value.get_node_shared_ptr())) {
                            if (is_neg_one_const(pow_lhs->get_input_node_shared_ptr(1))) {
                                div_pow_lhs = true;
                                lhs_value = rhs_value;  // numerator is the other input
                                a_shape_vec = node->get_input_shape(1);
                                rhs_value = pow_lhs->input_value(0);
                                b_shape_vec = pow_lhs->get_input_shape(0);
                            }
                        }
                    }
                }

                if (ov::as_type_ptr<const ov::op::v1::Add>(node)) {
                    op.kind = KernelOpKind::ElementwiseAdd;
                } else if (ov::as_type_ptr<const ov::op::v1::Subtract>(node)) {
                    op.kind = KernelOpKind::ElementwiseSub;
                } else if (ov::is_type<ov::op::v1::Divide>(node.get()) || (is_mul_node && (name_says_div || div_pow_rhs || div_pow_lhs))) {
                    op.kind = KernelOpKind::ElementwiseDiv;
                } else if (is_mul_node) {
                    op.kind = KernelOpKind::ElementwiseMul;
                } else if (ov::as_type_ptr<const ov::op::v1::Mod>(node)) {
                    op.kind = KernelOpKind::ElementwiseMod;
                } else if (ov::as_type_ptr<const ov::op::v1::FloorMod>(node)) {
                    op.kind = KernelOpKind::ElementwiseFloorMod;
                } else {
                    op.kind = KernelOpKind::ElementwisePow;
                }

                auto br = compute_broadcast(a_shape_vec, b_shape_vec);
                if (!br.success) {
                    METAL_LOG_DEBUG("mlir", "[METAL MLIR] Eltwise broadcast mismatch → mark unsupported");
                    continue;
                }
                op.is_broadcast = (a_shape_vec != b_shape_vec);
                op.out_shape.assign(br.out_shape.begin(), br.out_shape.end());
                op.stride0 = br.stride0;
                op.stride1 = br.stride1;
                // Normalize stride arrays to same rank length as out_shape
                auto normalize_stride = [&](std::vector<int64_t>& st) {
                    if (st.size() < op.out_shape.size()) {
                        st.insert(st.begin(), op.out_shape.size() - st.size(), 0);
                    }
                };
                normalize_stride(op.stride0);
                normalize_stride(op.stride1);

                if (div_pow_rhs || div_pow_lhs) {
                    size_t in0 = make_tensor(lhs_value);
                    size_t in1 = make_tensor(rhs_value);
                    op.input0 = &m_flat_tensors[in0];
                    op.input1 = &m_flat_tensors[in1];
                } else {
                    add_input_indices(op);
                }
                set_output(op);
                if (op.output) op.output->shape = op.out_shape;
                std::ostringstream oss;
                oss << "[METAL MLIR] elementwise node=" << node->get_friendly_name()
                    << " kind=" << static_cast<int>(op.kind);
                if (div_pow_rhs || div_pow_lhs) oss << " div_from_pow=-1";
                METAL_LOG_DEBUG("mlir", oss.str());
                // If RHS is Constant and requires broadcast, pre-expand it after inputs are wired
                // This helps all binary eltwise ops (Add/Sub/Mul/Div/Pow) keep simple row-major reads.
                if (op.input1 && op.input1->from_constant && op.is_broadcast && !op.out_shape.empty()) {
                    const size_t out_elems = ov::shape_size(op.out_shape);
                    if (out_elems > 0 && !op.input1->const_data.empty()) {
                        const auto rhs_orig_shape = op.input1->shape;  // save original
                        // Normalize RHS shape to rank of output
                        const size_t rank = op.out_shape.size();
                        ov::Shape rhs_shape_norm(rank, 1);
                        {
                            auto rhs_shape = rhs_orig_shape;
                            size_t off = rank - rhs_shape.size();
                            for (size_t i = 0; i < rhs_shape.size(); ++i)
                                rhs_shape_norm[off + i] = static_cast<size_t>(rhs_shape[i]);
                        }
                        // Strides for normalized RHS
                        std::vector<size_t> rhs_stride(rank, 1);
                        for (int i = static_cast<int>(rank) - 2; i >= 0; --i) {
                            rhs_stride[i] = rhs_stride[i + 1] * rhs_shape_norm[i + 1];
                        }
                        std::vector<float> broadcasted(out_elems, 0.f);
                        std::vector<int64_t> out_stride(rank, 1);
                        for (int i = static_cast<int>(rank) - 2; i >= 0; --i) {
                            out_stride[i] = out_stride[i + 1] * static_cast<int64_t>(op.out_shape[i + 1]);
                        }
                        METAL_LOG_DEBUG("mlir", "[METAL MLIR] rhs_shape_norm dims: " + std::to_string(rhs_shape_norm.size()));
                        for (size_t i = 0; i < rhs_shape_norm.size(); ++i) {
                            METAL_LOG_DEBUG("mlir", "  rhs_shape_norm[" + std::to_string(i) + "]=" + std::to_string(rhs_shape_norm[i]) +
                                      " rhs_stride=" + std::to_string(rhs_stride[i]) +
                                      " out_stride=" + std::to_string(out_stride[i]));
                        }
                        for (size_t idx = 0; idx < out_elems; ++idx) {
                            size_t tmp = idx;
                            size_t src_idx = 0;
                            for (size_t d = 0; d < rank; ++d) {
                                const size_t coord = tmp / static_cast<size_t>(out_stride[d]);
                                tmp -= coord * static_cast<size_t>(out_stride[d]);
                                if (rhs_shape_norm[d] == 1) continue;  // broadcast axis
                                src_idx += coord * rhs_stride[d];
                            }
                            broadcasted[idx] = op.input1->const_data[std::min(src_idx, op.input1->const_data.size() - 1)];
                        }
                        op.input1->const_data.swap(broadcasted);
                        op.input1->shape = op.out_shape;
                        // Recompute strides after expansion
                        auto make_row_major = [&](const std::vector<int64_t>& shape) {
                            std::vector<int64_t> st(shape.size(), 1);
                            for (int i = static_cast<int>(shape.size()) - 2; i >= 0; --i)
                                st[i] = st[i + 1] * shape[i + 1];
                            return st;
                        };
                        std::vector<size_t> lhs_shape_norm(rank, 1);
                        if (op.input0) {
                            auto lhs_shape = op.input0->shape;
                            size_t off_l = rank - lhs_shape.size();
                            for (size_t i = 0; i < lhs_shape.size(); ++i)
                                lhs_shape_norm[off_l + i] = static_cast<size_t>(lhs_shape[i]);
                        }
                        // Decide whether broadcast kernel is still needed: lhs may still require it
                        bool lhs_matches_out = true;
                        for (size_t k = 0; k < rank; ++k) {
                            if (lhs_shape_norm[k] != static_cast<size_t>(op.out_shape[k])) {
                                lhs_matches_out = false;
                                break;
                            }
                        }
                        bool rhs_matches_out = true;
                        for (size_t k = 0; k < rank; ++k) {
                            if (rhs_shape_norm[k] != static_cast<size_t>(op.out_shape[k])) {
                                rhs_matches_out = false;
                                break;
                            }
                        }
                        op.is_broadcast = !(lhs_matches_out && rhs_matches_out);
                        // Use row-major strides derived from original shapes; zero stride on broadcast axes
                        // lhs strides use original layout, rhs uses expanded (row-major out_shape)
                        std::vector<size_t> lhs_stride(rank, 1);
                        for (int i = static_cast<int>(rank) - 2; i >= 0; --i) {
                            lhs_stride[i] = lhs_stride[i + 1] * lhs_shape_norm[i + 1];
                        }
                        op.stride0.resize(rank);
                        op.stride1.assign(out_stride.begin(), out_stride.end());
                        for (size_t k = 0; k < rank; ++k) {
                            op.stride0[k] = (lhs_shape_norm[k] == 1) ? 0
                                                                     : static_cast<int64_t>(lhs_stride[k]);
                            if (rhs_shape_norm[k] == 1) op.stride1[k] = 0;
                        }
                        // Ensure stride0/1 are padded to rank
                        auto pad_stride = [&](std::vector<int64_t>& st) {
                            if (st.size() < rank) st.insert(st.begin(), rank - st.size(), 0);
                        };
                        pad_stride(op.stride0);
                        pad_stride(op.stride1);
                        std::ostringstream oss;
                        oss << "[METAL MLIR] rhs after expand first: ";
                        for (size_t i = 0; i < std::min<size_t>(8, op.input1->const_data.size()); ++i) {
                            oss << op.input1->const_data[i] << " ";
                        }
                        METAL_LOG_DEBUG("mlir", oss.str());
                    }
                }
                // Some front-end passes turn Divide by constant into Multiply by reciprocal.
                // If we see Div with Constant and the node type is Multiply, recover the original divisor
                // to improve f16 accuracy (compute in half then cast).
                if (op.kind == KernelOpKind::ElementwiseDiv &&
                    op.input1 && op.input1->from_constant &&
                    node->get_type_info().name == std::string("Multiply")) {
                    bool invert_ok = true;
                    for (float v : op.input1->const_data) {
                        if (v == 0.f) { invert_ok = false; break; }
                    }
                    if (invert_ok) {
                        for (auto& v : op.input1->const_data) {
                            v = 1.0f / v;
                        }
                        METAL_LOGSTREAM_TRACE("mlir") << "[METAL MLIR] Detected Div decomposed to Multiply; restored divisor, first="
                                  << op.input1->const_data[0] << "\n";
                    }
                }
                supported = true;
            } else if (auto slice = ov::as_type_ptr<const ov::op::v8::Slice>(node)) {
                op.kind = KernelOpKind::Slice;
                add_input_indices(op);
                set_output(op);
                auto in_shape = slice->get_input_shape(0);
                auto out_shape = slice->get_output_shape(0);
                op.slice.in_shape.assign(in_shape.begin(), in_shape.end());
                op.slice.out_shape.assign(out_shape.begin(), out_shape.end());

                auto get_const_vec = [](const std::shared_ptr<const ov::Node>& n) -> std::vector<int64_t> {
                    if (auto c = std::dynamic_pointer_cast<const ov::op::v0::Constant>(n)) {
                        return c->cast_vector<int64_t>();
                    }
                    return {};
                };
                auto starts_raw = get_const_vec(slice->get_input_node_shared_ptr(1));
                auto stops_raw  = get_const_vec(slice->get_input_node_shared_ptr(2));
                auto steps_raw  = get_const_vec(slice->get_input_node_shared_ptr(3));
                std::vector<int64_t> axes;
                if (slice->get_input_size() >= 5) {
                    axes = get_const_vec(slice->get_input_node_shared_ptr(4));
                }
                if (axes.empty()) {
                    axes.resize(starts_raw.size());
                    std::iota(axes.begin(), axes.end(), 0);
                }
                if (steps_raw.empty())
                    steps_raw.assign(starts_raw.size(), 1);
                op.slice.starts.assign(in_shape.size(), 0);
                op.slice.steps.assign(in_shape.size(), 1);
                op.slice.axes = axes;
                for (size_t i = 0; i < axes.size(); ++i) {
                    size_t ax = static_cast<size_t>(axes[i]);
                    if (ax >= in_shape.size() || i >= starts_raw.size())
                        continue;
                    int64_t dim = static_cast<int64_t>(in_shape[ax]);
                    int64_t start = starts_raw[i];
                    int64_t step  = (i < steps_raw.size() ? steps_raw[i] : 1);
                    if (step == 0) step = 1;
                    if (start < 0) start += dim;
                    start = std::max<int64_t>(0, std::min<int64_t>(start, dim - 1));
                    op.slice.starts[ax] = start;
                    op.slice.steps[ax] = step;
                }
                op.slice.in_strides.resize(in_shape.size(), 1);
                for (int i = static_cast<int>(in_shape.size()) - 2; i >= 0; --i) {
                    op.slice.in_strides[i] = op.slice.in_strides[i + 1] * static_cast<int64_t>(in_shape[i + 1]);
                }
                supported = true;
            } else if (ov::as_type_ptr<const ov::op::v0::Relu>(node) ||
                       ov::as_type_ptr<const ov::op::v0::Sigmoid>(node) ||
                       ov::as_type_ptr<const ov::op::v0::Tanh>(node) ||
                       ov::as_type_ptr<const ov::op::v0::Elu>(node) ||
                       ov::as_type_ptr<const ov::op::v0::PRelu>(node) ||
                       ov::as_type_ptr<const ov::op::v0::Gelu>(node) ||
                       ov::as_type_ptr<const ov::op::v4::Swish>(node) ||
                       node->get_type_info().name == std::string("Sign") ||
                       node->get_type_info().name == std::string("Abs")) {
                op.kind = KernelOpKind::Unary;
                if (ov::as_type_ptr<const ov::op::v0::Relu>(node)) op.activation = ActivationKind::Relu;
                else if (ov::as_type_ptr<const ov::op::v0::Sigmoid>(node)) op.activation = ActivationKind::Sigmoid;
                else if (ov::as_type_ptr<const ov::op::v0::Tanh>(node)) op.activation = ActivationKind::Tanh;
                else if (auto e = ov::as_type_ptr<const ov::op::v0::Elu>(node)) {
                    op.activation = ActivationKind::Elu;
                    op.alpha = static_cast<float>(e->get_alpha());
                } else if (auto p = ov::as_type_ptr<const ov::op::v0::PRelu>(node)) {
                    op.activation = ActivationKind::Prelu;
                    // store slope in alpha for now when it is scalar constant
                    if (auto slope_c = std::dynamic_pointer_cast<const ov::op::v0::Constant>(p->get_input_node_shared_ptr(1))) {
                        auto vals = slope_c->cast_vector<float>();
                        if (!vals.empty()) op.alpha = vals[0];
                    }
                } else if (ov::as_type_ptr<const ov::op::v0::Gelu>(node)) {
                    op.activation = ActivationKind::Gelu;
                } else if (ov::as_type_ptr<const ov::op::v4::Swish>(node)) {
                    op.activation = ActivationKind::Swish;
                } else if (ov::as_type_ptr<const ov::op::v0::Abs>(node)) {
                    op.activation = ActivationKind::Abs;
                } else if (ov::as_type_ptr<const ov::op::v0::Sign>(node)) {
                    op.activation = ActivationKind::Sign;
                } else if (node->get_type_info().name == std::string("Sign")) {
                    op.activation = ActivationKind::Sign;
                } else if (node->get_type_info().name == std::string("Abs")) {
                    op.activation = ActivationKind::Abs;
                }
                add_input_indices(op);
                set_output(op);
                supported = true;
            } else if (ov::is_type<ov::op::v1::Softmax>(node.get()) || ov::is_type<ov::op::v8::Softmax>(node.get())) {
                op.kind = KernelOpKind::Softmax;
                add_input_indices(op);
                set_output(op);
                int64_t axis = -1;
                if (auto sm1 = ov::as_type_ptr<const ov::op::v1::Softmax>(node))
                    axis = sm1->get_axis();
                else if (auto sm8 = ov::as_type_ptr<const ov::op::v8::Softmax>(node))
                    axis = sm8->get_axis();
                op.softmax_axis = axis;

                // Guard unsupported axes for rank 4/5 early to force CPU fallback.
                const auto in_pshape = node->get_input_partial_shape(0);
                if (!is_softmax_shape_supported(in_pshape, axis)) {
                    METAL_LOG_DEBUG("mlir", "[METAL MLIR] Softmax unsupported axis/shape in flat IR build, using host softmax");
                    m_softmax_dynamic_only = false;  // force CPU fallback for unsupported static axes
                    if (!m_allow_partial_offload) {
                        OPENVINO_THROW("METAL: unsupported Softmax axis/shape in pure device mode");
                    }
                    m_force_fallback = true;
                    ensure_fallback(m_original_model);
                    return;
                }

                if (node->get_output_size() > 0) {
                    const auto pshape = node->get_output_partial_shape(0);
                    if (pshape.rank().is_static()) {
                        const auto rank = pshape.rank().get_length();
                        if (axis < 0) axis += rank;
                        if (pshape.is_static()) {
                            auto shape = pshape.to_shape();
                            int64_t cols = static_cast<int64_t>(shape[static_cast<size_t>(axis)]);
                            int64_t outer = 1;
                            for (int64_t i = 0; i < axis; ++i) outer *= static_cast<int64_t>(shape[static_cast<size_t>(i)]);
                            int64_t inner = 1;
                            for (size_t i = static_cast<size_t>(axis) + 1; i < shape.size(); ++i) inner *= static_cast<int64_t>(shape[i]);
                            int64_t rows = outer * inner;
                            op.rows = rows;
                            op.cols = cols;
                            op.inner = inner;
                        } else {
                            op.rows = 0;
                            op.cols = 0;
                            op.inner = 1;
                        }
                    }
                }
                supported = true;
            } else if (auto mp = ov::as_type_ptr<const ov::op::v1::MaxPool>(node)) {
                op.kind = KernelOpKind::MaxPool2D;
                add_input_indices(op);
                set_output(op);
                const auto in_shape = mp->get_input_shape(0);
                const auto out_shape = mp->get_output_shape(0);
                const auto k = mp->get_kernel();
                const auto s = mp->get_strides();
                const auto pb = mp->get_pads_begin();
                const auto pe = mp->get_pads_end();
                if (in_shape.size() == 4 && out_shape.size() == 4 && k.size() == 2 && s.size() == 2 && pb.size() == 2 && pe.size() == 2) {
                    op.pool.N = static_cast<uint32_t>(in_shape[0]);
                    op.pool.C = static_cast<uint32_t>(in_shape[1]);
                    op.pool.H = static_cast<uint32_t>(in_shape[2]);
                    op.pool.W = static_cast<uint32_t>(in_shape[3]);
                    op.pool.outH = static_cast<uint32_t>(out_shape[2]);
                    op.pool.outW = static_cast<uint32_t>(out_shape[3]);
                    op.pool.kernelH = static_cast<uint32_t>(k[0]);
                    op.pool.kernelW = static_cast<uint32_t>(k[1]);
                    op.pool.strideH = static_cast<uint32_t>(s[0]);
                    op.pool.strideW = static_cast<uint32_t>(s[1]);
                    op.pool.padTop = static_cast<uint32_t>(pb[0]);
                    op.pool.padLeft = static_cast<uint32_t>(pb[1]);
                }
                supported = true;
            } else if (auto ap = ov::as_type_ptr<const ov::op::v1::AvgPool>(node)) {
                op.kind = KernelOpKind::AvgPool2D;
                add_input_indices(op);
                set_output(op);
                const auto in_shape = ap->get_input_shape(0);
                const auto out_shape = ap->get_output_shape(0);
                const auto k = ap->get_kernel();
                const auto s = ap->get_strides();
                const auto pb = ap->get_pads_begin();
                const auto pe = ap->get_pads_end();
                op.pool.exclude_pad = ap->get_exclude_pad();
                if (in_shape.size() == 4 && out_shape.size() == 4 && k.size() == 2 && s.size() == 2 && pb.size() == 2 && pe.size() == 2) {
                    op.pool.N = static_cast<uint32_t>(in_shape[0]);
                    op.pool.C = static_cast<uint32_t>(in_shape[1]);
                    op.pool.H = static_cast<uint32_t>(in_shape[2]);
                    op.pool.W = static_cast<uint32_t>(in_shape[3]);
                    op.pool.outH = static_cast<uint32_t>(out_shape[2]);
                    op.pool.outW = static_cast<uint32_t>(out_shape[3]);
                    op.pool.kernelH = static_cast<uint32_t>(k[0]);
                    op.pool.kernelW = static_cast<uint32_t>(k[1]);
                    op.pool.strideH = static_cast<uint32_t>(s[0]);
                    op.pool.strideW = static_cast<uint32_t>(s[1]);
                    op.pool.padTop = static_cast<uint32_t>(pb[0]);
                    op.pool.padLeft = static_cast<uint32_t>(pb[1]);
                }
                supported = true;
            } else if (ov::as_type_ptr<const ov::op::v1::Convolution>(node) ||
                       ov::as_type_ptr<const ov::op::v1::GroupConvolution>(node)) {
                const bool is_group = static_cast<bool>(ov::as_type_ptr<const ov::op::v1::GroupConvolution>(node));
                const auto in_shape = node->get_input_shape(0);
                const auto w_shape = node->get_input_shape(1);
                const auto out_shape = node->get_output_shape(0);

                // Rank-5 -> Conv3D, Rank-4 -> Conv2D
                if (in_shape.size() == 5 && w_shape.size() >= 5) {
                    if (is_group) {
                        METAL_LOG_DEBUG("mlir", "[METAL MLIR] Flat IR: GroupConvolution 3D not supported yet");
                        continue;
                    }
                    op.kind = KernelOpKind::Conv3D;
                    add_input_indices(op);
                    set_output(op);
                    const auto& s = ov::as_type_ptr<const ov::op::v1::Convolution>(node)->get_strides();
                    const auto& pb = ov::as_type_ptr<const ov::op::v1::Convolution>(node)->get_pads_begin();
                    const auto& pe = ov::as_type_ptr<const ov::op::v1::Convolution>(node)->get_pads_end();
                    const auto& dil = ov::as_type_ptr<const ov::op::v1::Convolution>(node)->get_dilations();
                    op.conv3d.N = static_cast<uint32_t>(in_shape[0]);
                    op.conv3d.C_in = static_cast<uint32_t>(in_shape[1]);
                    op.conv3d.D = static_cast<uint32_t>(in_shape[2]);
                    op.conv3d.H = static_cast<uint32_t>(in_shape[3]);
                    op.conv3d.W = static_cast<uint32_t>(in_shape[4]);
                    if (s.size() == 3) {
                        op.conv3d.strideD = static_cast<uint32_t>(s[0]);
                        op.conv3d.strideH = static_cast<uint32_t>(s[1]);
                        op.conv3d.strideW = static_cast<uint32_t>(s[2]);
                    }
                    if (dil.size() == 3) {
                        op.conv3d.dilationD = static_cast<uint32_t>(dil[0]);
                        op.conv3d.dilationH = static_cast<uint32_t>(dil[1]);
                        op.conv3d.dilationW = static_cast<uint32_t>(dil[2]);
                    }
                    if (pb.size() == 3 && pe.size() == 3) {
                        op.conv3d.padFront = static_cast<uint32_t>(pb[0]);
                        op.conv3d.padTop = static_cast<uint32_t>(pb[1]);
                        op.conv3d.padLeft = static_cast<uint32_t>(pb[2]);
                        op.conv3d.padBack = static_cast<uint32_t>(pe[0]);
                        op.conv3d.padBottom = static_cast<uint32_t>(pe[1]);
                        op.conv3d.padRight = static_cast<uint32_t>(pe[2]);
                    }
                    op.conv3d.C_out = static_cast<uint32_t>(w_shape[0]);
                    op.conv3d.kernelD = static_cast<uint32_t>(w_shape[2]);
                    op.conv3d.kernelH = static_cast<uint32_t>(w_shape[3]);
                    op.conv3d.kernelW = static_cast<uint32_t>(w_shape[4]);
                    op.conv3d.element_type = static_cast<uint32_t>(static_cast<ov::element::Type_t>(node->get_input_element_type(0)));
                    if (out_shape.size() == 5) {
                        op.conv3d.outD = static_cast<uint32_t>(out_shape[2]);
                        op.conv3d.outH = static_cast<uint32_t>(out_shape[3]);
                        op.conv3d.outW = static_cast<uint32_t>(out_shape[4]);
                        KernelTensor* out_t = op.output;
                        if (out_t) {
                            out_t->shape = {static_cast<int64_t>(out_shape[0]), static_cast<int64_t>(out_shape[1]),
                                            static_cast<int64_t>(out_shape[2]), static_cast<int64_t>(out_shape[3]),
                                            static_cast<int64_t>(out_shape[4])};
                        }
                    }
                    supported = true;
                } else if (in_shape.size() == 4 && w_shape.size() >= 4) {
                    op.kind = KernelOpKind::Conv2D;
                    add_input_indices(op);
                    set_output(op);
                    const auto& s = is_group
                        ? ov::as_type_ptr<const ov::op::v1::GroupConvolution>(node)->get_strides()
                        : ov::as_type_ptr<const ov::op::v1::Convolution>(node)->get_strides();
                    const auto& pb = is_group
                        ? ov::as_type_ptr<const ov::op::v1::GroupConvolution>(node)->get_pads_begin()
                        : ov::as_type_ptr<const ov::op::v1::Convolution>(node)->get_pads_begin();
                    const auto& pe = is_group
                        ? ov::as_type_ptr<const ov::op::v1::GroupConvolution>(node)->get_pads_end()
                        : ov::as_type_ptr<const ov::op::v1::Convolution>(node)->get_pads_end();
                    op.conv2d.N = static_cast<uint32_t>(in_shape[0]);
                    op.conv2d.H = static_cast<uint32_t>(in_shape[2]);
                    op.conv2d.W = static_cast<uint32_t>(in_shape[3]);
                    op.conv2d.strideH = static_cast<uint32_t>(s[0]);
                    op.conv2d.strideW = static_cast<uint32_t>(s[1]);
                    op.conv2d.padTop = static_cast<uint32_t>(pb[0]);
                    op.conv2d.padLeft = static_cast<uint32_t>(pb[1]);
                    op.conv2d.padBottom = static_cast<uint32_t>(pe[0]);
                    op.conv2d.padRight = static_cast<uint32_t>(pe[1]);
                    const auto& dil = is_group
                        ? ov::as_type_ptr<const ov::op::v1::GroupConvolution>(node)->get_dilations()
                        : ov::as_type_ptr<const ov::op::v1::Convolution>(node)->get_dilations();
                    if (dil.size() == 2) {
                        op.conv2d.dilationH = static_cast<uint32_t>(dil[0]);
                        op.conv2d.dilationW = static_cast<uint32_t>(dil[1]);
                    }
                    if (auto c = ov::as_type_ptr<const ov::op::v1::Convolution>(node)) {
                        op.conv2d.padType = static_cast<uint32_t>(c->get_auto_pad());
                    } else if (auto g = ov::as_type_ptr<const ov::op::v1::GroupConvolution>(node)) {
                        op.conv2d.padType = static_cast<uint32_t>(g->get_auto_pad());
                    }
                    op.conv2d.element_type = static_cast<uint32_t>(static_cast<ov::element::Type_t>(node->get_input_element_type(0)));
                    if (!is_group) {
                        op.conv2d.C_in = static_cast<uint32_t>(in_shape[1]);
                        op.conv2d.groups = 1;
                        op.conv2d.C_in_per_group = op.conv2d.C_in;
                        op.conv2d.C_out = static_cast<uint32_t>(w_shape[0]);
                        op.conv2d.C_out_per_group = op.conv2d.C_out;
                        op.conv2d.kernelH = static_cast<uint32_t>(w_shape[2]);
                        op.conv2d.kernelW = static_cast<uint32_t>(w_shape[3]);
                    } else {
                        uint32_t groups = static_cast<uint32_t>(w_shape[0]);
                        uint32_t c_out_pg = static_cast<uint32_t>(w_shape[1]);
                        uint32_t c_in_pg = static_cast<uint32_t>(w_shape[2]);
                        op.conv2d.groups = groups;
                        op.conv2d.C_in = static_cast<uint32_t>(in_shape[1]);
                        op.conv2d.C_in_per_group = c_in_pg;
                        op.conv2d.C_out = groups * c_out_pg;
                        op.conv2d.C_out_per_group = c_out_pg;
                        op.conv2d.kernelH = static_cast<uint32_t>(w_shape[3]);
                        op.conv2d.kernelW = static_cast<uint32_t>(w_shape[4]);
                    }
                    if (out_shape.size() == 4) {
                        KernelTensor* out_t = op.output;
                        if (out_t) {
                            out_t->shape = {static_cast<int64_t>(out_shape[0]), static_cast<int64_t>(out_shape[1]),
                                            static_cast<int64_t>(out_shape[2]), static_cast<int64_t>(out_shape[3])};
                        }
                    }
                    supported = true;
                } else {
                    METAL_LOG_DEBUG("mlir", "[METAL MLIR] Flat IR: Convolution rank not supported");
                    continue;
                }
            } else if (ov::as_type_ptr<const ov::op::v0::BatchNormInference>(node)) {
                op.kind = KernelOpKind::BatchNorm2D;
                add_input_indices(op);
                set_output(op);
                const auto in_shape = node->get_input_shape(0);
                if (in_shape.size() == 4) {
                    op.batchnorm.N = static_cast<uint32_t>(in_shape[0]);
                    op.batchnorm.C = static_cast<uint32_t>(in_shape[1]);
                    op.batchnorm.H = static_cast<uint32_t>(in_shape[2]);
                    op.batchnorm.W = static_cast<uint32_t>(in_shape[3]);
                }
                const auto bn_v5 = ov::as_type_ptr<const ov::op::v5::BatchNormInference>(node);
                const auto bn_v0 = ov::as_type_ptr<const ov::op::v0::BatchNormInference>(node);
                const bool use_v5 = static_cast<bool>(bn_v5);
                op.batchnorm.eps = static_cast<float>(use_v5 ? bn_v5->get_eps_value() : bn_v0->get_eps_value());
                supported = true;
            } else if (auto concat = ov::as_type_ptr<const ov::op::v0::Concat>(node)) {
                const auto axis = concat->get_axis();
                const size_t inputs_count = concat->get_input_size();
                if (inputs_count == 0) continue;
                const auto out_shape = concat->get_output_shape(0);
                if (axis < 0 || static_cast<size_t>(axis) >= out_shape.size()) continue;
                uint64_t outer = 1;
                for (size_t i = 0; i < static_cast<size_t>(axis); ++i) outer *= static_cast<uint64_t>(out_shape[i]);
                uint64_t inner = 1;
                for (size_t i = static_cast<size_t>(axis) + 1; i < out_shape.size(); ++i) inner *= static_cast<uint64_t>(out_shape[i]);

                size_t out_idx = make_tensor(node->output(0));
                KernelTensor* out_t = &m_flat_tensors[out_idx];
                uint64_t axis_offset = 0;
                for (size_t i = 0; i < inputs_count; ++i) {
                    KernelOp cop{};
                    cop.kind = KernelOpKind::Concat;
                    size_t in_idx = make_tensor(node->input_value(i));
                    cop.input0 = &m_flat_tensors[in_idx];
                    cop.output = out_t;
                    const auto in_shape = node->get_input_shape(i);
                    uint64_t axis_len = static_cast<uint64_t>(in_shape[static_cast<size_t>(axis)]);
                    cop.concat.dtype = resolve_metal_dtype(node->get_output_element_type(0));
                    cop.concat.axis = axis;
                    cop.concat.outer = outer;
                    cop.concat.inner = inner;
                    cop.concat.axis_sizes = {axis_len};
                    cop.concat.axis_offsets = {axis_offset};
                    axis_offset += axis_len;
                    propagate_dtype(cop);
                    m_flat_ops.push_back(cop);
                }
                continue;
            } else if (auto interp0 = ov::as_type_ptr<const ov::op::v0::Interpolate>(node)) {
                op.kind = KernelOpKind::Interpolate;
                add_input_indices(op);
                set_output(op);
                const auto in_shape = node->get_input_shape(0);
                const auto out_shape = node->get_output_shape(0);
                if (in_shape.size() == 4 && out_shape.size() == 4) {
                    op.interpolate.dtype = resolve_metal_dtype(node->get_output_element_type(0));
                    op.interpolate.N = static_cast<uint32_t>(in_shape[0]);
                    op.interpolate.C = static_cast<uint32_t>(in_shape[1]);
                    op.interpolate.H_in = static_cast<uint32_t>(in_shape[2]);
                    op.interpolate.W_in = static_cast<uint32_t>(in_shape[3]);
                    op.interpolate.H_out = static_cast<uint32_t>(out_shape[2]);
                    op.interpolate.W_out = static_cast<uint32_t>(out_shape[3]);
                    op.interpolate.nearest = interp0->get_attrs().mode == "nearest";
                    op.interpolate.align_corners = false;
                }
                supported = true;
            } else if (auto interp4 = ov::as_type_ptr<const ov::op::v4::Interpolate>(node)) {
                op.kind = KernelOpKind::Interpolate;
                add_input_indices(op);
                set_output(op);
                const auto in_shape = node->get_input_shape(0);
                const auto out_shape = node->get_output_shape(0);
                if (in_shape.size() == 4 && out_shape.size() == 4) {
                    op.interpolate.dtype = resolve_metal_dtype(node->get_output_element_type(0));
                    op.interpolate.N = static_cast<uint32_t>(in_shape[0]);
                    op.interpolate.C = static_cast<uint32_t>(in_shape[1]);
                    op.interpolate.H_in = static_cast<uint32_t>(in_shape[2]);
                    op.interpolate.W_in = static_cast<uint32_t>(in_shape[3]);
                    op.interpolate.H_out = static_cast<uint32_t>(out_shape[2]);
                    op.interpolate.W_out = static_cast<uint32_t>(out_shape[3]);
                    const auto& attrs = interp4->get_attrs();
                    op.interpolate.nearest = attrs.mode == ov::op::v4::Interpolate::InterpolateMode::NEAREST;
                    op.interpolate.align_corners = attrs.coordinate_transformation_mode == ov::op::v4::Interpolate::CoordinateTransformMode::ALIGN_CORNERS;
                }
                supported = true;
            } else if (auto tr = ov::as_type_ptr<const ov::op::v1::Transpose>(node)) {
                auto order_const = std::dynamic_pointer_cast<const ov::op::v0::Constant>(tr->input_value(1).get_node_shared_ptr());
                if (!order_const) {
                    METAL_LOG_DEBUG("mlir", "[METAL MLIR] Transpose order is not constant; skipping");
                    continue;
                }
                auto order = order_const->cast_vector<int64_t>();
                if (order.size() != tr->get_input_shape(0).size()) {
                    METAL_LOG_DEBUG("mlir", "[METAL MLIR] Transpose order size mismatch; skipping");
                    continue;
                }
                op.kind = KernelOpKind::Transpose;
                add_input_indices(op);
                set_output(op);
                op.transpose.in_shape.assign(tr->get_input_shape(0).begin(), tr->get_input_shape(0).end());
                op.transpose.out_shape.assign(tr->get_output_shape(0).begin(), tr->get_output_shape(0).end());
                op.transpose.perm.assign(order.begin(), order.end());
                supported = true;
            } else if (auto rs = ov::as_type_ptr<const ov::op::v1::Reshape>(node)) {
                op.kind = KernelOpKind::Reshape;
                add_input_indices(op);
                set_output(op);
                op.reshape.in_shape.assign(rs->get_input_shape(0).begin(), rs->get_input_shape(0).end());
                op.reshape.out_shape.assign(rs->get_output_shape(0).begin(), rs->get_output_shape(0).end());
                supported = true;
            } else if (ov::as_type_ptr<const ov::op::v0::Squeeze>(node) ||
                       ov::as_type_ptr<const ov::op::v0::Unsqueeze>(node) ||
                       ov::as_type_ptr<const ov::op::v0::Convert>(node)) {
                // Layout-preserving ops: record outputs only.
                set_output(op);
                continue;
            } else {
                METAL_LOG_DEBUG("mlir", "[METAL MLIR] Flat IR: unsupported node " + node->get_friendly_name() +
                          " type=" + node->get_type_name());
                continue;
            }

            if (supported) {
                propagate_dtype(op);
                m_flat_ops.push_back(op);
            }
        }

        if (!m_flat_ops.empty()) {
            Segment seg{};
            seg.first_op_index = 0;
            seg.op_count = m_flat_ops.size();
            for (const auto& in : model->inputs()) seg.input_ports.push_back(in);
            for (const auto& out : model->outputs()) seg.output_ports.push_back(out);
            m_flat_segments.push_back(std::move(seg));
        }

#if METAL_MLIR_DEBUG
        METAL_LOG_DEBUG("mlir", "[METAL MLIR] Flat IR ops count: " + std::to_string(m_flat_ops.size()));
        const size_t dump_n = std::min<size_t>(m_flat_ops.size(), 8);
        for (size_t i = 0; i < dump_n; ++i) {
            const auto& op = m_flat_ops[i];
            auto kind_int = static_cast<int>(op.kind);
            std::string in0 = op.input0 ? op.input0->name : "-";
            std::string in1 = op.input1 ? op.input1->name : "-";
            std::string out = op.output ? op.output->name : "-";
            METAL_LOG_DEBUG("mlir", "[METAL MLIR] Flat IR op[" + std::to_string(i) + "]: kind=" + std::to_string(kind_int) +
                      " in0=" + in0 + " in1=" + in1 + " out=" + out);
        }
#endif
    }
