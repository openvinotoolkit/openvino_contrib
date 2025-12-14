// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//
// Included from mlir_backend.mm to keep access to MlirBackend::Impl internals.

bool MlirBackend::Impl::try_build_eltwise(const std::shared_ptr<const ov::Model>& model,
                                          const ModelAnalysis& analysis,
                                          MetalKernelCompiler& compiler,
                                          std::string& log) {
    const bool has_add = analysis.has_add && !analysis.has_add_broadcast;
    const bool has_add_broadcast = analysis.has_add_broadcast;
    const bool has_matmul = analysis.has_matmul;
    const bool has_maxpool = analysis.has_maxpool;
    const bool has_avgpool = analysis.has_avgpool;
    const bool has_conv2d = analysis.has_conv2d;
    const bool has_batchnorm = analysis.has_batchnorm;

    if (!(has_add || has_add_broadcast || analysis.has_unary || analysis.has_mul || analysis.has_div || analysis.has_pow ||
          analysis.has_mod || analysis.has_floor_mod || analysis.has_squared_diff) ||
        has_matmul || analysis.has_softmax) {
        return false;
    }

    // Build IR op manually to preserve dtype/kind/broadcast for all eltwise variants.
    ov::element::Type elt_type = ov::element::f32;
    KernelOpKind detected_kind = KernelOpKind::ElementwiseAdd;
    bool is_squared_difference = false;
    bool mod_from_pattern = false;
    ov::Output<const ov::Node> mod_a, mod_b;
    bool mod_is_floor = false;
    for (const auto& n : model->get_ordered_ops()) {
        if (auto p = std::dynamic_pointer_cast<const ov::op::v1::Add>(n)) { detected_kind = KernelOpKind::ElementwiseAdd; elt_type = p->get_output_element_type(0); break; }
        if (auto p = std::dynamic_pointer_cast<const ov::op::v1::Subtract>(n)) { detected_kind = KernelOpKind::ElementwiseSub; elt_type = p->get_output_element_type(0); break; }
        if (auto p = std::dynamic_pointer_cast<const ov::op::v1::Multiply>(n)) { detected_kind = KernelOpKind::ElementwiseMul; elt_type = p->get_output_element_type(0); break; }
        if (auto p = std::dynamic_pointer_cast<const ov::op::v1::Divide>(n)) { detected_kind = KernelOpKind::ElementwiseDiv; elt_type = p->get_output_element_type(0); break; }
        if (auto p = std::dynamic_pointer_cast<const ov::op::v1::Power>(n)) { detected_kind = KernelOpKind::ElementwisePow; elt_type = p->get_output_element_type(0); break; }
        if (auto p = std::dynamic_pointer_cast<const ov::op::v1::Mod>(n)) { detected_kind = KernelOpKind::ElementwiseMod; elt_type = p->get_output_element_type(0); break; }
        if (auto p = std::dynamic_pointer_cast<const ov::op::v1::FloorMod>(n)) { detected_kind = KernelOpKind::ElementwiseFloorMod; elt_type = p->get_output_element_type(0); break; }
        if (auto p = std::dynamic_pointer_cast<const ov::op::v0::SquaredDifference>(n)) { detected_kind = KernelOpKind::ElementwiseSub; elt_type = p->get_output_element_type(0); is_squared_difference = true; break; }
        if (auto pat = match_mod_decomposition(n)) {
            auto [a_out, b_out, is_floor] = *pat;
            detected_kind = is_floor ? KernelOpKind::ElementwiseFloorMod : KernelOpKind::ElementwiseMod;
            elt_type = a_out.get_element_type();
            mod_a = a_out;
            mod_b = b_out;
            mod_is_floor = is_floor;
            mod_from_pattern = true;
            break;
        }
    }

    // shapes from detected node (allow dynamic via PartialShape)
    std::vector<int64_t> s0, s1, sout;
    std::shared_ptr<const ov::Node> elt_node;
    auto to_vec = [](const ov::PartialShape& ps) {
        std::vector<int64_t> v;
        v.reserve(ps.rank().is_static() ? ps.rank().get_length() : 8);
        for (const auto& d : ps) v.push_back(d.is_dynamic() ? -1 : static_cast<int64_t>(d.get_length()));
        return v;
    };
    std::shared_ptr<const ov::Model> model_for_build = model;
    if (mod_from_pattern) {
        s0 = to_vec(mod_a.get_partial_shape());
        s1 = to_vec(mod_b.get_partial_shape());
        // result shape = broadcast(s0,s1); will be recomputed below
        sout = {};
        // fabricate a minimal model with a single Mod/FloorMod op to feed the MLIR builder
        ov::ParameterVector params;
        auto p0 = std::make_shared<ov::op::v0::Parameter>(elt_type, mod_a.get_partial_shape());
        auto p1 = std::make_shared<ov::op::v0::Parameter>(elt_type, mod_b.get_partial_shape());
        params = {p0, p1};
        std::shared_ptr<ov::Node> mod_op;
        if (mod_is_floor)
            mod_op = std::make_shared<ov::op::v1::FloorMod>(p0, p1, ov::op::AutoBroadcastType::NUMPY);
        else
            mod_op = std::make_shared<ov::op::v1::Mod>(p0, p1, ov::op::AutoBroadcastType::NUMPY);
        auto res = std::make_shared<ov::op::v0::Result>(mod_op);
        auto mod_model = std::make_shared<ov::Model>(ov::ResultVector{res}, params);
        model_for_build = mod_model;
        elt_node = mod_op;
    } else {
        for (const auto& n : model->get_ordered_ops()) {
            if (n->get_output_element_type(0) == elt_type) {
                s0 = to_vec(n->get_input_partial_shape(0));
                s1 = to_vec(n->get_input_partial_shape(1));
                sout = to_vec(n->get_output_partial_shape(0));
                elt_node = n;
                break;
            }
        }
    }

    m_ir.tensors.clear();
    m_ir.tensors.push_back({"in0", s0, resolve_metal_dtype(elt_type)});
    m_ir.tensors.push_back({"in1", s1, resolve_metal_dtype(elt_type)});
    m_ir.tensors.push_back({"out", sout, resolve_metal_dtype(elt_type)});
    auto params = model->get_parameters();
    if (!params.empty()) {
        m_ir.tensors[0].from_parameter = true;
        m_ir.tensors[0].source_node = params[0].get();
    }
    if (params.size() > 1) {
        m_ir.tensors[1].from_parameter = true;
        m_ir.tensors[1].source_node = params[1].get();
    }
    // Tag constants for proper buffer upload
    if (elt_node) {
        auto maybe_const0 = std::dynamic_pointer_cast<const ov::op::v0::Constant>(elt_node->get_input_node_shared_ptr(0));
        if (maybe_const0) {
            m_ir.tensors[0].from_constant = true;
            m_ir.tensors[0].source_node = maybe_const0.get();
            m_ir.tensors[0].const_data = maybe_const0->cast_vector<float>();
        }
        auto maybe_const1 = std::dynamic_pointer_cast<const ov::op::v0::Constant>(elt_node->get_input_node_shared_ptr(1));
        if (maybe_const1) {
            m_ir.tensors[1].from_constant = true;
            m_ir.tensors[1].source_node = maybe_const1.get();
            m_ir.tensors[1].const_data = maybe_const1->cast_vector<float>();
        }
    }

    KernelOp op{};
    op.kind = detected_kind;
    op.input0 = &m_ir.tensors[0];
    op.input1 = &m_ir.tensors[1];
    op.output = &m_ir.tensors[2];
    op.dtype = resolve_metal_dtype(elt_type);
    op.element_type = static_cast<uint32_t>(static_cast<ov::element::Type_t>(elt_type));

    auto to_ov_shape = [](const std::vector<int64_t>& shp) {
        ov::Shape r;
        r.reserve(shp.size());
        for (auto d : shp) r.push_back(static_cast<size_t>(d < 0 ? 1 : d));
        return r;
    };
    auto br = compute_broadcast(to_ov_shape(op.input0->shape), to_ov_shape(op.input1->shape));
    std::vector<int64_t> br_out;
    if (br.success) br_out.assign(br.out_shape.begin(), br.out_shape.end());
    op.is_broadcast = br.success && (br_out != op.input0->shape || br_out != op.input1->shape);
    op.out_shape = br.success ? br_out : op.output->shape;
    if (op.out_shape.empty())
        op.out_shape.push_back(1);  // treat scalar as [1] for runtime buffers
    op.stride0 = br.success ? br.stride0 : std::vector<int64_t>(op.out_shape.size(), 1);
    op.stride1 = br.success ? br.stride1 : std::vector<int64_t>(op.out_shape.size(), 1);
    if (op.stride0.empty()) op.stride0.push_back(0);
    if (op.stride1.empty()) op.stride1.push_back(0);

    m_ir.ops.clear();
    m_ir.ops.push_back(op);
    m_ops = m_ir.ops;
    m_segments.clear();
    if (!m_ops.empty()) {
        m_segments.push_back(Segment{0, m_ops.size()});
    }
    m_ops_from_flat_segment = false;
    m_pipelines.clear();
    mlir::MLIRContext ctx;
    mlir::ModuleOp module;
    switch (detected_kind) {
        case KernelOpKind::ElementwiseAdd: module = build_mlir_add_from_model(model_for_build, ctx); break;
        case KernelOpKind::ElementwiseSub:
            if (is_squared_difference)
                module = build_mlir_squared_difference_from_model(model_for_build, ctx);
            else
                module = build_mlir_sub_from_model(model_for_build, ctx);
            break;
        case KernelOpKind::ElementwiseMul: module = build_mlir_mul_from_model(model_for_build, ctx); break;
        case KernelOpKind::ElementwiseDiv: module = build_mlir_div_from_model(model_for_build, ctx); break;
        case KernelOpKind::ElementwisePow: module = build_mlir_pow_from_model(model_for_build, ctx); break;
        case KernelOpKind::ElementwiseMod: module = build_mlir_mod_from_model(model_for_build, ctx); break;
        case KernelOpKind::ElementwiseFloorMod: module = build_mlir_floor_mod_from_model(model_for_build, ctx); break;
        default: OPENVINO_THROW("Unexpected eltwise kind in MLIR builder");
    }
    run_mlir_pipeline(module);

    EltwiseCodegenDesc desc;
    desc.kind = op.kind;
    desc.eltwise_kind = op.kind;
    desc.element_type = elt_type;
    desc.is_broadcast = op.is_broadcast;
    desc.out_shape = op.out_shape;
    desc.stride0 = op.stride0;
    desc.stride1 = op.stride1;

    auto source = generate_msl_from_mlir(module, desc);
    m_pipelines.push_back(compiler.compile_msl_from_source(source, "eltwise_kernel", log));
    return true;
}
