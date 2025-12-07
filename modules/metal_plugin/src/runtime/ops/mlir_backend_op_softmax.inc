// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//
// Included from mlir_backend.mm to keep access to MlirBackend::Impl internals.

bool MlirBackend::Impl::try_build_softmax(const std::shared_ptr<const ov::Model>& model,
                                          const ModelAnalysis& analysis,
                                          MetalKernelCompiler& compiler,
                                          std::string& log) {
    if (!(analysis.has_softmax) || analysis.has_matmul || analysis.has_add || analysis.has_add_broadcast || analysis.has_unary)
        return false;

    // Always generate a Metal softmax kernel (no CPU fallback).
    // Build a minimal IR manually (one input/one output softmax).
    int64_t axis = -1;
    for (const auto& node : model->get_ordered_ops()) {
        if (auto s1 = ov::as_type_ptr<const ov::op::v1::Softmax>(node)) { axis = s1->get_axis(); break; }
        if (auto s8 = ov::as_type_ptr<const ov::op::v8::Softmax>(node)) { axis = s8->get_axis(); break; }
    }

    auto pshape = model->input().get_partial_shape();
    if (!pshape.rank().is_static()) {
        OPENVINO_THROW("METAL Softmax expects static rank");
    }
    const int64_t rank = pshape.rank().get_length();
    if (axis < 0) axis += rank;

    std::vector<int64_t> in_shape;
    in_shape.reserve(static_cast<size_t>(rank));
    for (int64_t i = 0; i < rank; ++i) {
        const auto& d = pshape[static_cast<size_t>(i)];
        in_shape.push_back(d.is_static() ? d.get_length() : int64_t{-1});
    }

    int64_t cols = 0, inner = 0, outer = 0, rows = 0;
    if (pshape.is_static()) {
        cols = in_shape[static_cast<size_t>(axis)];
        inner = 1;
        for (int64_t i = axis + 1; i < rank; ++i) inner *= in_shape[static_cast<size_t>(i)];
        outer = 1;
        for (int64_t i = 0; i < axis; ++i) outer *= in_shape[static_cast<size_t>(i)];
        rows = outer * inner;
    }

    MetalKernelIR ir;
    ir.tensors.resize(2);
    ir.ops.clear();

    auto dtype = resolve_metal_dtype(model->input().get_element_type());
    ir.tensors[0].name = "input0";
    ir.tensors[0].shape = pshape.is_static() ? in_shape : std::vector<int64_t>{};
    ir.tensors[0].dtype = dtype;
    ir.tensors[1].name = "output0";
    ir.tensors[1].shape = pshape.is_static() ? in_shape : std::vector<int64_t>{};
    ir.tensors[1].dtype = dtype;

    KernelOp op{};
    op.kind = KernelOpKind::Softmax;
    op.input0 = &ir.tensors[0];
    op.output = &ir.tensors[1];
    op.softmax_axis = axis;
    op.rows = rows;
    op.cols = cols;
    op.inner = inner;
    op.out_shape = pshape.is_static() ? in_shape : std::vector<int64_t>{};
    op.dtype = dtype;
    ir.ops.push_back(op);

    m_ir = std::move(ir);
    m_ops = m_ir.ops;
    m_segments.clear();
    if (!m_ops.empty()) {
        m_segments.push_back(Segment{0, m_ops.size()});
    }
    m_ops_from_flat_segment = true;
    m_pipelines.clear();
    m_pipelines.push_back(compiler.compile_softmax_kernel(m_ops[0], log));
    return true;
}

