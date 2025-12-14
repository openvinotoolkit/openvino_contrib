void MlirBackend::Impl::build_segments_from_flat_ir() {
        m_flat_segments.clear();
        if (m_flat_ops.empty())
            return;

        bool all_slice = true;
        for (const auto& op : m_flat_ops) {
            if (op.kind != KernelOpKind::Slice) {
                all_slice = false;
                break;
            }
        }
        if (all_slice) {
            m_flat_segments.push_back(Segment{0, m_flat_ops.size()});
            m_ops_from_flat_segment = true;
            return;
        }

        auto is_segmentable = [](KernelOpKind k) {
            switch (k) {
                case KernelOpKind::Conv2D:
                case KernelOpKind::Conv3D:
                case KernelOpKind::BatchNorm2D:
                case KernelOpKind::Unary:
                case KernelOpKind::MatMul:
                case KernelOpKind::ElementwiseAdd:
                case KernelOpKind::ElementwiseMul:
                case KernelOpKind::ElementwiseSub:
                case KernelOpKind::ElementwiseDiv:
                case KernelOpKind::ElementwisePow:
                case KernelOpKind::ElementwiseMod:
                case KernelOpKind::ElementwiseFloorMod:
                case KernelOpKind::Softmax:
                case KernelOpKind::MaxPool2D:
                case KernelOpKind::AvgPool2D:
                case KernelOpKind::Interpolate:
                case KernelOpKind::Slice:
                case KernelOpKind::Split:
                case KernelOpKind::Concat:
                case KernelOpKind::Convert:
                case KernelOpKind::Transpose:
                case KernelOpKind::Reshape:
                    return true;
                default:
                    return false;
            }
        };

        size_t seg_start = 0;
        size_t seg_len = 0;
        auto push_seg = [&](size_t start, size_t len) {
            if (len == 0) return;
            m_flat_segments.push_back(Segment{start, len});
        };

        for (size_t i = 0; i < m_flat_ops.size(); ++i) {
            if (is_segmentable(m_flat_ops[i].kind)) {
                if (seg_len == 0) seg_start = i;
                ++seg_len;
            } else {
                push_seg(seg_start, seg_len);
                seg_len = 0;
            }
        }
        push_seg(seg_start, seg_len);

#if METAL_MLIR_DEBUG
        METAL_LOG_DEBUG("mlir", "[METAL MLIR] Segments built: " + std::to_string(m_flat_segments.size()) +
                  " over flat ops: " + std::to_string(m_flat_ops.size()));
        const size_t dump_n = std::min<size_t>(m_flat_segments.size(), 4);
        for (size_t i = 0; i < dump_n; ++i) {
            auto s = m_flat_segments[i];
            METAL_LOG_DEBUG("mlir", "[METAL MLIR] Segment[" + std::to_string(i) + "] start=" + std::to_string(s.first_op_index) +
                      " len=" + std::to_string(s.op_count));
        }
#endif

        if (!m_flat_segments.empty()) {
            m_ops_from_flat_segment = true;
        }
    }
