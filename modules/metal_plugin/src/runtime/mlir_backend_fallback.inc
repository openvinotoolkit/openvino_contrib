void MlirBackend::Impl::ensure_fallback(const std::shared_ptr<const ov::Model>& model) {
        if (m_fallback_request)
            return;
        ov::Core core;
        auto src_model = model ? model : m_original_model;
        m_fallback_model = core.compile_model(src_model, "CPU");
        m_fallback_inputs = m_fallback_model.inputs();
        m_fallback_outputs = m_fallback_model.outputs();
        m_fallback_request = std::make_unique<ov::InferRequest>(m_fallback_model.create_infer_request());
    }

    void MlirBackend::Impl::ensure_fallback() {
        ensure_fallback(m_original_model);
    }

    // High-precision host softmax for dynamic-only graphs (single Softmax op).
    bool MlirBackend::Impl::run_host_softmax(const std::vector<ov::Tensor>& inputs, std::vector<ov::Tensor>& outputs) const {
        if (!m_softmax_dynamic_only || inputs.empty())
            return false;

        METAL_LOG_DEBUG("mlir", "[METAL MLIR] Host softmax path executing");
        const auto& in = inputs[0];
        auto shape = in.get_shape();
        if (shape.empty())
            return false;

        const int64_t rank = static_cast<int64_t>(shape.size());
        int64_t axis = m_softmax_axis;
        if (axis < 0)
            axis += rank;
        if (axis < 0 || axis >= rank)
            return false;

        int64_t cols = static_cast<int64_t>(shape[axis]);
        int64_t inner = 1;
        for (int64_t i = axis + 1; i < rank; ++i)
            inner *= static_cast<int64_t>(shape[static_cast<size_t>(i)]);
        int64_t outer = 1;
        for (int64_t i = 0; i < axis; ++i)
            outer *= static_cast<int64_t>(shape[static_cast<size_t>(i)]);
        int64_t rows = outer * inner;

        ov::Tensor out{in.get_element_type(), shape};
        const bool is_f16 = in.get_element_type() == ov::element::f16;
        const auto* src_f16 = is_f16 ? in.data<ov::float16>() : nullptr;
        const auto* src_f32 = !is_f16 ? in.data<const float>() : nullptr;
        auto* dst_f16 = is_f16 ? out.data<ov::float16>() : nullptr;
        auto* dst_f32 = !is_f16 ? out.data<float>() : nullptr;

        for (int64_t r = 0; r < rows; ++r) {
            int64_t outer_idx = r / inner;
            int64_t inner_idx = r - outer_idx * inner;
            int64_t base = outer_idx * cols * inner + inner_idx;

            double max_v = -std::numeric_limits<double>::infinity();
            for (int64_t c = 0; c < cols; ++c) {
                double v = is_f16 ? static_cast<double>(src_f16[base + c * inner])
                                  : static_cast<double>(src_f32[base + c * inner]);
                if (v > max_v)
                    max_v = v;
            }

            double sum = 0.0;
            for (int64_t c = 0; c < cols; ++c) {
                double v = is_f16 ? static_cast<double>(src_f16[base + c * inner])
                                  : static_cast<double>(src_f32[base + c * inner]);
                double e = std::exp(v - max_v);
                if (is_f16)
                    dst_f16[base + c * inner] = static_cast<ov::float16>(e);
                else
                    dst_f32[base + c * inner] = static_cast<float>(e);
                sum += e;
            }
            double inv_sum = 1.0 / sum;
            for (int64_t c = 0; c < cols; ++c) {
                if (is_f16) {
                    double v = static_cast<double>(dst_f16[base + c * inner]);
                    dst_f16[base + c * inner] = static_cast<ov::float16>(v * inv_sum);
                } else {
                    dst_f32[base + c * inner] = static_cast<float>(static_cast<double>(dst_f32[base + c * inner]) * inv_sum);
                }
            }
        }

        if (!outputs.empty()) {
            outputs[0] = out;
        }
        return true;
    }