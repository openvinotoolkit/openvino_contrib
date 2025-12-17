void MlirBackend::Impl::ensure_fallback(const std::shared_ptr<const ov::Model>& model) {
        (void)model;
        OPENVINO_THROW("METAL: CPU fallback disabled");
    }

    void MlirBackend::Impl::ensure_fallback() {
        OPENVINO_THROW("METAL: CPU fallback disabled");
    }

    // High-precision host softmax for dynamic-only graphs (single Softmax op).
bool MlirBackend::Impl::run_host_softmax(const std::vector<ov::Tensor>& inputs, std::vector<ov::Tensor>& outputs) const {
        (void)inputs;
        (void)outputs;
        OPENVINO_THROW("METAL: host softmax disabled (GPU only)");
    }
