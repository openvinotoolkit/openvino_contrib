ModelAnalysis analyze_model_for_mlir(const std::shared_ptr<const ov::Model>& model,
                                     const MlirCapabilities& caps) {
    ModelAnalysis res;

    auto mark_disabled = [&](const std::shared_ptr<const ov::Node>& node) {
        res.has_disabled_ops = true;
        res.disabled_list.push_back(describe_node(node));
        METAL_LOG_DEBUG("mlir", "[METAL MLIR] Fallback reason: disabled op " + describe_node(node));
        fprintf(stderr, "[METAL][analysis][disabled] %s\n", describe_node(node).c_str());
    };
    auto mark_future = [&](const std::shared_ptr<const ov::Node>& node) {
        res.has_future_ops = true;
        res.future_list.push_back(describe_node(node));
        METAL_LOG_DEBUG("mlir", "[METAL MLIR] Fallback reason: future op " + describe_node(node));
        fprintf(stderr, "[METAL][analysis][future] %s\n", describe_node(node).c_str());
    };
    auto mark_unsupported = [&](const std::shared_ptr<const ov::Node>& node) {
        res.has_unsupported_ops = true;
        res.unsupported_list.push_back(describe_node(node));
        METAL_LOG_DEBUG("mlir", "[METAL MLIR] Fallback reason: unsupported op " + describe_node(node));
        fprintf(stderr, "[METAL][analysis][unsupported] %s\n", describe_node(node).c_str());
    };

    for (const auto& node : model->get_ordered_ops()) {
        if (ov::as_type_ptr<const ov::op::v0::Parameter>(node) ||
            ov::as_type_ptr<const ov::op::v0::Constant>(node) ||
            ov::as_type_ptr<const ov::op::v0::Result>(node)) {
            continue;
        } else if (ov::as_type_ptr<const ov::op::v1::Split>(node) ||
                   ov::as_type_ptr<const ov::op::v1::VariadicSplit>(node)) {
            if (!caps.split) {
                mark_disabled(node);
                continue;
            }
