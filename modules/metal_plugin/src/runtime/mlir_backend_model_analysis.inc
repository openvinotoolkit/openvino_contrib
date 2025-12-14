struct ModelAnalysis {
    bool has_matmul = false;
    bool has_add = false;
    bool has_mul = false;
    bool has_div = false;
    bool has_mod = false;
    bool has_floor_mod = false;
    bool has_pow = false;
    bool has_squared_diff = false;
    bool has_add_broadcast = false;
    bool has_unary = false;
    bool has_softmax = false;
    bool has_maxpool = false;
    bool has_avgpool = false;
    bool has_conv2d = false;
    bool has_conv3d = false;
    bool has_batchnorm = false;
    bool has_split = false;
    ActivationKind unary_kind = ActivationKind::Relu;
    float unary_alpha = 0.0f;
    size_t compute_ops = 0;
    bool has_future_ops = false;  // Ops recognized but not yet lowered in current backend.
    bool has_disabled_ops = false;  // Ops known but capability flag is off.
    bool has_unsupported_ops = false;  // Completely unknown to the backend.
    std::vector<std::string> disabled_list;
    std::vector<std::string> unsupported_list;
    std::vector<std::string> future_list;
};

