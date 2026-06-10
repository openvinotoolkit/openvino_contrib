// OpenVINO Extension Entry Point for Sparse Encoder

#include <openvino/core/extension.hpp>
#include <openvino/core/op_extension.hpp>
#include <openvino/frontend/extension.hpp>

#include "sparse_encoder_op.hpp"

OPENVINO_CREATE_EXTENSIONS(
    std::vector<ov::Extension::Ptr>({
        std::make_shared<ov::OpExtension<BEVFusionExtension::SparseEncoderOp>>(),
        std::make_shared<ov::frontend::OpExtension<BEVFusionExtension::SparseEncoderOp>>(),
    }));
