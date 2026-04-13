// OpenVINO Extension Entry Point for BEVPool Operations

#include <openvino/core/extension.hpp>
#include <openvino/core/op_extension.hpp>
#include <openvino/frontend/extension.hpp>

#include "bev_pool_op.hpp"

OPENVINO_CREATE_EXTENSIONS(
    std::vector<ov::Extension::Ptr>({
        // Original fused BEVPool (float CAS atomics)
        std::make_shared<ov::OpExtension<BEVFusionExtension::BEVPool>>(),
        std::make_shared<ov::frontend::OpExtension<BEVFusionExtension::BEVPool>>(),
        // Optimized: BEVPoolScatter (int32 native atomics)
        std::make_shared<ov::OpExtension<BEVFusionExtension::BEVPoolScatter>>(),
        std::make_shared<ov::frontend::OpExtension<BEVFusionExtension::BEVPoolScatter>>(),
        // Optimized: BEVPoolConvert (int32 → float32)
        std::make_shared<ov::OpExtension<BEVFusionExtension::BEVPoolConvert>>(),
        std::make_shared<ov::frontend::OpExtension<BEVFusionExtension::BEVPoolConvert>>(),
    }));
