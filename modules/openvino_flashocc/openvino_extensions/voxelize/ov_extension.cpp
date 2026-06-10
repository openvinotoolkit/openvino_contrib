// OpenVINO Extension Entry Point for Voxelization Operations

#include <openvino/core/extension.hpp>
#include <openvino/core/op_extension.hpp>
#include <openvino/frontend/extension.hpp>

#include "voxelize_op.hpp"

OPENVINO_CREATE_EXTENSIONS(
    std::vector<ov::Extension::Ptr>({
        std::make_shared<ov::OpExtension<BEVFusionExtension::VoxelizeScatter>>(),
        std::make_shared<ov::frontend::OpExtension<BEVFusionExtension::VoxelizeScatter>>(),
        std::make_shared<ov::OpExtension<BEVFusionExtension::VoxelizeMean>>(),
        std::make_shared<ov::frontend::OpExtension<BEVFusionExtension::VoxelizeMean>>(),
    }));
