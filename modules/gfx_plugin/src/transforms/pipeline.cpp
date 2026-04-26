// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "transforms/pipeline.hpp"

#include "transforms/gfx_layout_cleanup.hpp"

#include "openvino/pass/manager.hpp"
#include "transformations/common_optimizations/common_optimizations.hpp"
#include "transformations/common_optimizations/rms_fusion.hpp"
#include "transformations/control_flow/unroll_if.hpp"
#include "transformations/fp16_compression/convert_compression_only_to_legacy.hpp"
#include "transformations/fp16_compression/mark_decompression_convert_constant_folding.hpp"
#include "transformations/op_conversions/convert_avgpool_downgrade.hpp"
#include "transformations/op_conversions/convert_maxpool_downgrade.hpp"
#include "transformations/op_conversions/convert_mod.hpp"
#include "transformations/op_conversions/convert_reduce_to_pooling.hpp"

namespace ov {
namespace gfx_plugin {
namespace transforms {

std::shared_ptr<const ov::Model> run_pipeline(const std::shared_ptr<const ov::Model>& model) {
    OPENVINO_ASSERT(model, "run_pipeline: model is null");
    // Work on a clone to preserve the original model passed by the caller.
    auto cloned = model->clone();

    ov::pass::Manager manager("Plugin:GFX");
    // Common optimizations from OpenVINO transformations library.
    manager.register_pass<ov::pass::CommonOptimizations>();
    // LLM RMSNorm frequently arrives as Power->ReduceMean->Add->Sqrt/Divide->Multiply->Multiply.
    // Fuse both no-tail-convert and Divide variants into one backend-lowerable op.
    manager.register_pass<ov::pass::RMSFusion>(false, true);
    // Plugin-local structural cleanup before stage selection / MLIR lowering.
    manager.register_pass<ov::gfx_plugin::transforms::GfxLayoutCleanup>();

    // Align behaviour with template plugin: disable a few transformations that can be harmful for backend mapping.
    auto pass_config = manager.get_pass_config();
    pass_config->disable<ov::pass::UnrollIf>();
    pass_config->disable<ov::pass::ConvertMaxPool14ToMaxPool8>();
    pass_config->disable<ov::pass::ConvertAvgPool14ToAvgPool1>();
    pass_config->disable<ov::pass::ConvertReduceSumToPooling>();
    pass_config->disable<ov::pass::ConvertMod>();

    // Allow FP16 converts to be folded and decompression converts to be upgraded back to FP32.
    pass_config->disable<ov::pass::DisableDecompressionConvertConstantFolding>();
    pass_config->disable<ov::pass::ConvertCompressedOnlyToLegacy>();

    manager.run_passes(cloned);
    return cloned;
}

}  // namespace transforms
}  // namespace gfx_plugin
}  // namespace ov
