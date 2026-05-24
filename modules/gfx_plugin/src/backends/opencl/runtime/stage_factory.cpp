// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "backends/opencl/runtime/stage_factory.hpp"

#include "backends/opencl/runtime/opencl_source_stage.hpp"
#include "openvino/core/except.hpp"
#include "plugin/stateful_stage.hpp"
#include "runtime/execution_dispatcher.hpp"

namespace ov {
namespace gfx_plugin {

std::unique_ptr<GpuStage> create_opencl_stage(const std::shared_ptr<const ov::Node>& node,
                                              void*,
                                              void*) {
    if (auto stateful = create_stateful_stage(node)) {
        return stateful;
    }
    if (auto source_stage = create_opencl_source_stage(node, OpenClRuntimeContext::instance())) {
        return source_stage;
    }
    OPENVINO_THROW("GFX OpenCL: source-kernel stage materialization is not wired for op ",
                   node ? node->get_type_name() : "<null>",
                   ". Baseline OpenCL source artifacts currently cover f32 linear copy/layout, f32/i32/i64 convert casts, and "
                   "static f32 matmul/softmax plus transpose/slice/strided-slice/range/tile/gather/gather-elements/gather-nd/scatter-update/scatter-elements/scatter-nd/shapeof/concat/split/variadic-split plus unary/binary/compare/select elementwise seeds, including scalar and rank-1..4 broadcast binary/compare/select cases; "
                   "add the next operation through the common manifest/artifact/stage path before enabling it.");
}

void ensure_opencl_stage_factory_registered() {
    static const bool registered = GpuStageFactory::register_factory(GpuBackend::OpenCL, &create_opencl_stage);
    (void)registered;
}

namespace {
const bool kRegistered = (ensure_opencl_stage_factory_registered(), true);
}  // namespace

}  // namespace gfx_plugin
}  // namespace ov
