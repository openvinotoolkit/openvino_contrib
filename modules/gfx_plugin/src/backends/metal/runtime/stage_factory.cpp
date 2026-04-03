// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "backends/metal/runtime/stage_factory.hpp"

#include "backends/metal/runtime/metal_executor.hpp"
#include "backends/metal/runtime/op_factory.hpp"
#include "runtime/execution_dispatcher.hpp"

namespace ov {
namespace gfx_plugin {

namespace {
// Anchor to keep MetalOpFactory symbols in the final shared library even though
// the MLIR-based MetalStage no longer uses the legacy per-op MetalOp path.
inline void keep_metal_op_factory_symbols() {
    using CreateFn = std::unique_ptr<MetalOp> (*)(const std::shared_ptr<const ov::Node>&, void*, void*);
    using CloneFn = std::unique_ptr<MetalOp> (*)(const MetalOp&);
    static volatile CreateFn keep_create = &MetalOpFactory::create;
    static volatile CloneFn keep_clone = &MetalOpFactory::clone;
    (void)keep_create;
    (void)keep_clone;
}
}  // namespace

std::unique_ptr<GpuStage> create_metal_stage(const std::shared_ptr<const ov::Node>& node,
                                             void* device,
                                             void* queue) {
    keep_metal_op_factory_symbols();
    return std::make_unique<MetalStage>(node, device, queue);
}

void ensure_metal_stage_factory_registered() {
    static const bool registered = GpuStageFactory::register_factory(GpuBackend::Metal, &create_metal_stage);
    (void)registered;
}

namespace {
const bool kRegistered = (ensure_metal_stage_factory_registered(), true);
}  // namespace

}  // namespace gfx_plugin
}  // namespace ov
