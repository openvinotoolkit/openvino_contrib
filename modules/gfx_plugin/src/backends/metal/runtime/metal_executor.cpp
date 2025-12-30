// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "backends/metal/runtime/metal_executor.hpp"

#include "backends/metal/runtime/op_factory.hpp"
#include "backends/metal/runtime/profiling/profiler.hpp"

namespace ov {
namespace gfx_plugin {

MetalStage::MetalStage(std::unique_ptr<MetalOp> op) : m_op(std::move(op)) {}

void MetalStage::init(GpuBufferManager* buffer_manager) {
    m_op->init(static_cast<MetalBufferManager*>(buffer_manager));
}

void MetalStage::compile(GpuBufferManager* buffer_manager) {
    m_op->compile(static_cast<MetalBufferManager*>(buffer_manager));
}

void MetalStage::execute(GpuCommandBufferHandle command_buffer) {
    m_op->execute(command_buffer);
}

void MetalStage::set_inputs(const std::vector<GpuTensor*>& inputs) {
    m_op->set_inputs(inputs);
}

void MetalStage::set_output(GpuTensor* output) {
    m_op->set_output(output);
}

void MetalStage::set_outputs(const std::vector<std::unique_ptr<GpuTensor>>& outputs) {
    m_op->set_outputs(outputs);
}

bool MetalStage::fuse_activation(ActivationKind kind, float alpha) {
    return m_op->fuse_activation(kind, alpha);
}

void MetalStage::enable_profiling(bool enable) {
    m_op->enable_profiling(enable);
}

void MetalStage::set_profiler(void* profiler,
                              uint32_t node_id,
                              const std::string& node_name,
                              const std::string& node_type) {
    m_op->set_profiler(static_cast<MetalProfiler*>(profiler), node_id, node_name, node_type);
}

const std::string& MetalStage::name() const {
    return m_op->name();
}

const std::string& MetalStage::type() const {
    return m_op->type();
}

std::unique_ptr<GpuStage> MetalStage::clone() const {
    auto cloned = MetalOpFactory::clone(*m_op);
    if (!cloned) {
        return nullptr;
    }
    return std::make_unique<MetalStage>(std::move(cloned));
}

}  // namespace gfx_plugin
}  // namespace ov
