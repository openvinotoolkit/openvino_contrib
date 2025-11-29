// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//
#pragma once

#include <memory>

#include "openvino/runtime/icompiled_model.hpp"

#include "runtime/backend.hpp"
#include "runtime/mlir_backend.hpp"

namespace ov {
namespace metal_plugin {

class Plugin;
class InferRequest;

class CompiledModel : public ov::ICompiledModel {
public:
    CompiledModel(const std::shared_ptr<const ov::Model>& model,
                  const std::shared_ptr<const ov::IPlugin>& plugin);

    std::shared_ptr<const ov::Model> get_runtime_model() const override { return m_runtime_model; }
    void export_model(std::ostream& model) const override;

    void set_property(const ov::AnyMap& properties) override;
    ov::Any get_property(const std::string& name) const override;

    MetalBackend* backend() const { return m_backend.get(); }

protected:
    std::shared_ptr<ov::ISyncInferRequest> create_sync_infer_request() const override;

private:
    MetalBackendPtr m_backend;
    std::shared_ptr<const ov::Model> m_runtime_model;
    ov::AnyMap m_config;
};

}  // namespace metal_plugin
}  // namespace ov
