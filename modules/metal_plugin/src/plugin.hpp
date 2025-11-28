// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//
#pragma once

#include <memory>
#include <string>
#include <vector>

#include "openvino/runtime/iplugin.hpp"
#include "openvino/runtime/properties.hpp"
#include "openvino/runtime/iremote_context.hpp"

namespace ov {
namespace metal_plugin {

class Plugin : public ov::IPlugin {
public:
    Plugin();
    ~Plugin() override = default;

    std::shared_ptr<ov::ICompiledModel> compile_model(const std::shared_ptr<const ov::Model>& model,
                                                      const ov::AnyMap& properties) const override;

    std::shared_ptr<ov::ICompiledModel> compile_model(const std::shared_ptr<const ov::Model>& model,
                                                      const ov::AnyMap& properties,
                                                      const ov::SoPtr<ov::IRemoteContext>& context) const override;

    std::shared_ptr<ov::ICompiledModel> import_model(std::istream& model,
                                                     const ov::AnyMap& properties) const override;

    std::shared_ptr<ov::ICompiledModel> import_model(std::istream& model,
                                                     const ov::SoPtr<ov::IRemoteContext>& context,
                                                     const ov::AnyMap& properties) const override;

    std::shared_ptr<ov::ICompiledModel> import_model(const ov::Tensor& model,
                                                     const ov::AnyMap& properties) const override;

    std::shared_ptr<ov::ICompiledModel> import_model(const ov::Tensor& model,
                                                     const ov::SoPtr<ov::IRemoteContext>& context,
                                                     const ov::AnyMap& properties) const override;

    ov::SupportedOpsMap query_model(const std::shared_ptr<const ov::Model>& model,
                                    const ov::AnyMap& properties) const override;

    void set_property(const ov::AnyMap& properties) override;

    ov::Any get_property(const std::string& name, const ov::AnyMap& arguments) const override;

    ov::SoPtr<ov::IRemoteContext> create_context(const ov::AnyMap& remote_properties) const override;

    ov::SoPtr<ov::IRemoteContext> get_default_context(const ov::AnyMap& remote_properties) const override;

private:
    ov::AnyMap m_config;
    // Stored value of ov::hint::PerformanceMode for the METAL backend; currently kept for API correctness,
    // may not yet affect actual scheduling/execution policy.
    ov::hint::PerformanceMode m_performance_mode = ov::hint::PerformanceMode::LATENCY;
};

}  // namespace metal_plugin
}  // namespace ov
