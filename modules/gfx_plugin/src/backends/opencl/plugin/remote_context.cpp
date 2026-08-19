// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <memory>
#include <string>

#include "backends/opencl/plugin/remote_tensor.hpp"
#include "backends/opencl/runtime/opencl_api.hpp"
#include "compiler/backend_target.hpp"
#include "openvino/core/except.hpp"
#include "openvino/runtime/iremote_context.hpp"
#include "openvino/runtime/so_ptr.hpp"
#include "plugin/gfx_property_utils.hpp"
#include "runtime/gfx_remote_context.hpp"

namespace ov {
namespace gfx_plugin {
namespace {

class OpenClRemoteContext final : public GfxRemoteContext {
public:
    OpenClRemoteContext(const std::string& device,
                        const RemoteContextParams& params,
                        std::shared_ptr<OpenClRuntimeContext> runtime)
        : GfxRemoteContext(device,
                           params.device_id,
                           params.target,
                           reinterpret_cast<GpuDeviceHandle>(runtime->device()),
                           params.backend_name,
                           params.merged),
          m_runtime(std::move(runtime)) {
        OPENVINO_ASSERT(m_runtime, "GFX OpenCL: remote context runtime is null");
    }

private:
    RemoteTensorCreateResult create_remote_tensor(const ov::element::Type& type,
                                                  const ov::Shape& shape,
                                                  const ov::AnyMap& params,
                                                  size_t bytes) override {
        return create_opencl_remote_tensor(type, shape, params, m_runtime, bytes);
    }

    std::shared_ptr<OpenClRuntimeContext> m_runtime;
};

}  // namespace

ov::SoPtr<ov::IRemoteContext> create_opencl_remote_context(
    const std::string& resolved_name,
    const RemoteContextParams& params) {
    auto runtime = OpenClRuntimeContext::instance();
    const auto runtime_info = runtime->execution_device_info();
    const auto runtime_target = compiler::BackendTarget::from_backend_device_family(
        GpuBackend::OpenCL, runtime_info.device_family);
    OPENVINO_ASSERT(runtime_target.is_compatible_with_fingerprint(params.target.fingerprint()),
                    "GFX OpenCL: remote context target mismatch. Requested target: ",
                    params.target.debug_string(),
                    "; runtime target: ",
                    runtime_target.debug_string());

    auto ctx = std::make_shared<OpenClRemoteContext>(resolved_name, params, std::move(runtime));
    return ov::SoPtr<ov::IRemoteContext>{ctx, nullptr};
}

}  // namespace gfx_plugin
}  // namespace ov
