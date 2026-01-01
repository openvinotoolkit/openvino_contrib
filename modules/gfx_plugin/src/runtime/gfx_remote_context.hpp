// Remote context for GFX plugin
#pragma once

#include <string>
#include <vector>

#include "openvino/runtime/iremote_context.hpp"
#include "openvino/runtime/iremote_tensor.hpp"
#include "openvino/runtime/so_ptr.hpp"
#include "runtime/gfx_remote_tensor.hpp"

namespace ov {
namespace gfx_plugin {

class GfxRemoteContext : public ov::IRemoteContext {
public:
    GfxRemoteContext(const std::string& device,
                     int device_id,
                     GpuBackend backend,
                     GpuDeviceHandle handle,
                     std::string backend_name,
                     const ov::AnyMap& params)
        : m_device(device),
          m_device_id(device_id),
          m_backend(backend),
          m_backend_name(std::move(backend_name)),
          m_handle(handle),
          m_params(params) {}

    const std::string& get_device_name() const override { return m_device; }
    const ov::AnyMap& get_property() const override { return m_params; }
    int device_id() const { return m_device_id; }
    GpuBackend backend() const { return m_backend; }
    const std::string& backend_name() const { return m_backend_name; }
    GpuDeviceHandle device_handle() const { return m_handle; }

    ov::SoPtr<ov::IRemoteTensor> create_tensor(const ov::element::Type& type,
                                               const ov::Shape& shape,
                                               const ov::AnyMap& params = {}) override;

protected:
    virtual RemoteTensorCreateResult create_remote_tensor(const ov::element::Type& type,
                                                          const ov::Shape& shape,
                                                          const ov::AnyMap& params,
                                                          size_t bytes) = 0;

private:
    std::string m_device;
    int m_device_id = 0;
    GpuBackend m_backend = GpuBackend::Metal;
    std::string m_backend_name;
    GpuDeviceHandle m_handle = nullptr;
    ov::AnyMap m_params;
};

}  // namespace gfx_plugin
}  // namespace ov
