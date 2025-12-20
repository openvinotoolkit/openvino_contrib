// Remote context/tensor for METAL plugin
#pragma once

#include <memory>
#include <string>
#include <vector>

#include "openvino/core/shape.hpp"
#include "openvino/runtime/iremote_context.hpp"
#include "openvino/runtime/iremote_tensor.hpp"
#include "openvino/runtime/so_ptr.hpp"
#include "runtime/metal_memory.hpp"

namespace ov {
namespace metal_plugin {

class MetalRemoteTensor : public ov::IRemoteTensor {
public:
    MetalRemoteTensor(const ov::element::Type& type,
                      const ov::Shape& shape,
                      const ov::AnyMap& params,
                      const std::string& dev,
                      const MetalTensor& tensor,
                      bool owns_buffer);

    ~MetalRemoteTensor() override;

    const ov::element::Type& get_element_type() const override { return m_type; }
    const ov::Shape& get_shape() const override { return m_shape; }
    const Strides& get_strides() const override { return m_strides; }
    size_t get_size() const override { return shape_size(m_shape); }
    size_t get_byte_size() const override { return get_size() * m_type.size(); }
    const std::string& get_device_name() const override { return m_device; }
    const ov::AnyMap& get_properties() const override { return m_params; }
    void set_shape(ov::Shape shape) override {
        m_shape = std::move(shape);
        m_tensor.shape = m_shape;
        recalc_strides();
    }

    const MetalTensor& metal_tensor() const { return m_tensor; }
    MetalTensor& metal_tensor() { return m_tensor; }
    bool owns_buffer() const { return m_owns_buffer; }

private:
    void recalc_strides() {
        const auto elem_strides = ov::row_major_strides(m_shape);
        m_strides.resize(elem_strides.size());
        const size_t elem_size = m_type.size();
        for (size_t i = 0; i < elem_strides.size(); ++i) {
            m_strides[i] = elem_strides[i] * elem_size;
        }
    }

    ov::element::Type m_type;
    ov::Shape m_shape;
    Strides m_strides{};
    ov::AnyMap m_params;
    std::string m_device;
    MetalTensor m_tensor{};
    bool m_owns_buffer = false;
};

class MetalRemoteContext : public ov::IRemoteContext {
public:
    MetalRemoteContext(const std::string& device, int device_id, MetalDeviceHandle handle, const ov::AnyMap& params)
        : m_device(device), m_device_id(device_id), m_handle(handle), m_params(params) {}

    const std::string& get_device_name() const override { return m_device; }
    const ov::AnyMap& get_property() const override { return m_params; }
    int device_id() const { return m_device_id; }
    MetalDeviceHandle device_handle() const { return m_handle; }

    ov::SoPtr<ov::IRemoteTensor> create_tensor(const ov::element::Type& type,
                                               const ov::Shape& shape,
                                               const ov::AnyMap& params = {}) override;

private:
    std::string m_device;
    int m_device_id = 0;
    MetalDeviceHandle m_handle = nullptr;
    ov::AnyMap m_params;
};

}  // namespace metal_plugin
}  // namespace ov
