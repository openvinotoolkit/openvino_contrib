// Minimal remote context/tensor stub for METAL plugin behavior tests
#pragma once
#include "openvino/runtime/iremote_context.hpp"
#include "openvino/runtime/iremote_tensor.hpp"
#include "openvino/runtime/so_ptr.hpp"
#include "openvino/core/shape.hpp"
#include <vector>

namespace ov {
namespace metal_plugin {

class MetalRemoteTensor : public ov::IRemoteTensor {
public:
    MetalRemoteTensor(const ov::element::Type& type, const ov::Shape& shape, const ov::AnyMap& params, const std::string& dev)
        : m_type(type), m_shape(shape), m_params(params), m_device(dev) {
        recalc_strides();
    }

    const ov::element::Type& get_element_type() const override { return m_type; }
    const ov::Shape& get_shape() const override { return m_shape; }
    const Strides& get_strides() const override { return m_strides; }
    size_t get_size() const override { return shape_size(m_shape); }
    size_t get_byte_size() const override { return get_size() * m_type.size(); }
    const std::string& get_device_name() const override { return m_device; }
    const ov::AnyMap& get_properties() const override { return m_params; }
    void set_shape(ov::Shape shape) override {
        m_shape = std::move(shape);
        recalc_strides();
    }

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
};

class MetalRemoteContext : public ov::IRemoteContext {
public:
    explicit MetalRemoteContext(const std::string& device, const ov::AnyMap& params) : m_device(device), m_params(params) {}

    const std::string& get_device_name() const override { return m_device; }
    const ov::AnyMap& get_property() const override { return m_params; }

    ov::SoPtr<ov::IRemoteTensor> create_tensor(const ov::element::Type& type, const ov::Shape& shape, const ov::AnyMap& params = {}) override {
        ov::AnyMap merged = m_params;
        merged.insert(params.begin(), params.end());
        auto t = std::make_shared<MetalRemoteTensor>(type, shape, merged, m_device);
        return ov::SoPtr<ov::IRemoteTensor>{t, nullptr};
    }

private:
    std::string m_device;
    ov::AnyMap m_params;
};

}  // namespace metal_plugin
}  // namespace ov
