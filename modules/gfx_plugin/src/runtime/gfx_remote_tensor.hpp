// Remote tensor for GFX plugin
#pragma once

#include <string>
#include <vector>

#include "openvino/core/any.hpp"
#include "openvino/core/shape.hpp"
#include "openvino/core/except.hpp"
#include "openvino/runtime/iremote_tensor.hpp"
#include "runtime/gpu_tensor.hpp"

namespace ov {
namespace gfx_plugin {

using RemoteTensorReleaseFn = void (*)(GpuTensor&);

class GfxRemoteTensor : public ov::IRemoteTensor {
public:
    GfxRemoteTensor(const ov::element::Type& type,
                    const ov::Shape& shape,
                    const ov::AnyMap& params,
                    const std::string& dev,
                    const GpuTensor& tensor,
                    RemoteTensorReleaseFn release_fn);

    ~GfxRemoteTensor() override;

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
        const auto elem_type = m_type != ov::element::dynamic ? m_type : m_tensor.expected_type;
        if (elem_type != ov::element::dynamic && m_tensor.buf.size > 0) {
            const size_t required = shape_size(m_shape) * elem_type.size();
            OPENVINO_ASSERT(required <= m_tensor.buf.size,
                            "GFX: remote tensor shape exceeds buffer size");
        }
    }

    const GpuTensor& gpu_tensor() const { return m_tensor; }
    GpuTensor& gpu_tensor() { return m_tensor; }
    GpuBackend backend() const { return m_tensor.buf.backend; }
    bool owns_buffer() const { return m_tensor.buf.owned; }

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
    GpuTensor m_tensor{};
    RemoteTensorReleaseFn m_release_fn = nullptr;
};

struct RemoteTensorCreateResult {
    GpuTensor tensor{};
    ov::AnyMap properties{};
    RemoteTensorReleaseFn release_fn = nullptr;
};

}  // namespace gfx_plugin
}  // namespace ov
