// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//
#pragma once

#include <memory>
#include <string>
#include <string_view>

#include "kernel_ir/gfx_custom_kernel_families.hpp"
#include "openvino/core/node.hpp"
#include "openvino/core/shape.hpp"

namespace ov {
namespace gfx_plugin {

class KernelSpec {
public:
    KernelSpec(std::shared_ptr<const ov::Node> node, uint32_t arg_count = 0)
        : m_node(std::move(node)),
          m_arg_count(arg_count) {
        if (m_node) {
            m_name = m_node->get_friendly_name();
            m_type = m_node->get_type_name();
            if (m_node->get_output_partial_shape(0).is_static()) {
                m_output_shape = m_node->get_output_shape(0);
            }
        }
    }

    const std::shared_ptr<const ov::Node>& node() const { return m_node; }
    uint32_t arg_count() const { return m_arg_count; }
    const std::string& name() const { return m_name; }
    const std::string& type() const { return m_type; }
    const ov::Shape& output_shape() const { return m_output_shape; }

private:
    std::shared_ptr<const ov::Node> m_node;
    uint32_t m_arg_count = 0;
    std::string m_name;
    std::string m_type;
    ov::Shape m_output_shape;
};

inline uint32_t infer_kernel_spec_arg_count_from_custom_abi(std::string_view stage_type,
                                                            std::string_view entry_point) {
    const auto family = classify_gfx_custom_kernel_family(stage_type, entry_point);
    const auto abi = gfx_kernel_external_buffer_abi_spec_for_stage(stage_type, entry_point, family);
    if (abi.valid && !abi.roles.empty()) {
        return static_cast<uint32_t>(abi.roles.size());
    }
    return 0;
}

inline KernelSpec make_kernel_spec_from_custom_kernel_abi(const std::shared_ptr<const ov::Node>& node,
                                                          std::string_view entry_point) {
    const std::string_view stage_type = node ? std::string_view(node->get_type_name()) : std::string_view{};
    return KernelSpec(node, infer_kernel_spec_arg_count_from_custom_abi(stage_type, entry_point));
}

}  // namespace gfx_plugin
}  // namespace ov
