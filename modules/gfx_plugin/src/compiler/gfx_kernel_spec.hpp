// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//
#pragma once

#include <memory>
#include <string>

#include "openvino/core/node.hpp"
#include "openvino/core/shape.hpp"

namespace ov {
namespace gfx_plugin {

class KernelSpec {
public:
    KernelSpec(std::shared_ptr<const ov::Node> node, uint32_t arg_count)
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

}  // namespace gfx_plugin
}  // namespace ov
