// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#ifndef __PP_EXTENSION_HPP__
#define __PP_EXTENSION_HPP__

#include <algorithm>
#include <initializer_list>

#include <openvino/core/node.hpp>
#include <openvino/runtime/tensor.hpp>

namespace PpExtension {

// Check an input that evaluators and kernels read as a flat buffer of a fixed element count.
inline void validate_input(const ov::Node* node, size_t port, std::initializer_list<ov::element::Type> types,
                           size_t elements) {
    const auto& type = node->get_input_element_type(port);
    const auto& shape = node->get_input_partial_shape(port);
    NODE_VALIDATION_CHECK(node, std::find(types.begin(), types.end(), type) != types.end(),
                          "Input ", port, " has unsupported element type ", type, ".");
    NODE_VALIDATION_CHECK(node, shape.is_static() && ov::shape_size(shape.to_shape()) == elements,
                          "Input ", port, " must have a static shape with ", elements, " elements, got ", shape, ".");
}

inline void check_input(const ov::Node* node, const ov::TensorVector& inputs, size_t port,
                        std::initializer_list<ov::element::Type> types, size_t elements) {
    const auto& type = inputs[port].get_element_type();
    OPENVINO_ASSERT(std::find(types.begin(), types.end(), type) != types.end() && inputs[port].get_size() == elements,
                    node->get_type_name(), " input ", port, " must hold ", elements, " elements of the validated type, got ",
                    type, " ", inputs[port].get_shape(), ".");
}

}  // namespace PpExtension

#endif  // __PP_EXTENSION_HPP__
