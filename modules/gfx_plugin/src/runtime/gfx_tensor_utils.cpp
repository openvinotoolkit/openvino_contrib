// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "runtime/gfx_tensor_utils.hpp"

#include "openvino/core/except.hpp"
#include "openvino/core/type/float16.hpp"

namespace ov {
namespace gfx_plugin {

ov::Tensor to_float32_tensor(const ov::Tensor& src) {
    const auto et = src.get_element_type();
    if (et == ov::element::f32) {
        return src;
    }

    ov::Tensor dst{ov::element::f32, src.get_shape()};
    auto* dst_data = dst.data<float>();
    const size_t count = src.get_size();

    if (et == ov::element::f16) {
        const auto* src_data = src.data<ov::float16>();
        for (size_t i = 0; i < count; ++i) dst_data[i] = static_cast<float>(src_data[i]);
        return dst;
    }
    if (et == ov::element::u8) {
        const auto* src_data = src.data<uint8_t>();
        for (size_t i = 0; i < count; ++i) dst_data[i] = static_cast<float>(src_data[i]);
        return dst;
    }
    if (et == ov::element::i8) {
        const auto* src_data = src.data<int8_t>();
        for (size_t i = 0; i < count; ++i) dst_data[i] = static_cast<float>(src_data[i]);
        return dst;
    }
    if (et == ov::element::i32) {
        const auto* src_data = src.data<int32_t>();
        for (size_t i = 0; i < count; ++i) dst_data[i] = static_cast<float>(src_data[i]);
        return dst;
    }
    if (et == ov::element::i64) {
        const auto* src_data = src.data<int64_t>();
        for (size_t i = 0; i < count; ++i) dst_data[i] = static_cast<float>(src_data[i]);
        return dst;
    }

    OPENVINO_THROW("to_float32_tensor supports only f32/f16/u8/i8/i32/i64 inputs, got ",
                   et.get_type_name());
}

}  // namespace gfx_plugin
}  // namespace ov
