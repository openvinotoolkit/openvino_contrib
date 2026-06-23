// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "runtime/gfx_tensor_utils.hpp"

#include "openvino/core/except.hpp"
#include "openvino/core/type/bfloat16.hpp"
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

ov::Tensor convert_tensor_element_type(const ov::Tensor& src,
                                       const ov::element::Type& dst_type) {
    const auto src_type = src.get_element_type();
    if (src_type == dst_type) {
        return src;
    }

    OPENVINO_ASSERT(src && src.data(), "convert_tensor_element_type: source tensor is empty");
    const size_t count = src.get_size();
    ov::Tensor dst{dst_type, src.get_shape()};

    auto read_as_float = [&](size_t i) -> float {
        if (src_type == ov::element::f32) {
            return src.data<const float>()[i];
        }
        if (src_type == ov::element::f16) {
            return static_cast<float>(src.data<const ov::float16>()[i]);
        }
        if (src_type == ov::element::bf16) {
            return static_cast<float>(src.data<const ov::bfloat16>()[i]);
        }
        if (src_type == ov::element::u8) {
            return static_cast<float>(src.data<const uint8_t>()[i]);
        }
        if (src_type == ov::element::i8) {
            return static_cast<float>(src.data<const int8_t>()[i]);
        }
        if (src_type == ov::element::i32) {
            return static_cast<float>(src.data<const int32_t>()[i]);
        }
        if (src_type == ov::element::i64) {
            return static_cast<float>(src.data<const int64_t>()[i]);
        }
        OPENVINO_THROW("convert_tensor_element_type: unsupported source type ",
                       src_type.get_type_name());
    };

    if (dst_type == ov::element::f32) {
        auto* out = dst.data<float>();
        for (size_t i = 0; i < count; ++i) {
            out[i] = read_as_float(i);
        }
        return dst;
    }
    if (dst_type == ov::element::f16) {
        auto* out = dst.data<ov::float16>();
        for (size_t i = 0; i < count; ++i) {
            out[i] = ov::float16(read_as_float(i));
        }
        return dst;
    }
    if (dst_type == ov::element::bf16) {
        auto* out = dst.data<ov::bfloat16>();
        for (size_t i = 0; i < count; ++i) {
            out[i] = ov::bfloat16(read_as_float(i));
        }
        return dst;
    }

    OPENVINO_THROW("convert_tensor_element_type: unsupported destination type ",
                   dst_type.get_type_name());
}

}  // namespace gfx_plugin
}  // namespace ov
