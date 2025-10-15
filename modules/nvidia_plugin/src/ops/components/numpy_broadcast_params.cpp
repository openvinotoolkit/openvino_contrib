// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "numpy_broadcast_params.h"

#include <cuda/runtime.hpp>

namespace ov {
namespace nvidia_gpu {

template <typename T>
static auto size_in_bytes(const std::vector<T>& v) noexcept {
    return sizeof(T) * v.size();
}

template <typename T>
static void uploadDataToWorkbuffer(CUDA::DevicePointer<void*> buffer, const std::vector<T>& v) {
    auto& stream = CUDA::DefaultStream::stream();
    stream.upload(buffer, v.data(), size_in_bytes(v));
}

std::unique_ptr<NumpyBroadcastParams> NumpyBroadcastParams::create(const ov::Shape& in_shape,
                                                                   const ov::Shape& out_shape) {
    if (in_shape == out_shape) {
        return std::make_unique<NumpyBroadcastParamsIdentity>();
    } else {
        return std::make_unique<NumpyBroadcastParamsImpl>(in_shape, out_shape);
    }
}

NumpyBroadcastParamsImpl::NumpyBroadcastParamsImpl(const ov::Shape& in_shape, const ov::Shape& out_shape)
    : shape_rank_{out_shape.size()}, dst_strides_{ov::row_major_strides(out_shape)} {
    ov::Shape broadcasted_shape{in_shape};
    OPENVINO_ASSERT(broadcasted_shape.size() <= shape_rank_);
    while (broadcasted_shape.size() < shape_rank_) {
        broadcasted_shape.insert(broadcasted_shape.begin(), 1);
    }
    OPENVINO_ASSERT(broadcasted_shape.size() == shape_rank_);

    src_strides_ = ov::row_major_strides(broadcasted_shape);

    for (size_t i = 0; i < shape_rank_; ++i) {
        auto in_dim = broadcasted_shape.at(i);
        broadcasted_dims_.emplace_back(in_dim == 1 ? 0 : 1);
        OPENVINO_ASSERT((in_dim == 1) || (in_dim == out_shape.at(i)));
    }
    OPENVINO_ASSERT(broadcasted_dims_.size() == shape_rank_);
}

void NumpyBroadcastParamsImpl::addWorkbufferRequests(
    std::vector<WorkbufferRequest::size_in_bytes_t>& immutable_buffer_sizes) {
    ib_src_strides_.addRequest(immutable_buffer_sizes, size_in_bytes(src_strides_));
    ib_dst_strides_.addRequest(immutable_buffer_sizes, size_in_bytes(dst_strides_));
    ib_broadcasted_dims_.addRequest(immutable_buffer_sizes, size_in_bytes(broadcasted_dims_));
}

void NumpyBroadcastParamsImpl::initWorkbuffers(const std::vector<CUDA::DevicePointer<void*>>& buffers) const {
    uploadDataToWorkbuffer(CUDA::DevicePointer<void*>{ib_src_strides_.requiredPtr(buffers)}, src_strides_);
    uploadDataToWorkbuffer(CUDA::DevicePointer<void*>{ib_dst_strides_.requiredPtr(buffers)}, dst_strides_);
    uploadDataToWorkbuffer(CUDA::DevicePointer<void*>{ib_broadcasted_dims_.requiredPtr(buffers)}, broadcasted_dims_);
}

kernel::NumpyBroadcastMapper NumpyBroadcastParamsImpl::mapper(
    const std::vector<CUDA::DevicePointer<const void*>>& immutable_buffers) const {
    return kernel::NumpyBroadcastMapper{static_cast<const size_t*>(ib_src_strides_.requiredPtr(immutable_buffers)),
                                        static_cast<const size_t*>(ib_dst_strides_.requiredPtr(immutable_buffers)),
                                        static_cast<const size_t*>(ib_broadcasted_dims_.requiredPtr(immutable_buffers)),
                                        shape_rank_};
}

}  // namespace nvidia_gpu
}  // namespace ov
