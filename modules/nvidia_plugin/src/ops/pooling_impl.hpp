// Copyright (C) 2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <cuda/dnn.hpp>
#include <ngraph/shape.hpp>
#include <openvino/op/avg_pool.hpp>
#include <openvino/op/max_pool.hpp>

namespace ov {
namespace nvidia_gpu {

class PoolingImpl {
public:
    explicit PoolingImpl(const ov::op::v1::MaxPool& node);

    explicit PoolingImpl(const ov::op::v1::AvgPool& node);

    ~PoolingImpl() = default;
    void Execute(const CUDA::DnnHandle& handle,
                 const void* input_tensor_device_ptr,
                 void* output_tensor_device_ptr) const;

private:
    int spatial_dims() const;
    std::vector<int> tensor_shape_from_ngraph(const ov::Shape& ngraph_shape) const;
    std::vector<int> spatial_shape_from_ngraph(const ov::Shape& ngraph_shape) const;
    std::vector<int> tensor_strides_from_ngraph(const ov::Shape& ngraph_strides) const;
    std::vector<int> paddings_from_ngraph(const ov::Shape& pads_begin,
                                          const ov::Shape& pads_end,
                                          cudnnPoolingMode_t pooling_mode) const;

public:
    static constexpr size_t input_index{0};
    static constexpr size_t output_index{0};

private:
    const int dims_;
    cudnnPoolingMode_t mode_;
    CUDA::DnnPoolingDescriptor pooling_descriptor_;
    CUDA::DnnTensorDescriptor input_tensor_descriptor_;
    CUDA::DnnTensorDescriptor output_tensor_descriptor_;
};

}  // namespace nvidia_gpu
}  // namespace ov
