// Copyright (C) 2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <cuda/dnn.hpp>
#include <ngraph/op/avg_pool.hpp>
#include <ngraph/op/max_pool.hpp>
#include <ngraph/shape.hpp>

namespace CUDAPlugin {

class PoolingImpl {
 public:
  explicit PoolingImpl(const ngraph::op::v1::MaxPool& node);

  explicit PoolingImpl(const ngraph::op::AvgPool& node);

  ~PoolingImpl() = default;

  void Execute(const CUDA::DnnHandle& handle,
               const void* input_tensor_device_ptr,
               void* output_tensor_device_ptr);

  static constexpr size_t input_index{0};
  static constexpr size_t output_index{0};

 private:
  cudnnPoolingMode_t mode_;
  CUDA::DnnPoolingDescriptor pooling_descriptor_;
  CUDA::DnnTensorDescriptor input_tensor_descriptor_;
  CUDA::DnnTensorDescriptor output_tensor_descriptor_;

  static std::vector<int> tensor_shape_from_ngraph(
      const ngraph::Shape& ngraph_shape);
  static std::vector<int> spatial_shape_from_ngraph(
      const ngraph::Shape& ngraph_shape);
  static std::vector<int> tensor_strides_from_ngraph(
      const ngraph::Shape& ngraph_strides);
  static std::vector<int> paddings_from_ngraph(const ngraph::Shape& pads_begin,
                                               const ngraph::Shape& pads_end,
                                               cudnnPoolingMode_t pooling_mode);
};

}  // namespace CUDAPlugin
