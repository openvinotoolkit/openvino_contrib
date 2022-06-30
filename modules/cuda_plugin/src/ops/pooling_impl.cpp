// Copyright (C) 2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "pooling_impl.hpp"

#include <fmt/format.h>

#include <cuda_operation_registry.hpp>
#include <gsl/gsl_assert>
#include <ngraph/type/element_type.hpp>

#include "converters.hpp"
#include "cuda/constant_factory.hpp"

namespace CUDAPlugin {

static constexpr size_t non_spatial_dims_{2};
static constexpr size_t min_spatial_dims_{2};
static constexpr size_t max_spatial_dims_{3};
static constexpr size_t min_total_dims_{min_spatial_dims_ + non_spatial_dims_};
static constexpr size_t max_total_dims_{max_spatial_dims_ + non_spatial_dims_};
static constexpr size_t default_stride_{1};

// for 1d shape, the dimension is extended to 2d
// For the rest, 2d and 3d, the dimension correspondent tensor shape.
static int pooling_extend_dimension(size_t shape_size) {
    auto ret = shape_size > min_total_dims_ ? shape_size : min_total_dims_;
    return static_cast<int>(ret);
}

PoolingImpl::PoolingImpl(const ov::op::v1::MaxPool& node)
    : dims_{pooling_extend_dimension(node.get_input_shape(input_index).size())},
      mode_{CUDNN_POOLING_MAX},
      pooling_descriptor_{},
      input_tensor_descriptor_{},
      output_tensor_descriptor_{} {
    pooling_descriptor_.set(mode_,
                            CUDNN_PROPAGATE_NAN,
                            spatial_dims(),
                            spatial_shape_from_ngraph(node.get_kernel()).data(),
                            paddings_from_ngraph(node.get_pads_begin(), node.get_pads_end(), mode_).data(),
                            spatial_shape_from_ngraph(node.get_strides()).data());

    input_tensor_descriptor_.set(convertDataType<cudnnDataType_t>(node.get_element_type()),
                                 dims_,
                                 tensor_shape_from_ngraph(node.get_input_shape(input_index)).data(),
                                 tensor_strides_from_ngraph(node.get_input_shape(input_index)).data());

    output_tensor_descriptor_.set(convertDataType<cudnnDataType_t>(node.get_element_type()),
                                  dims_,
                                  tensor_shape_from_ngraph(node.get_output_shape(output_index)).data(),
                                  tensor_strides_from_ngraph(node.get_output_shape(output_index)).data());

    Expects(node.get_input_shape(input_index).size() == node.get_output_shape(output_index).size());
}

PoolingImpl::PoolingImpl(const ov::op::v1::AvgPool& node)
    : dims_{pooling_extend_dimension(node.get_input_shape(input_index).size())},
      mode_(node.get_exclude_pad() ? CUDNN_POOLING_AVERAGE_COUNT_EXCLUDE_PADDING
                                   : CUDNN_POOLING_AVERAGE_COUNT_INCLUDE_PADDING),
      pooling_descriptor_{},
      input_tensor_descriptor_{},
      output_tensor_descriptor_{} {
    pooling_descriptor_.set(mode_,
                            CUDNN_PROPAGATE_NAN,
                            spatial_dims(),
                            spatial_shape_from_ngraph(node.get_kernel()).data(),
                            paddings_from_ngraph(node.get_pads_begin(), node.get_pads_end(), mode_).data(),
                            spatial_shape_from_ngraph(node.get_strides()).data());
    input_tensor_descriptor_.set(convertDataType<cudnnDataType_t>(node.get_element_type()),
                                 dims_,
                                 tensor_shape_from_ngraph(node.get_input_shape(input_index)).data(),
                                 tensor_strides_from_ngraph(node.get_input_shape(input_index)).data());
    output_tensor_descriptor_.set(convertDataType<cudnnDataType_t>(node.get_element_type()),
                                  dims_,
                                  tensor_shape_from_ngraph(node.get_output_shape(output_index)).data(),
                                  tensor_strides_from_ngraph(node.get_output_shape(output_index)).data());
    Expects(node.get_input_shape(input_index).size() == node.get_output_shape(output_index).size());
}

void PoolingImpl::Execute(const CUDA::DnnHandle& cudnn_context_handle,
                          const void* input_tensor_device_ptr,
                          void* output_tensor_device_ptr) const {
    throwIfError(cudnnPoolingForward(cudnn_context_handle.get(),       //
                                     pooling_descriptor_.get(),        //
                                     &CUDA::constants::one<float>::value,    //
                                     input_tensor_descriptor_.get(),   //
                                     input_tensor_device_ptr,          //
                                     &CUDA::constants::zero<float>::value,   //
                                     output_tensor_descriptor_.get(),  //
                                     output_tensor_device_ptr));
}

std::vector<int> PoolingImpl::tensor_shape_from_ngraph(const ov::Shape& ngraph_shape) const {
    Expects(pooling_extend_dimension(ngraph_shape.size()) >= min_total_dims_ &&
            pooling_extend_dimension(ngraph_shape.size()) <= max_total_dims_);
    std::vector<int> shape(dims_, 1);
    std::copy(ngraph_shape.rbegin(), ngraph_shape.rend(), shape.rbegin());
    return shape;
}

std::vector<int> PoolingImpl::spatial_shape_from_ngraph(const ov::Shape& ngraph_shape) const {
    Expects(ngraph_shape.size() <= max_spatial_dims_);
    Expects(spatial_dims() >= min_spatial_dims_ && spatial_dims() <= max_spatial_dims_);
    std::vector<int> shape(spatial_dims(), 1);
    std::copy(ngraph_shape.rbegin(), ngraph_shape.rend(), shape.rbegin());
    return shape;
}

std::vector<int> PoolingImpl::tensor_strides_from_ngraph(const ov::Shape& ngraph_shape) const {
    std::vector<int> strides(dims_, default_stride_);
    auto in_strides = ov::row_major_strides(ngraph_shape);
    std::copy(in_strides.rbegin(), in_strides.rend(), strides.rbegin());
    return strides;
}

std::vector<int> PoolingImpl::paddings_from_ngraph(
    const ov::Shape& pads_begin, const ov::Shape& pads_end,
    cudnnPoolingMode_t pooling_mode) const {
  // Input tensor rank means:
  // 3 dims == 1D pooling, 4 dims == 2D pooling, 5 dims == 3D pooling
  Expects(pads_begin.size() == pads_end.size());
  Expects(pads_begin.size() <= max_spatial_dims_);

  auto validate_padding_symmetry = [](size_t axis, size_t begin, size_t end) {
    if (begin == end) {
      return;
    }
    throwIEException(
        fmt::format("Error: cuDNN pooling ops support only symmetric padding "
                    "(begin==end), while given: begin {}"
                    ", end {} for spatial axis {}",
                    begin, end, axis));
  };

  if (pooling_mode == CUDNN_POOLING_AVERAGE_COUNT_INCLUDE_PADDING) {
    for (size_t axis{0}; axis < pads_begin.size(); axis++) {
      validate_padding_symmetry(axis, pads_begin.at(axis), pads_end.at(axis));
    }
  }

  // As the IE opset BasePooling strides, paddings and kernel dimensions go in
  // the [depth, height, width] [height, width] or [width] orders, the
  // corresponding member arrays are being filled tail-to-head.
  std::vector<int> paddings(spatial_dims(), 0);
  std::copy(pads_begin.rbegin(), pads_begin.rend(), paddings.rbegin());

  return paddings;
}

int PoolingImpl::spatial_dims() const { return dims_ - non_spatial_dims_; }

}  // namespace CUDAPlugin
