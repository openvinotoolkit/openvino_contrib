// Copyright (C) 2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "converters.hpp"
#include "pooling_impl.hpp"
#include "constant_factory.hpp"

#include <cuda_operation_registry.hpp>
#include <gsl/gsl_assert>
#include <ngraph/type/element_type.hpp>

namespace CUDAPlugin {

static constexpr size_t non_spatial_dims_{2};
static constexpr size_t min_spatial_dims_{1};
static constexpr size_t max_spatial_dims_{3};
static constexpr size_t min_total_dims_{min_spatial_dims_ + non_spatial_dims_};
static constexpr size_t max_total_dims_{max_spatial_dims_ + non_spatial_dims_};
static constexpr size_t default_stride_{1};

PoolingImpl::PoolingImpl(const ngraph::op::v1::MaxPool& node)
    : mode_{CUDNN_POOLING_MAX},
      pooling_descriptor_{mode_,
                          CUDNN_PROPAGATE_NAN,
                          max_spatial_dims_,
                          spatial_shape_from_ngraph(node.get_kernel()).data(),
                          paddings_from_ngraph(node.get_pads_begin(),
                                               node.get_pads_end(), mode_)
                              .data(),
                          spatial_shape_from_ngraph(node.get_strides()).data()},
      input_tensor_descriptor_{
          convertDataType<cudnnDataType_t>(node.get_element_type()), max_total_dims_,
          tensor_shape_from_ngraph(node.get_input_shape(input_index)).data(),
          tensor_strides_from_ngraph(node.get_input_shape(input_index)).data()},
      output_tensor_descriptor_{
          convertDataType<cudnnDataType_t>(node.get_element_type()), max_total_dims_,
          tensor_shape_from_ngraph(node.get_output_shape(output_index)).data(),
          tensor_strides_from_ngraph(node.get_output_shape(output_index))
              .data()} {
  Expects(node.get_input_shape(input_index).size() ==
          node.get_output_shape(output_index).size());
}

PoolingImpl::PoolingImpl(const ngraph::op::AvgPool& node)
    : mode_(node.get_exclude_pad()
                ? CUDNN_POOLING_AVERAGE_COUNT_EXCLUDE_PADDING
                : CUDNN_POOLING_AVERAGE_COUNT_INCLUDE_PADDING),
      pooling_descriptor_{mode_,
                          CUDNN_PROPAGATE_NAN,
                          max_spatial_dims_,
                          spatial_shape_from_ngraph(node.get_kernel()).data(),
                          paddings_from_ngraph(node.get_pads_begin(),
                                               node.get_pads_end(), mode_)
                              .data(),
                          spatial_shape_from_ngraph(node.get_strides()).data()},
      input_tensor_descriptor_{
          convertDataType<cudnnDataType_t>(node.get_element_type()), max_total_dims_,
          tensor_shape_from_ngraph(node.get_input_shape(input_index)).data(),
          tensor_strides_from_ngraph(node.get_input_shape(input_index)).data()},
      output_tensor_descriptor_{
          convertDataType<cudnnDataType_t>(node.get_element_type()), max_total_dims_,
          tensor_shape_from_ngraph(node.get_output_shape(output_index)).data(),
          tensor_strides_from_ngraph(node.get_output_shape(output_index))
              .data()} {
  Expects(node.get_input_shape(input_index).size() ==
          node.get_output_shape(output_index).size());
}

void PoolingImpl::Execute(const CUDA::DnnHandle& cudnn_context_handle,
                            const void* input_tensor_device_ptr,
                            void* output_tensor_device_ptr) {
  CUDA::throwIfError(cudnnPoolingForward(cudnn_context_handle.get(),       //
                                         pooling_descriptor_.get(),        //
                                         &constants::one<float>::value,    //
                                         input_tensor_descriptor_.get(),   //
                                         input_tensor_device_ptr,          //
                                         &constants::zero<float>::value,   //
                                         output_tensor_descriptor_.get(),  //
                                         output_tensor_device_ptr));
}

std::vector<int> PoolingImpl::tensor_shape_from_ngraph(
    const ngraph::Shape& ngraph_shape) {
  Expects(ngraph_shape.size() >= min_total_dims_ &&
          ngraph_shape.size() <= max_total_dims_);
  std::vector<int> shape(max_total_dims_, 1);
  std::copy(ngraph_shape.rbegin(), ngraph_shape.rend(), shape.rbegin());
  return shape;
}

std::vector<int> PoolingImpl::spatial_shape_from_ngraph(
    const ngraph::Shape& ngraph_shape) {
  Expects(ngraph_shape.size() >= min_spatial_dims_ &&
          ngraph_shape.size() <= max_spatial_dims_);
  std::vector<int> shape(max_spatial_dims_, 1);
  std::copy(ngraph_shape.rbegin(), ngraph_shape.rend(), shape.rbegin());
  return shape;
}

std::vector<int> PoolingImpl::tensor_strides_from_ngraph(
    const ngraph::Shape& ngraph_shape) {
  std::vector<int> strides(max_total_dims_, default_stride_);
  auto in_strides = ngraph::row_major_strides(ngraph_shape);
  std::copy(in_strides.rbegin(), in_strides.rend(), strides.rbegin());
  return strides;
}

std::vector<int> PoolingImpl::paddings_from_ngraph(
    const ngraph::Shape& pads_begin, const ngraph::Shape& pads_end,
    cudnnPoolingMode_t pooling_mode) {
  // Input tensor rank means:
  // 3 dims == 1D pooling, 4 dims == 2D pooling, 5 dims == 3D pooling
  Expects(pads_begin.size() == pads_end.size());
  Expects(pads_begin.size() >= min_spatial_dims_ &&
          pads_begin.size() <= max_spatial_dims_);

  auto validate_padding_symmetry = [](size_t axis, size_t begin, size_t end) {
    if (begin == end) {
      return;
    }
    THROW_IE_EXCEPTION
        << "Error: cuDNN pooling ops support only symmetric padding "
           "(begin==end), while given: begin "
        << begin << ", end " << end << " for spatial axis " << axis;
  };

  if (pooling_mode == CUDNN_POOLING_AVERAGE_COUNT_INCLUDE_PADDING) {
    for (size_t axis{0}; axis < pads_begin.size(); axis++) {
      validate_padding_symmetry(axis, pads_begin.at(axis), pads_end.at(axis));
    }
  }

  // As the IE opset BasePooling strides, paddings and kernel dimensions go in
  // the [depth, height, width] [height, width] or [width] orders, the
  // corresponding member arrays are being filled tail-to-head.
  std::vector<int> paddings(max_spatial_dims_, 0);
  std::copy(pads_begin.rbegin(), pads_begin.rend(), paddings.rbegin());

  return paddings;
}

}  // namespace CUDAPlugin
