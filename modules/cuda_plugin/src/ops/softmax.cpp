// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <gsl/gsl_assert>
#include <ngraph/node.hpp>
#include <cuda/dnn.hpp>
#include <cuda_operation_registry.hpp>
#include "softmax.hpp"

namespace CUDAPlugin {

static int take(const ngraph::Shape& shape, size_t i) noexcept {
  return i < shape.size() ? shape[i] : 1;
}

static constexpr long long prod(int a, int b) noexcept {
  return static_cast<long long>(a) * static_cast<long long>(b);
}

/** @brief Dimension Mapping Rationales
 *
 * 1. Problem statement
 *
 * 1.1. cudnnSoftmaxForward operates in terms of N,C,H,W, dimensions and mode of operation - channel or instance.
 * 1.2. OpenVINO defines Softmax operation in terms of shape and axis.
 * 1.3. cudnnSoftmaxForward supports 4D and 5D tensors while OpenVINO supports any tensor rank
 *
 * 2. Excerpts from cuDNN Documentation
 *
 * @ref https://docs.nvidia.com/deeplearning/cudnn/api/index.html#cudnnSoftmaxMode_t
 * 3.1.2.26. cudnnSoftmaxMode_t
 * cudnnSoftmaxMode_t is used to select over which data the cudnnSoftmaxForward() and cudnnSoftmaxBackward() are computing their results.
 * CUDNN_SOFTMAX_MODE_INSTANCE The softmax operation is computed per image (N) across the dimensions C,H,W.
 * CUDNN_SOFTMAX_MODE_CHANNEL  The softmax operation is computed per spatial location (H,W) per image (N) across the dimension C.
 *
 * @ref https://docs.nvidia.com/deeplearning/cudnn/api/index.html#cudnnSoftmaxForward
 * 3.2.98. cudnnSoftmaxForward()
 * All tensor formats are supported for all modes and algorithms with 4 and 5D tensors.
 * Performance is expected to be highest with NCHW fully-packed tensors.
 * For more than 5 dimensions tensors must be packed in their spatial dimensions
 *
 * 3. Design Decisions
 *
 * 3.1. In the first release support only tensors of ranks 1-5
 * 3.2. Insert extra dimensions into the shape to get 4D tensor for ranks 1-3
 * 3.3. Merge some dimensions of the shape to get 4D tensor for ranks 4 or 5
 * 3.4. Use channel mode of cudnnSoftmaxForward
 * 3.5. Assume default tensor format to be CUDNN_TENSOR_NCHW
 * 3.6. Map axis to dimension C
 *
 * 4. Rank/Axis Mapping of Shapes and Modes
 *
 *rank.axis,    CUDNN_TENSOR_NCHW
 *             N            C     H     W
 *   1.0  - {  1,          d0,    1,    1   },
 *   2.0  - {  1,          d0,   d1,    1   },
 *   2.1  - { d0,          d1,    1,    1   },
 *   3.0  - {  1,          d0,   d1,   d2   },
 *   3.1  - { d0,          d1,   d2,    1   },
 *   3.2  - {d0*d1,        d2,    1,    1   },
 *   4.0  - {  1,          d0,  d1*d2, d3   },
 *   4.1  - { d0,          d1,   d2,   d3   },
 *   4.2  - { d0*d1,       d2,   d3,    1   },
 *   4.3  - { d0*d1*d2,    d3,    1     1   },
 *   5.0  - {  1,          d0,  d1*d2,d3*d4 },
 *   5.1  - { d0,          d1,   d2,  d3*d4 },
 *   5.2  - { d0*d1,       d2,   d3,   d4   },
 *   5.3  - { d0*d1*d2,    d3,   d4,    1   },
 *   5.4  - { d0*d1*d2*d3, d4,    1,    1   },
 */
void SoftmaxOp::mapRankAxis(const ngraph::Shape& shape, int axis) {
  constexpr long long maxint = std::numeric_limits<int>::max();
  const auto rank = shape.size();
  Expects(rank <= 5 && rank > 0);
  Expects(axis < rank);

  const int d0 = shape[0];
  const int d1 = take(shape, 1);
  const int d2 = take(shape, 2);
  const int d3 = take(shape, 3);
  const int d4 = take(shape, 4);

  switch ((rank << 4) | axis) {
                      // N            C     H      W
  case 0x10: shape_ = {  1,          d0,    1,     1   }; break;
  case 0x20: shape_ = {  1,          d0,   d1,     1   }; break;
  case 0x21: shape_ = { d0,          d1,    1,     1   }; break;
  case 0x30: shape_ = {  1,          d0,   d1,    d2   }; break;
  case 0x31: shape_ = { d0,          d1,   d2,     1   }; break;
  case 0x32: shape_ = {d0*d1,        d2,    1,     1   }; Ensures(prod(d0, d1) < maxint); break;
  case 0x40: shape_ = {  1,          d0,   d1*d2, d3   }; Ensures(prod(d1, d2) < maxint); break;
  case 0x41: shape_ = { d0,          d1,   d2,    d3   }; break;
  case 0x42: shape_ = { d0*d1,       d2,   d3,     1   }; Ensures(prod(d0, d1) < maxint); break;
  case 0x43: shape_ = { d0*d1*d2,    d3,    1,     1   }; Ensures(prod(d0, d1)*d2 < maxint); break;
  case 0x50: shape_ = {  1,          d0,  d1*d2, d3*d4 }; Ensures((prod(d1, d2) < maxint) && (prod(d3, d4) < maxint)); break;
  case 0x51: shape_ = { d0,          d1,   d2,   d3*d4 }; Ensures(prod(d3, d4) < maxint); break;
  case 0x52: shape_ = { d0*d1,       d2,   d3,    d4   }; Ensures(prod(d0, d1) < maxint); break;
  case 0x53: shape_ = { d0*d1*d2,    d3,   d4,     1   }; Ensures(prod(d0, d1)*d2 < maxint); break;
  case 0x54: shape_ = { d0*d1*d2*d3, d4,    1,     1   }; Ensures(prod(d0, d1)*d2*d3 < maxint); break;
  default:
    THROW_IE_EXCEPTION << "Unsupported combination of tensor rank (" << rank << ") and axis attribute (" << axis << ")";
  }
}

/**
 * @brief Converts OpenVINO data type to cuDNN data type
 */
constexpr cudnnDataType_t convertDataType(const ngraph::element::Type& type) {
  using ngraph::element::Type_t;
  switch (static_cast<Type_t>(type)) {
    case Type_t::bf16:
      return CUDNN_DATA_BFLOAT16;
    case Type_t::f16:
      return CUDNN_DATA_HALF;
    case Type_t::f32:
      return CUDNN_DATA_FLOAT;
    case Type_t::f64:
      return CUDNN_DATA_DOUBLE;
    case Type_t::i8:
      return CUDNN_DATA_INT8;
    case Type_t::i32:
      return CUDNN_DATA_INT32;
    case Type_t::i64:
      return CUDNN_DATA_INT64;
    default:
      THROW_IE_EXCEPTION << "Unsupported ngraph element type "  << type.c_type_string();
  }
}

SoftmaxOp::SoftmaxOp(const NodeOp& node,
                     IndexCollection&& inputIds,
                     IndexCollection&& outputIds)
  : OperationBase{node, move(inputIds), move(outputIds) },
    type_ {convertDataType(node.input(0).get_element_type())} {
      const int axis = node.get_axis();
      mapRankAxis(node.input(0).get_shape(), axis);
      tensor_descriptor_.set(type_, cudnnTensorFormat_t::CUDNN_TENSOR_NCHW, shape_.data());
  }

void SoftmaxOp::Execute(const InferenceRequestContext& context, Inputs inputs, Outputs outputs) {
  Expects(inputs.size() == 1);
  Expects(outputs.size() == 1);
  CUDA::throwIfError(cudnnSoftmaxForward(
      context.getThreadContext().dnnHandle().get(),
      cudnnSoftmaxAlgorithm_t::CUDNN_SOFTMAX_ACCURATE,
      cudnnSoftmaxMode_t::CUDNN_SOFTMAX_MODE_CHANNEL,
      &CUDA::NoScaling.alpha,
      tensor_descriptor_.get(),
      inputs[0].get(),
      &CUDA::NoScaling.beta,
      tensor_descriptor_.get(),
      outputs[0].get()));
}

OPERATION_REGISTER(SoftmaxOp, Softmax);
} // namespace CUDAPlugin
