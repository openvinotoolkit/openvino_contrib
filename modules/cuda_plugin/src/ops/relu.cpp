// Copyright (C) 2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//
#include <cuda_operation_base.hpp>
#include <cuda_operation_registry.hpp>
#include <ngraph/node.hpp>

namespace CUDAPlugin {
namespace {
constexpr cudnnDataType_t toDataType(const ngraph::element::Type& type) {
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
      CUDA::throwIEException("unsupported ngraph element type " +
                             type.c_type_string());
  }
}

std::array<int, 5> toArray(const ngraph::Shape& shape) {
  std::array<int, 5> a{1, 1, 1, 1, 1};
  if (shape.empty()) return a;
  for (std::size_t i = std::min(shape.size(), a.size()); i > 0;) {
    i--;
    a[i] = shape[i];
  }
  return a;
}
CUDA::DnnTensorDescriptor desc(
    const ngraph::element::Type& type,
    const ngraph::Shape&
        shape) {  // TODO: different ops have different shape limitations
  auto dims = toArray(shape);
  decltype(dims) strides;
  strides.back() = 1;
  for (int i = dims.size() - 1; i > 0; i--)
    strides[i - 1] = strides[i] * dims[i];
  return {toDataType(type), dims.size(), dims.data(), strides.data()};
}

CUDA::DnnTensorDescriptor inputDesc(const ngraph::Node& node, int n) {
  return desc(node.get_input_element_type(n), node.get_input_shape(n));
}

CUDA::DnnTensorDescriptor outputDesc(const ngraph::Node& node, int n) {
  return desc(node.get_output_element_type(n), node.get_output_shape(n));
}

class Relu : public OperationBase {
  CUDA::ReluDescriptor reluDesc;
  CUDA::DnnTensorDescriptor xDesc;
  CUDA::DnnTensorDescriptor yDesc;
  static inline float one = 1;
  static inline float zero = 0;

 public:
  Relu(const std::shared_ptr<ngraph::Node>& node,
       std::vector<unsigned> inputIds, std::vector<unsigned> outputIds)
      : OperationBase{node, move(inputIds), move(outputIds)},
        xDesc{inputDesc(*node, 0)},
        yDesc{outputDesc(*node, 0)} {}
  void Execute(const InferenceRequestContext& context, Inputs inputTensors,
               Outputs outputTensors) override {
    context.getThreadContext().dnnHandle().activationForward(
        reluDesc, &one, xDesc, inputTensors[0].get(), &zero, yDesc,
        outputTensors[0].get());
  }
};

OperationRegistry::Register<Relu> reg{"Relu"};

}  // namespace
}  // namespace CUDAPlugin
