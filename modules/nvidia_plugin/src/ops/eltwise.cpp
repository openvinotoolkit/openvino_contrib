// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <cuda/runtime.hpp>
#include <gsl/gsl_util>
#include <kernels/details/eltwise.cuh>
#include <string>

using namespace std::string_literals;

namespace ov {
namespace nvidia_gpu {

namespace eltwise {

namespace {
template <typename T>
unsigned majorGridDim(T majorShapeDim, unsigned elementsPerBlock) {
    majorShapeDim = gsl::narrow_cast<unsigned>(majorShapeDim);
    return majorShapeDim / elementsPerBlock + (majorShapeDim % elementsPerBlock ? 1u : 0u);
}
}  // anonymous namespace

KernelExecAttrs::KernelExecAttrs(const ov::Shape& shape, unsigned threadsPerBlock, unsigned elementsPerThread)
    : grid{[&shape, threadsPerBlock, elementsPerThread]() {
          switch (shape.size()) {
              case 1:
                  return dim3{majorGridDim(shape[0], threadsPerBlock * elementsPerThread)};
              case 2:
                  return dim3{gsl::narrow_cast<unsigned>(shape[0]),
                              majorGridDim(shape[1], threadsPerBlock * elementsPerThread)};
              case 3:
                  return dim3{gsl::narrow_cast<unsigned>(shape[0]),
                              gsl::narrow_cast<unsigned>(shape[1]),
                              majorGridDim(shape[2], threadsPerBlock * elementsPerThread)};
              case 4:
                  return dim3{gsl::narrow_cast<unsigned>(shape[0]) * gsl::narrow_cast<unsigned>(shape[1]),
                              gsl::narrow_cast<unsigned>(shape[2]),
                              majorGridDim(shape[3], threadsPerBlock * elementsPerThread)};
              case 5:
                  return dim3{gsl::narrow_cast<unsigned>(shape[0]) * gsl::narrow_cast<unsigned>(shape[1]),
                              gsl::narrow_cast<unsigned>(shape[2]) * gsl::narrow_cast<unsigned>(shape[3]),
                              majorGridDim(shape[4], threadsPerBlock * elementsPerThread)};
              default:
                  throw_ov_exception("Shape with rank "s + std::to_string(shape.size()) + " is not supported"s);
          }
      }()},
      block{threadsPerBlock},
      elementsPerThread{elementsPerThread} {}

}  // namespace eltwise
}  // namespace nvidia_gpu
}  // namespace ov
