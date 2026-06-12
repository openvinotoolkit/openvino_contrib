// Copyright (c) Facebook, Inc. and its affiliates.
// 
// This source code is licensed under the MIT license found in the
// LICENSE file in the root directory of this source tree.

#include "group_points.h"
#include "utils.h"

// CUDA kernel wrapper declarations - only available when CUDA is available
#if CUDA_AVAILABLE
void group_points_kernel_wrapper(int b, int c, int n, int npoints, int nsample,
                                 const float *points, const int *idx,
                                 float *out);

void group_points_grad_kernel_wrapper(int b, int c, int n, int npoints,
                                      int nsample, const float *grad_out,
                                      const int *idx, float *grad_points);
#endif

void group_points_kernel_cpu_wrapper(int b, int c, int n, int npoints, int nsample,
                                 const float *points, const int *idx,
                                 float *out){
    // std::cout << "========= group_points_kernel_cpu_wrapper =======" << std::endl;
    for (int batch_index = 0; batch_index < b; ++batch_index) {
      // Calculate offset for current batch
      const float *current_points = points + batch_index * n * c;
      const int *current_idx = idx + batch_index * npoints * nsample;
      float *current_out = out + batch_index * npoints * nsample * c;

      // Iterate through each channel c and each sampling point npoints
      for (int l = 0; l < c; ++l) { // Iterate through each channel
        for (int j = 0; j < npoints; ++j) { // Iterate through each sampling point
          for (int k = 0; k < nsample; ++k) { // For each sample point's nsample neighbors
            int ii = current_idx[j * nsample + k]; // Get corresponding original point index
            if(ii >= 0 && ii < n) { // Ensure index is valid
              current_out[(l * npoints + j) * nsample + k] = current_points[l * n + ii];
            } else {
              // If index is invalid, can set a default value or throw exception, etc.
              current_out[(l * npoints + j) * nsample + k] = 0.0f; // Simply set to 0.0 here
            }
          }
        }
      }
    }
  }

void group_points_grad_kernel_cpu_wrapper(int b, int c, int n, int npoints,
                                      int nsample, const float *grad_out,
                                      const int *idx, float *grad_points){
    // Iterate through each batch
    for (int batch_index = 0; batch_index < b; ++batch_index) {
      // Calculate offset for current batch
      const float *current_grad_out = grad_out + batch_index * npoints * nsample * c;
      const int *current_idx = idx + batch_index * npoints * nsample;
      float *current_grad_points = grad_points + batch_index * n * c;

      // Initialize gradient point array to 0 to ensure no errors when accumulating repeatedly
      for (int i = 0; i < n * c; ++i) {
        current_grad_points[i] = 0.0f;
      }

      // Iterate through each channel c and each sampling point npoints
      for (int l = 0; l < c; ++l) { // Iterate through each channel
        for (int j = 0; j < npoints; ++j) { // Iterate through each sampling point
          for (int k = 0; k < nsample; ++k) { // For each sample point's nsample neighbors
            int ii = current_idx[j * nsample + k]; // Get corresponding original point index
            if(ii >= 0 && ii < n) { // Ensure index is valid
              // Accumulate gradient value to corresponding grad_points position
              current_grad_points[l * n + ii] += current_grad_out[(l * npoints + j) * nsample + k];
            }
            // If index is invalid, ignore this gradient contribution
          }
        }
      }
    }
  }


at::Tensor group_points(at::Tensor points, at::Tensor idx) {
  CHECK_CONTIGUOUS(points);
  CHECK_CONTIGUOUS(idx);
  CHECK_IS_FLOAT(points);
  CHECK_IS_INT(idx);

  if (points.type().is_cuda()) {
    CHECK_CUDA(idx);
  }

  at::Tensor output =
      torch::zeros({points.size(0), points.size(1), idx.size(1), idx.size(2)},
                   at::device(points.device()).dtype(at::ScalarType::Float));

  if (points.type().is_cuda()) {
#if CUDA_AVAILABLE
    group_points_kernel_wrapper(points.size(0), points.size(1), points.size(2),
                                idx.size(1), idx.size(2), points.data<float>(),
                                idx.data<int>(), output.data<float>());
#else
    TORCH_CHECK(false, "CUDA not available, but CUDA tensor provided");
#endif
  } else {
    group_points_kernel_cpu_wrapper(points.size(0), points.size(1), points.size(2),
                                idx.size(1), idx.size(2), points.data<float>(),
                                idx.data<int>(), output.data<float>());
  }

  return output;
}

at::Tensor group_points_grad(at::Tensor grad_out, at::Tensor idx, const int n) {
  CHECK_CONTIGUOUS(grad_out);
  CHECK_CONTIGUOUS(idx);
  CHECK_IS_FLOAT(grad_out);
  CHECK_IS_INT(idx);

  if (grad_out.type().is_cuda()) {
    CHECK_CUDA(idx);
  }

  at::Tensor output =
      torch::zeros({grad_out.size(0), grad_out.size(1), n},
                   at::device(grad_out.device()).dtype(at::ScalarType::Float));

  if (grad_out.type().is_cuda()) {
#if CUDA_AVAILABLE
    group_points_grad_kernel_wrapper(
        grad_out.size(0), grad_out.size(1), n, idx.size(1), idx.size(2),
        grad_out.data<float>(), idx.data<int>(), output.data<float>());
#else
    TORCH_CHECK(false, "CUDA not available, but CUDA tensor provided");
#endif
  } else {
    group_points_grad_kernel_cpu_wrapper(
        grad_out.size(0), grad_out.size(1), n, idx.size(1), idx.size(2),
        grad_out.data<float>(), idx.data<int>(), output.data<float>());
  }

  return output;
}
