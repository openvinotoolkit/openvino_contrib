// Copyright (c) Facebook, Inc. and its affiliates.
// 
// This source code is licensed under the MIT license found in the
// LICENSE file in the root directory of this source tree.

#include "interpolate.h"
#include "utils.h"

// CUDA kernel wrapper declarations - only available when CUDA is available
#if CUDA_AVAILABLE
void three_nn_kernel_wrapper(int b, int n, int m, const float *unknown,
                             const float *known, float *dist2, int *idx);
void three_interpolate_kernel_wrapper(int b, int c, int m, int n,
                                      const float *points, const int *idx,
                                      const float *weight, float *out);
void three_interpolate_grad_kernel_wrapper(int b, int c, int n, int m,
                                           const float *grad_out,
                                           const int *idx, const float *weight,
                                           float *grad_points);
#endif

void three_nn_kernel_cpu_wrapper(int b, int n, int m, const float *unknown,
                             const float *known, float *dist2, int *idx){
    // Iterate through each batch
    for (int batch_index = 0; batch_index < b; ++batch_index) {
      // Calculate offset for current batch
      const float *current_unknown = unknown + batch_index * n * 3;
      const float *current_known = known + batch_index * m * 3;
      float *current_dist2 = dist2 + batch_index * n * 3;
      int *current_idx = idx + batch_index * n * 3;

      // Iterate through each unknown point
      for (int j = 0; j < n; ++j) {
        float ux = current_unknown[j * 3 + 0];
        float uy = current_unknown[j * 3 + 1];
        float uz = current_unknown[j * 3 + 2];

        double best1 = std::numeric_limits<double>::max();
        double best2 = std::numeric_limits<double>::max();
        double best3 = std::numeric_limits<double>::max();
        int besti1 = -1, besti2 = -1, besti3 = -1;

        // Iterate through all known points to find the three nearest points
        for (int k = 0; k < m; ++k) {
          float x = current_known[k * 3 + 0];
          float y = current_known[k * 3 + 1];
          float z = current_known[k * 3 + 2];
          float d = (ux - x) * (ux - x) + (uy - y) * (uy - y) + (uz - z) * (uz - z);

          if (d < best1) {
            best3 = best2;
            besti3 = besti2;
            best2 = best1;
            besti2 = besti1;
            best1 = d;
            besti1 = k;
          } else if (d < best2) {
            best3 = best2;
            besti3 = besti2;
            best2 = d;
            besti2 = k;
          } else if (d < best3) {
            best3 = d;
            besti3 = k;
          }
        }

        // Update result arrays
        current_dist2[j * 3 + 0] = static_cast<float>(best1);
        current_dist2[j * 3 + 1] = static_cast<float>(best2);
        current_dist2[j * 3 + 2] = static_cast<float>(best3);

        current_idx[j * 3 + 0] = besti1;
        current_idx[j * 3 + 1] = besti2;
        current_idx[j * 3 + 2] = besti3;
      }
    }
  }
                             
void three_interpolate_kernel_cpu_wrapper(int b, int c, int m, int n,
                                      const float *points, const int *idx,
                                      const float *weight, float *out){
    // Iterate through each batch
    for (int batch_index = 0; batch_index < b; ++batch_index) {
      // Calculate offset for current batch
      const float *current_points = points + batch_index * m * c;
      const int *current_idx = idx + batch_index * n * 3;
      const float *current_weight = weight + batch_index * n * 3;
      float *current_out = out + batch_index * n * c;

      // Iterate through each channel c and each point n
      for (int l = 0; l < c; ++l) { // Iterate through each channel
        for (int j = 0; j < n; ++j) { // Iterate through each point
          float w1 = current_weight[j * 3 + 0];
          float w2 = current_weight[j * 3 + 1];
          float w3 = current_weight[j * 3 + 2];

          int i1 = current_idx[j * 3 + 0];
          int i2 = current_idx[j * 3 + 1];
          int i3 = current_idx[j * 3 + 2];

          // Ensure indices are valid
          if(i1 >= 0 && i1 < m && i2 >= 0 && i2 < m && i3 >= 0 && i3 < m) {
            current_out[l * n + j] = current_points[l * m + i1] * w1 +
                                    current_points[l * m + i2] * w2 +
                                    current_points[l * m + i3] * w3;
          } else {
            // If indices are invalid, set a default value or handle it appropriately
            current_out[l * n + j] = 0.0f; // Simply set to 0.0 here
          }
        }
      }
    }
  }

void three_interpolate_grad_kernel_cpu_wrapper(int b, int c, int n, int m,
                                           const float *grad_out,
                                           const int *idx, const float *weight,
                                           float *grad_points){
    // Initialize grad_points array to 0 to avoid accumulation errors
    for (int batch_index = 0; batch_index < b; ++batch_index) {
      float *current_grad_points = grad_points + batch_index * m * c;
      for (int i = 0; i < m * c; ++i) {
        current_grad_points[i] = 0.0f;
      }
    }

    // Iterate through each batch
    for (int batch_index = 0; batch_index < b; ++batch_index) {
      // Calculate offset for current batch
      const float *current_grad_out = grad_out + batch_index * n * c;
      const int *current_idx = idx + batch_index * n * 3;
      const float *current_weight = weight + batch_index * n * 3;
      float *current_grad_points = grad_points + batch_index * m * c;

      // Iterate through each channel c and each point n
      for (int l = 0; l < c; ++l) { // Iterate through each channel
        for (int j = 0; j < n; ++j) { // Iterate through each point
          float w1 = current_weight[j * 3 + 0];
          float w2 = current_weight[j * 3 + 1];
          float w3 = current_weight[j * 3 + 2];

          int i1 = current_idx[j * 3 + 0];
          int i2 = current_idx[j * 3 + 1];
          int i3 = current_idx[j * 3 + 2];

          // Ensure indices are valid
          if(i1 >= 0 && i1 < m && i2 >= 0 && i2 < m && i3 >= 0 && i3 < m) {
            current_grad_points[l * m + i1] += current_grad_out[l * n + j] * w1;
            current_grad_points[l * m + i2] += current_grad_out[l * n + j] * w2;
            current_grad_points[l * m + i3] += current_grad_out[l * n + j] * w3;
          } else {
            // If indices are invalid, set a default value or handle it appropriately
            // Here, we choose to ignore invalid indices as grad_points is already initialized to 0
          }
        }
      }
    }
  }

std::vector<at::Tensor> three_nn(at::Tensor unknowns, at::Tensor knows) {
  CHECK_CONTIGUOUS(unknowns);
  CHECK_CONTIGUOUS(knows);
  CHECK_IS_FLOAT(unknowns);
  CHECK_IS_FLOAT(knows);

  if (unknowns.type().is_cuda()) {
    CHECK_CUDA(knows);
  }

  at::Tensor idx =
      torch::zeros({unknowns.size(0), unknowns.size(1), 3},
                   at::device(unknowns.device()).dtype(at::ScalarType::Int));
  at::Tensor dist2 =
      torch::zeros({unknowns.size(0), unknowns.size(1), 3},
                   at::device(unknowns.device()).dtype(at::ScalarType::Float));

  if (unknowns.type().is_cuda()) {
    #if CUDA_AVAILABLE
    three_nn_kernel_wrapper(unknowns.size(0), unknowns.size(1), knows.size(1),
                            unknowns.data<float>(), knows.data<float>(),
                            dist2.data<float>(), idx.data<int>());
    #else
    TORCH_CHECK(false, "CUDA not available");
    #endif
  } else {
    // TORCH_CHECK(false, "CPU not supported");
    three_nn_kernel_cpu_wrapper(unknowns.size(0), unknowns.size(1), knows.size(1),
                            unknowns.data<float>(), knows.data<float>(),
                            dist2.data<float>(), idx.data<int>());
  }

  return {dist2, idx};
}

at::Tensor three_interpolate(at::Tensor points, at::Tensor idx,
                             at::Tensor weight) {
  CHECK_CONTIGUOUS(points);
  CHECK_CONTIGUOUS(idx);
  CHECK_CONTIGUOUS(weight);
  CHECK_IS_FLOAT(points);
  CHECK_IS_INT(idx);
  CHECK_IS_FLOAT(weight);

  if (points.type().is_cuda()) {
    CHECK_CUDA(idx);
    CHECK_CUDA(weight);
  }

  at::Tensor output =
      torch::zeros({points.size(0), points.size(1), idx.size(1)},
                   at::device(points.device()).dtype(at::ScalarType::Float));

  if (points.type().is_cuda()) {
    #if CUDA_AVAILABLE
    three_interpolate_kernel_wrapper(
        points.size(0), points.size(1), points.size(2), idx.size(1),
        points.data<float>(), idx.data<int>(), weight.data<float>(),
        output.data<float>());
    #else
    TORCH_CHECK(false, "CUDA not available");
    #endif
  } else {
    // TORCH_CHECK(false, "CPU not supported");
    three_interpolate_kernel_cpu_wrapper(
        points.size(0), points.size(1), points.size(2), idx.size(1),
        points.data<float>(), idx.data<int>(), weight.data<float>(),
        output.data<float>());
  }

  return output;
}
at::Tensor three_interpolate_grad(at::Tensor grad_out, at::Tensor idx,
                                  at::Tensor weight, const int m) {
  CHECK_CONTIGUOUS(grad_out);
  CHECK_CONTIGUOUS(idx);
  CHECK_CONTIGUOUS(weight);
  CHECK_IS_FLOAT(grad_out);
  CHECK_IS_INT(idx);
  CHECK_IS_FLOAT(weight);

  if (grad_out.type().is_cuda()) {
    CHECK_CUDA(idx);
    CHECK_CUDA(weight);
  }

  at::Tensor output =
      torch::zeros({grad_out.size(0), grad_out.size(1), m},
                   at::device(grad_out.device()).dtype(at::ScalarType::Float));

  if (grad_out.type().is_cuda()) {
    #if CUDA_AVAILABLE
    three_interpolate_grad_kernel_wrapper(
        grad_out.size(0), grad_out.size(1), grad_out.size(2), m,
        grad_out.data<float>(), idx.data<int>(), weight.data<float>(),
        output.data<float>());
    #else
    TORCH_CHECK(false, "CUDA not available");
    #endif
  } else {
    // TORCH_CHECK(false, "CPU not supported");
    three_interpolate_grad_kernel_cpu_wrapper(
        grad_out.size(0), grad_out.size(1), grad_out.size(2), m,
        grad_out.data<float>(), idx.data<int>(), weight.data<float>(),
        output.data<float>());
  }

  return output;
}
