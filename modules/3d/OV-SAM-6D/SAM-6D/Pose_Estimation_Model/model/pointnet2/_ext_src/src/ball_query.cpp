// Copyright (c) Facebook, Inc. and its affiliates.
// 
// This source code is licensed under the MIT license found in the
// LICENSE file in the root directory of this source tree.

#include "ball_query.h"
#include "utils.h"

// CUDA kernel wrapper declaration - only available when CUDA is available
#if CUDA_AVAILABLE
void query_ball_point_kernel_wrapper(int b, int n, int m, float radius,
                                     int nsample, const float *new_xyz,
                                     const float *xyz, int *idx);
#endif

void query_ball_point_kernel_cpu_wrapper(int b, int n, int m, float radius,
                                     int nsample, const float *new_xyz,
                                     const float *xyz, int *idx){
    // Calculate radius squared for distance comparison
    float radius2 = radius * radius;

    for (int batch_index = 0; batch_index < b; ++batch_index) {
      // Calculate offset for current batch
      const float *current_xyz = xyz + batch_index * n * 3;
      const float *current_new_xyz = new_xyz + batch_index * m * 3;
      int *current_idx = idx + batch_index * m * nsample;

      for (int j = 0; j < m; ++j) {
        float new_x = current_new_xyz[j * 3 + 0];
        float new_y = current_new_xyz[j * 3 + 1];
        float new_z = current_new_xyz[j * 3 + 2];
        int cnt = 0;

        // Iterate through all original points to find those within specified radius
        for (int k = 0; k < n && cnt < nsample; ++k) {
          float x = current_xyz[k * 3 + 0];
          float y = current_xyz[k * 3 + 1];
          float z = current_xyz[k * 3 + 2];
          float d2 = (new_x - x) * (new_x - x) +
                    (new_y - y) * (new_y - y) +
                    (new_z - z) * (new_z - z);

          if (d2 < radius2) {
            if (cnt == 0) {
              // Initialize index array, if not enough neighbors found, repeat the last valid neighbor index
              for (int l = 0; l < nsample; ++l) {
                current_idx[j * nsample + l] = k;
              }
            }
            current_idx[j * nsample + cnt] = k;
            ++cnt;
          }
        }

        // If fewer points found than nsample, fill remaining indices with last valid index or -1
        // while (cnt < nsample) {
        //   current_idx[j * nsample + cnt] = (cnt == 0) ? -1 : current_idx[j * nsample + cnt - 1];
        //   ++cnt;
        // }
      }
    }
  }  

at::Tensor ball_query(at::Tensor new_xyz, at::Tensor xyz, const float radius,
                      const int nsample) {
  CHECK_CONTIGUOUS(new_xyz);
  CHECK_CONTIGUOUS(xyz);
  CHECK_IS_FLOAT(new_xyz);
  CHECK_IS_FLOAT(xyz);

  if (new_xyz.type().is_cuda()) {
    CHECK_CUDA(xyz);
  }

  at::Tensor idx =
      torch::zeros({new_xyz.size(0), new_xyz.size(1), nsample},
                   at::device(new_xyz.device()).dtype(at::ScalarType::Int));

  if (new_xyz.type().is_cuda()) {
#if CUDA_AVAILABLE
    query_ball_point_kernel_wrapper(xyz.size(0), xyz.size(1), new_xyz.size(1),
                                    radius, nsample, new_xyz.data<float>(),
                                    xyz.data<float>(), idx.data<int>());
#else
    TORCH_CHECK(false, "CUDA not available, but CUDA tensor provided");
#endif
  } else {
    query_ball_point_kernel_cpu_wrapper(xyz.size(0), xyz.size(1), new_xyz.size(1),
                                    radius, nsample, new_xyz.data<float>(),
                                    xyz.data<float>(), idx.data<int>());
  }

  return idx;
}
