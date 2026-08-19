// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//
#pragma once

#include <vector>

#include "common/gpu_parallelism_plan.hpp"
#include "openvino/core/shape.hpp"

namespace ov {
namespace gfx_plugin {

class GpuBufferManager;

GfxParallelismCaps
query_parallelism_caps(const GpuBufferManager *buffer_manager);
std::vector<MatMulParallelismPlan>
enumerate_matmul_parallelism_candidates(const GfxParallelismCaps &caps,
                                        const ov::Shape &output_shape);
MatMulParallelismPlan select_matmul_parallelism(const GfxParallelismCaps &caps,
                                                const ov::Shape &output_shape);
void remember_matmul_parallelism(const GfxParallelismCaps &caps,
                                 const ov::Shape &output_shape,
                                 const MatMulParallelismPlan &plan);
ConvParallelismPlan select_conv_parallelism(const GfxParallelismCaps &caps,
                                            const ov::Shape &output_shape,
                                            uint64_t input_channels,
                                            uint64_t output_channels,
                                            uint64_t kernel_work, bool stride2,
                                            bool depthwise);
ChunkDispatchPlan select_chunk_dispatch_plan(const GfxParallelismCaps &caps,
                                             const std::string &op_kind,
                                             uint64_t total_elems,
                                             uint64_t work_per_elem);

} // namespace gfx_plugin
} // namespace ov
