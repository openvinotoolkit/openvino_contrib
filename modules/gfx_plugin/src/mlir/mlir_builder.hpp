// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//
#pragma once

#include <memory>
#include <vector>

#include "mlir/IR/BuiltinOps.h"
#include "openvino/core/node.hpp"
#include "runtime/gfx_activation.hpp"
#include "runtime/gfx_input_transform.hpp"

namespace ov {
class Model;
namespace op {
namespace v1 {
class Convolution;
class GroupConvolution;
}  // namespace v1
}  // namespace op
namespace gfx_plugin {

using MlirInputTransformDesc = GfxInputTransform;

// Build a minimal MLIR module that wraps a single MatMul using linalg.matmul.
mlir::ModuleOp build_mlir_module_from_model(const std::shared_ptr<const ov::Model>& model,
                                            mlir::MLIRContext& ctx);

// Build MLIR module for a single unary activation using linalg.generic.
mlir::ModuleOp build_mlir_unary_from_node(const std::shared_ptr<const ov::Node>& node,
                                          mlir::MLIRContext& ctx,
                                          ActivationKind kind,
                                          float alpha,
                                          std::optional<std::pair<double, double>> clamp_range = std::nullopt);

// Build MLIR modules for binary eltwise ops with broadcast & dynamic shapes.
mlir::ModuleOp build_mlir_add_from_model(const std::shared_ptr<const ov::Model>& model, mlir::MLIRContext& ctx);
mlir::ModuleOp build_mlir_add_from_node(const std::shared_ptr<const ov::Node>& node,
                                        mlir::MLIRContext& ctx,
                                        const std::vector<MlirInputTransformDesc>& input_transforms);
mlir::ModuleOp build_mlir_sub_from_model(const std::shared_ptr<const ov::Model>& model, mlir::MLIRContext& ctx);
mlir::ModuleOp build_mlir_mul_from_model(const std::shared_ptr<const ov::Model>& model, mlir::MLIRContext& ctx);
mlir::ModuleOp build_mlir_div_from_model(const std::shared_ptr<const ov::Model>& model, mlir::MLIRContext& ctx);
mlir::ModuleOp build_mlir_pow_from_model(const std::shared_ptr<const ov::Model>& model, mlir::MLIRContext& ctx);
mlir::ModuleOp build_mlir_mod_from_model(const std::shared_ptr<const ov::Model>& model, mlir::MLIRContext& ctx);
mlir::ModuleOp build_mlir_floor_mod_from_model(const std::shared_ptr<const ov::Model>& model, mlir::MLIRContext& ctx);
mlir::ModuleOp build_mlir_prelu_from_model(const std::shared_ptr<const ov::Model>& model, mlir::MLIRContext& ctx);
mlir::ModuleOp build_mlir_min_from_model(const std::shared_ptr<const ov::Model>& model, mlir::MLIRContext& ctx);
mlir::ModuleOp build_mlir_max_from_model(const std::shared_ptr<const ov::Model>& model, mlir::MLIRContext& ctx);
mlir::ModuleOp build_mlir_logical_and_from_model(const std::shared_ptr<const ov::Model>& model, mlir::MLIRContext& ctx);
mlir::ModuleOp build_mlir_logical_or_from_model(const std::shared_ptr<const ov::Model>& model, mlir::MLIRContext& ctx);
mlir::ModuleOp build_mlir_logical_xor_from_model(const std::shared_ptr<const ov::Model>& model, mlir::MLIRContext& ctx);
mlir::ModuleOp build_mlir_equal_from_model(const std::shared_ptr<const ov::Model>& model, mlir::MLIRContext& ctx);
mlir::ModuleOp build_mlir_not_equal_from_model(const std::shared_ptr<const ov::Model>& model, mlir::MLIRContext& ctx);
mlir::ModuleOp build_mlir_less_from_model(const std::shared_ptr<const ov::Model>& model, mlir::MLIRContext& ctx);
mlir::ModuleOp build_mlir_greater_from_model(const std::shared_ptr<const ov::Model>& model, mlir::MLIRContext& ctx);
mlir::ModuleOp build_mlir_less_equal_from_model(const std::shared_ptr<const ov::Model>& model, mlir::MLIRContext& ctx);
mlir::ModuleOp build_mlir_greater_equal_from_model(const std::shared_ptr<const ov::Model>& model, mlir::MLIRContext& ctx);
mlir::ModuleOp build_mlir_squared_difference_from_model(const std::shared_ptr<const ov::Model>& model,
                                                        mlir::MLIRContext& ctx);

// Build MLIR module for Softmax over an arbitrary normalized axis.
mlir::ModuleOp build_mlir_softmax_from_model(const std::shared_ptr<const ov::Model>& model,
                                             mlir::MLIRContext& ctx);
mlir::ModuleOp build_mlir_logsoftmax_from_model(const std::shared_ptr<const ov::Model>& model,
                                                mlir::MLIRContext& ctx);
// Build MLIR module for a single Softmax/LogSoftmax node using the provided input shape.
mlir::ModuleOp build_mlir_softmax_from_node(const std::shared_ptr<const ov::Node>& node,
                                            mlir::MLIRContext& ctx,
                                            const ov::Shape& input_shape);
mlir::ModuleOp build_mlir_logsoftmax_from_node(const std::shared_ptr<const ov::Node>& node,
                                               mlir::MLIRContext& ctx,
                                               const ov::Shape& input_shape);
// Tiled softmax variants accept an extra parameter buffer with {offset, count} for rows*inner slicing.
mlir::ModuleOp build_mlir_softmax_tiled_from_node(const std::shared_ptr<const ov::Node>& node,
                                                  mlir::MLIRContext& ctx,
                                                  const ov::Shape& input_shape);
mlir::ModuleOp build_mlir_logsoftmax_tiled_from_node(const std::shared_ptr<const ov::Node>& node,
                                                     mlir::MLIRContext& ctx,
                                                     const ov::Shape& input_shape);
mlir::ModuleOp build_mlir_maxpool_from_model(const std::shared_ptr<const ov::Model>& model,
                                             mlir::MLIRContext& ctx);
mlir::ModuleOp build_mlir_avgpool_from_model(const std::shared_ptr<const ov::Model>& model,
                                             mlir::MLIRContext& ctx);
mlir::ModuleOp build_mlir_conv2d_from_model(const std::shared_ptr<const ov::Model>& model,
                                            mlir::MLIRContext& ctx);
mlir::ModuleOp build_mlir_conv2d_from_node(const std::shared_ptr<const ov::op::v1::Convolution>& conv,
                                           mlir::MLIRContext& ctx,
                                           const MlirInputTransformDesc* input_transform = nullptr);
// Fused variant: applies unary activation (if any) inside the same tensor function.
mlir::ModuleOp build_mlir_conv2d_from_model(const std::shared_ptr<const ov::Model>& model,
                                            mlir::MLIRContext& ctx,
                                            std::optional<std::pair<ActivationKind, float>> unary_kind);
// Conv2D + bias (+optional unary) fused in one tensor function.
mlir::ModuleOp build_mlir_conv2d_with_bias_from_model(const std::shared_ptr<const ov::Model>& model,
                                                      mlir::MLIRContext& ctx,
                                                      std::optional<std::pair<ActivationKind, float>> unary_kind);
mlir::ModuleOp build_mlir_conv2d_vulkan(const std::shared_ptr<const ov::op::v1::Convolution>& conv,
                                        mlir::MLIRContext& ctx);
mlir::ModuleOp build_mlir_group_conv2d_vulkan(const std::shared_ptr<const ov::op::v1::GroupConvolution>& gconv,
                                              mlir::MLIRContext& ctx,
                                              const MlirInputTransformDesc* input_transform = nullptr);
mlir::ModuleOp build_mlir_group_conv2d_from_model(const std::shared_ptr<const ov::Model>& model,
                                                  mlir::MLIRContext& ctx);
mlir::ModuleOp build_mlir_group_conv2d_from_node(const std::shared_ptr<const ov::op::v1::GroupConvolution>& gconv,
                                                 mlir::MLIRContext& ctx,
                                                 const MlirInputTransformDesc* input_transform = nullptr);
mlir::ModuleOp build_mlir_conv3d_from_model(const std::shared_ptr<const ov::Model>& model,
                                            mlir::MLIRContext& ctx);
mlir::ModuleOp build_mlir_batchnorm_from_model(const std::shared_ptr<const ov::Model>& model,
                                               mlir::MLIRContext& ctx);
mlir::ModuleOp build_mlir_convert_from_model(const std::shared_ptr<const ov::Model>& model,
                                             mlir::MLIRContext& ctx);
mlir::ModuleOp build_mlir_transpose_from_model(const std::shared_ptr<const ov::Model>& model,
                                               mlir::MLIRContext& ctx);
mlir::ModuleOp build_mlir_slice_from_model(const std::shared_ptr<const ov::Model>& model,
                                           mlir::MLIRContext& ctx);
mlir::ModuleOp build_mlir_concat_from_model(const std::shared_ptr<const ov::Model>& model,
                                            mlir::MLIRContext& ctx);
mlir::ModuleOp build_mlir_split_from_model(const std::shared_ptr<const ov::Model>& model,
                                           mlir::MLIRContext& ctx);
// Build MLIR module for a Split/VariadicSplit node using the provided input shape.
mlir::ModuleOp build_mlir_split_from_node(const std::shared_ptr<const ov::Node>& node,
                                          mlir::MLIRContext& ctx,
                                          const ov::Shape& input_shape,
                                          const MlirInputTransformDesc* input_transform = nullptr);
mlir::ModuleOp build_mlir_interpolate_from_model(const std::shared_ptr<const ov::Model>& model,
                                                 mlir::MLIRContext& ctx);
mlir::ModuleOp build_mlir_gather_from_model(const std::shared_ptr<const ov::Model>& model,
                                            mlir::MLIRContext& ctx);
mlir::ModuleOp build_mlir_gathernd_from_model(const std::shared_ptr<const ov::Model>& model,
                                              mlir::MLIRContext& ctx);
mlir::ModuleOp build_mlir_gather_elements_from_model(const std::shared_ptr<const ov::Model>& model,
                                                     mlir::MLIRContext& ctx);
mlir::ModuleOp build_mlir_depth_to_space_from_model(const std::shared_ptr<const ov::Model>& model,
                                                    mlir::MLIRContext& ctx);
mlir::ModuleOp build_mlir_space_to_depth_from_model(const std::shared_ptr<const ov::Model>& model,
                                                    mlir::MLIRContext& ctx);
mlir::ModuleOp build_mlir_scatter_elements_update_from_model(const std::shared_ptr<const ov::Model>& model,
                                                             mlir::MLIRContext& ctx);
mlir::ModuleOp build_mlir_scatter_nd_update_from_model(const std::shared_ptr<const ov::Model>& model,
                                                       mlir::MLIRContext& ctx);
mlir::ModuleOp build_mlir_shapeof_from_model(const std::shared_ptr<const ov::Model>& model,
                                             mlir::MLIRContext& ctx);
mlir::ModuleOp build_mlir_select_from_model(const std::shared_ptr<const ov::Model>& model,
                                            mlir::MLIRContext& ctx);
mlir::ModuleOp build_mlir_reducesum_from_model(const std::shared_ptr<const ov::Model>& model, mlir::MLIRContext& ctx);
mlir::ModuleOp build_mlir_reducemean_from_model(const std::shared_ptr<const ov::Model>& model, mlir::MLIRContext& ctx);
mlir::ModuleOp build_mlir_reducemax_from_model(const std::shared_ptr<const ov::Model>& model, mlir::MLIRContext& ctx);
mlir::ModuleOp build_mlir_reducemin_from_model(const std::shared_ptr<const ov::Model>& model, mlir::MLIRContext& ctx);
mlir::ModuleOp build_mlir_reduceprod_from_model(const std::shared_ptr<const ov::Model>& model, mlir::MLIRContext& ctx);
mlir::ModuleOp build_mlir_reducel1_from_model(const std::shared_ptr<const ov::Model>& model, mlir::MLIRContext& ctx);
mlir::ModuleOp build_mlir_reducel2_from_model(const std::shared_ptr<const ov::Model>& model, mlir::MLIRContext& ctx);
mlir::ModuleOp build_mlir_pad_from_model(const std::shared_ptr<const ov::Model>& model, mlir::MLIRContext& ctx);
mlir::ModuleOp build_mlir_tile_from_model(const std::shared_ptr<const ov::Model>& model, mlir::MLIRContext& ctx);
mlir::ModuleOp build_mlir_broadcast_from_model(const std::shared_ptr<const ov::Model>& model, mlir::MLIRContext& ctx);
mlir::ModuleOp build_mlir_range_from_model(const std::shared_ptr<const ov::Model>& model, mlir::MLIRContext& ctx);
mlir::ModuleOp build_mlir_topk_from_model(const std::shared_ptr<const ov::Model>& model, mlir::MLIRContext& ctx);
mlir::ModuleOp build_mlir_reverse_from_model(const std::shared_ptr<const ov::Model>& model, mlir::MLIRContext& ctx);

}  // namespace gfx_plugin
}  // namespace ov
