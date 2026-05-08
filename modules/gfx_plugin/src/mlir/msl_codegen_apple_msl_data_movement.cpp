// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "mlir/msl_codegen_apple_msl.hpp"

#include <algorithm>
#include <string>
#include <utility>

#include "mlir/msl_codegen.hpp"
#include "mlir/msl_codegen_apple_msl_common.hpp"
#include "openvino/core/except.hpp"
#include "openvino/core/shape_util.hpp"
#include "openvino/op/constant.hpp"
#include "openvino/op/gather.hpp"
#include "openvino/op/gather_elements.hpp"
#include "openvino/op/gather_nd.hpp"
#include "openvino/op/interpolate.hpp"
#include "openvino/op/scatter_elements_update.hpp"
#include "openvino/op/scatter_nd_update.hpp"
#include "openvino/op/scatter_update.hpp"
#include "openvino/op/slice.hpp"
#include "openvino/op/strided_slice.hpp"
#include "openvino/util/common_util.hpp"
#include "runtime/gfx_shape_utils.hpp"

namespace ov {
namespace gfx_plugin {

bool configure_apple_metal_data_movement_kernel_source(
    KernelSource &source, const std::shared_ptr<const ov::Node> &node) {
  if (!node) {
    return false;
  }

  auto set_desc = [&](auto &&desc, const char *entry) {
    source.entry_point = entry;
    source.msl_generator = [desc](mlir::ModuleOp module) mutable {
      return generate_msl_from_mlir(module, desc);
    };
  };

  if (auto interp =
          std::dynamic_pointer_cast<const ov::op::v0::Interpolate>(node)) {
    InterpolateCodegenDesc desc{};
    const auto in = interp->get_input_shape(0);
    const auto out = interp->get_output_shape(0);
    desc.element_type = interp->get_output_element_type(0);
    desc.N = in[0];
    desc.C = in[1];
    desc.H_in = in[2];
    desc.W_in = in[3];
    desc.H_out = out[2];
    desc.W_out = out[3];
    desc.scale_h = desc.H_out ? static_cast<float>(desc.H_in) /
                                    static_cast<float>(desc.H_out)
                              : 1.f;
    desc.scale_w = desc.W_out ? static_cast<float>(desc.W_in) /
                                    static_cast<float>(desc.W_out)
                              : 1.f;
    desc.align_corners = interp->get_attrs().align_corners;
    desc.nearest = ov::util::to_lower(interp->get_attrs().mode) == "nearest";
    desc.use_half_pixel = !desc.align_corners;
    desc.nearest_mode = 0;
    set_desc(desc, "interpolate_kernel");
    return true;
  }

  if (auto interp =
          std::dynamic_pointer_cast<const ov::op::v4::Interpolate>(node)) {
    using Base = ov::op::util::InterpolateBase;
    InterpolateCodegenDesc desc{};
    const auto in = interp->get_input_shape(0);
    const auto out = interp->get_output_shape(0);
    desc.element_type = interp->get_output_element_type(0);
    desc.N = in[0];
    desc.C = in[1];
    desc.H_in = in[2];
    desc.W_in = in[3];
    desc.H_out = out[2];
    desc.W_out = out[3];
    desc.scale_h = desc.H_out ? static_cast<float>(desc.H_in) /
                                    static_cast<float>(desc.H_out)
                              : 1.f;
    desc.scale_w = desc.W_out ? static_cast<float>(desc.W_in) /
                                    static_cast<float>(desc.W_out)
                              : 1.f;
    desc.align_corners = interp->get_attrs().coordinate_transformation_mode ==
                         Base::CoordinateTransformMode::ALIGN_CORNERS;
    desc.nearest = interp->get_attrs().mode == Base::InterpolateMode::NEAREST;
    desc.use_half_pixel = interp->get_attrs().coordinate_transformation_mode ==
                          Base::CoordinateTransformMode::HALF_PIXEL;
    switch (interp->get_attrs().nearest_mode) {
    case Base::NearestMode::FLOOR:
    case Base::NearestMode::ROUND_PREFER_FLOOR:
      desc.nearest_mode = 1;
      break;
    case Base::NearestMode::CEIL:
    case Base::NearestMode::ROUND_PREFER_CEIL:
      desc.nearest_mode = 2;
      break;
    case Base::NearestMode::SIMPLE:
    default:
      desc.nearest_mode = 0;
      break;
    }
    set_desc(desc, "interpolate_kernel");
    return true;
  }

  if (auto interp =
          std::dynamic_pointer_cast<const ov::op::v11::Interpolate>(node)) {
    using Base = ov::op::util::InterpolateBase;
    InterpolateCodegenDesc desc{};
    const auto in = interp->get_input_shape(0);
    const auto out = interp->get_output_shape(0);
    desc.element_type = interp->get_output_element_type(0);
    desc.N = in[0];
    desc.C = in[1];
    desc.H_in = in[2];
    desc.W_in = in[3];
    desc.H_out = out[2];
    desc.W_out = out[3];
    desc.scale_h = desc.H_out ? static_cast<float>(desc.H_in) /
                                    static_cast<float>(desc.H_out)
                              : 1.f;
    desc.scale_w = desc.W_out ? static_cast<float>(desc.W_in) /
                                    static_cast<float>(desc.W_out)
                              : 1.f;
    desc.align_corners = interp->get_attrs().coordinate_transformation_mode ==
                         Base::CoordinateTransformMode::ALIGN_CORNERS;
    desc.nearest = interp->get_attrs().mode == Base::InterpolateMode::NEAREST;
    desc.use_half_pixel = interp->get_attrs().coordinate_transformation_mode ==
                          Base::CoordinateTransformMode::HALF_PIXEL;
    switch (interp->get_attrs().nearest_mode) {
    case Base::NearestMode::FLOOR:
    case Base::NearestMode::ROUND_PREFER_FLOOR:
      desc.nearest_mode = 1;
      break;
    case Base::NearestMode::CEIL:
    case Base::NearestMode::ROUND_PREFER_CEIL:
      desc.nearest_mode = 2;
      break;
    case Base::NearestMode::SIMPLE:
    default:
      desc.nearest_mode = 0;
      break;
    }
    set_desc(desc, "interpolate_kernel");
    return true;
  }

  if (std::dynamic_pointer_cast<const ov::op::v8::Slice>(node) ||
      std::dynamic_pointer_cast<const ov::op::v1::StridedSlice>(node)) {
    ConvertCodegenDesc desc{};
    desc.element_type = node->get_output_element_type(0);
    desc.dst_type = desc.element_type;
    source.entry_point = "slice_kernel";
    source.msl_generator = [desc](mlir::ModuleOp module) mutable {
      return generate_msl_for_slice_generic(desc, module);
    };
    if (source.module) {
      require_apple_msl_custom_kernel_binding(
          source.module, node->get_type_name(), "slice_kernel");
    }
    return true;
  }

  if (auto gather =
          std::dynamic_pointer_cast<const ov::op::util::GatherBase>(node)) {
    GatherCodegenDesc desc{};
    if (auto g7 = std::dynamic_pointer_cast<const ov::op::v7::Gather>(node)) {
      OPENVINO_ASSERT(g7->get_batch_dims() == 0,
                      "GFX Metal Gather: batch_dims not supported");
    } else if (auto g8 =
                   std::dynamic_pointer_cast<const ov::op::v8::Gather>(node)) {
      OPENVINO_ASSERT(g8->get_batch_dims() == 0,
                      "GFX Metal Gather: batch_dims not supported");
    }
    desc.index_type = gather->get_input_element_type(1);
    desc.element_type = gather->get_output_element_type(0);
    const auto data_pshape = gather->get_input_partial_shape(0);
    OPENVINO_ASSERT(data_pshape.rank().is_static(),
                    "GFX Metal Gather: data rank must be static");
    const size_t rank = static_cast<size_t>(data_pshape.rank().get_length());
    OPENVINO_ASSERT(rank > 0, "GFX Metal Gather: data rank must be positive");
    const size_t axis = static_cast<size_t>(
        normalize_axis(gather->get_axis(), rank, "GFX Metal Gather"));
    (void)axis;
    desc.outer = 1;
    desc.inner = 1;
    desc.axis_dim = 1;
    desc.indices_count = 1;
    set_desc(desc, "gather_kernel");
    return true;
  }

  if (auto gather_nd =
          std::dynamic_pointer_cast<const ov::op::v5::GatherND>(node)) {
    GatherNDCodegenDesc desc{};
    const auto data = gather_nd->get_input_shape(0);
    const auto indices = gather_nd->get_input_shape(1);
    desc.index_type = gather_nd->get_input_element_type(1);
    desc.k = static_cast<uint32_t>(indices.back());
    desc.num_indices = static_cast<uint32_t>(ov::shape_size(indices) / desc.k);
    desc.element_type = gather_nd->get_output_element_type(0);
    uint32_t stride = 1;
    const size_t rank = data.size();
    for (int i = static_cast<int>(rank) - 1; i >= 0; --i) {
      desc.dims[static_cast<size_t>(i)] =
          static_cast<uint32_t>(data[static_cast<size_t>(i)]);
      desc.strides[static_cast<size_t>(i)] = stride;
      stride *= desc.dims[static_cast<size_t>(i)];
    }
    desc.inner = desc.strides[desc.k];
    desc.total = static_cast<uint32_t>(ov::shape_size(data));
    set_desc(desc, "gathernd_kernel");
    return true;
  }

  if (auto gather_elements =
          std::dynamic_pointer_cast<const ov::op::v6::GatherElements>(node)) {
    GatherElementsCodegenDesc desc{};
    const auto data = gather_elements->get_input_shape(0);
    const auto out = gather_elements->get_output_shape(0);
    desc.index_type = gather_elements->get_input_element_type(1);
    desc.rank = static_cast<uint32_t>(out.size());
    desc.axis = static_cast<uint32_t>(gather_elements->get_axis());
    desc.total = static_cast<uint32_t>(ov::shape_size(out));
    auto data_strides = make_strides(data);
    auto out_strides = make_strides(out);
    for (size_t i = 0; i < out.size() && i < desc.kMaxDims; ++i) {
      desc.out_dims[i] = static_cast<uint32_t>(out[i]);
      desc.out_strides[i] = static_cast<uint32_t>(out_strides[i]);
      desc.data_dims[i] = static_cast<uint32_t>(data[i]);
      desc.data_strides[i] = static_cast<uint32_t>(data_strides[i]);
    }
    set_desc(desc, "gather_elements_kernel");
    source.signature.output_arg_count = 1;
    return true;
  }

  if (auto scatter =
          std::dynamic_pointer_cast<const ov::op::v3::ScatterUpdate>(node)) {
    ScatterUpdateCodegenDesc desc{};
    desc.element_type = scatter->get_output_element_type(0);
    desc.index_type = scatter->get_input_element_type(1);
    set_desc(desc, "scatter_update_kernel");
    if (source.module) {
      require_apple_msl_custom_kernel_binding(source.module, "ScatterUpdate",
                                              "scatter_update_kernel");
    }
    return true;
  }

  if (auto scatter =
          std::dynamic_pointer_cast<const ov::op::v3::ScatterNDUpdate>(node)) {
    ScatterNDUpdateCodegenDesc desc{};
    const auto data = scatter->get_input_shape(0);
    const auto indices = scatter->get_input_shape(1);
    desc.index_type = scatter->get_input_element_type(1);
    desc.k = static_cast<uint32_t>(indices.back());
    uint32_t stride = 1;
    for (int i = static_cast<int>(data.size()) - 1; i >= 0; --i) {
      desc.dims[static_cast<size_t>(i)] =
          static_cast<uint32_t>(data[static_cast<size_t>(i)]);
      desc.strides[static_cast<size_t>(i)] = stride;
      stride *= desc.dims[static_cast<size_t>(i)];
    }
    desc.inner = desc.strides[desc.k];
    desc.num_indices = static_cast<uint32_t>(ov::shape_size(indices) / desc.k);
    desc.total_updates =
        static_cast<uint32_t>(ov::shape_size(scatter->get_input_shape(2)));
    desc.total_data = static_cast<uint32_t>(ov::shape_size(data));
    desc.element_type = scatter->get_output_element_type(0);
    set_desc(desc, "scatter_nd_update");
    return true;
  }

  if (auto scatter =
          std::dynamic_pointer_cast<const ov::op::v3::ScatterElementsUpdate>(
              node)) {
    ScatterElementsUpdateCodegenDesc desc{};
    const auto data = scatter->get_input_shape(0);
    const auto indices = scatter->get_input_shape(1);
    desc.index_type = scatter->get_input_element_type(1);
    auto axis_const = ov::as_type_ptr<const ov::op::v0::Constant>(
        scatter->input_value(2).get_node_shared_ptr());
    OPENVINO_ASSERT(axis_const, "ScatterElementsUpdate axis must be constant");
    desc.axis = static_cast<uint32_t>(axis_const->cast_vector<int64_t>()[0]);
    desc.rank = static_cast<uint32_t>(data.size());
    desc.total_updates = static_cast<uint32_t>(ov::shape_size(indices));
    desc.total_data = static_cast<uint32_t>(ov::shape_size(data));
    auto data_strides = make_strides(data);
    auto update_strides = make_strides(indices);
    for (size_t i = 0; i < data.size() && i < desc.kMaxDims; ++i) {
      desc.data_dims[i] = static_cast<uint32_t>(data[i]);
      desc.data_strides[i] = static_cast<uint32_t>(data_strides[i]);
    }
    for (size_t i = 0; i < indices.size() && i < desc.kMaxDims; ++i) {
      desc.update_dims[i] = static_cast<uint32_t>(indices[i]);
      desc.update_strides[i] = static_cast<uint32_t>(update_strides[i]);
    }
    desc.element_type = scatter->get_output_element_type(0);
    set_desc(desc, "scatter_elements_update");
    return true;
  }

  return false;
}

} // namespace gfx_plugin
} // namespace ov
