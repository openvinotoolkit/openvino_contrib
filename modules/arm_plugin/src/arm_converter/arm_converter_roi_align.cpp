// Copyright (C) 2020-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include <details/ie_exception.hpp>

#include "arm_converter/arm_converter.hpp"
#include <ngraph/runtime/reference/roi_align.hpp>

namespace ArmPlugin {
template <typename T, typename U>
static void wrapper_roi_align(const T* feature_maps,
                              const T* rois,
                              const U* batch_indices,
                              T* out,
                              const ngraph::Shape& feature_maps_shape,
                              const ngraph::Shape& rois_shape,
                              const ngraph::Shape& batch_indices_shape,
                              const ngraph::Shape& out_shape,
                              const int pooled_height,
                              const int pooled_width,
                              const int sampling_ratio,
                              const float spatial_scale,
                              const opset::ROIAlign::PoolingMode& pooling_mode) {
    auto size = ngraph::shape_size(batch_indices_shape);
    std::vector<int64_t> indices(size);
    for (size_t i = 0; i < size; i++) {
        indices[i] = static_cast<int64_t>(batch_indices[i]);
    }
    ngraph::runtime::reference::roi_align(feature_maps,
                                          rois,
                                          indices.data(),
                                          out,
                                          feature_maps_shape,
                                          rois_shape,
                                          batch_indices_shape,
                                          out_shape,
                                          pooled_height,
                                          pooled_width,
                                          sampling_ratio,
                                          spatial_scale,
                                          pooling_mode);
}

template<> Converter::Conversion::Ptr Converter::Convert(const opset::ROIAlign& node) {
    auto make = [&] (auto refFunction) {
        return this->MakeConversion(refFunction,
                                    node.input(0),
                                    node.input(1),
                                    node.input(2),
                                    node.output(0),
                                    node.get_input_shape(0),
                                    node.get_input_shape(1),
                                    node.get_input_shape(2),
                                    node.get_output_shape(0),
                                    node.get_pooled_h(),
                                    node.get_pooled_w(),
                                    node.get_sampling_ratio(),
                                    node.get_spatial_scale(),
                                    node.get_mode());
    };
    return CallSwitch(
        AP_WRAP(make, wrapper_roi_align),
        node.input(0), floatTypes,
        node.input(2), intTypes);
}
}  //  namespace ArmPlugin
