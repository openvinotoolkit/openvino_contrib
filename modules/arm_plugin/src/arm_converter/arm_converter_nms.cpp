// Copyright (C) 2020-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0


#include "arm_converter/arm_converter.hpp"
#include <ngraph/runtime/reference/non_max_suppression.hpp>

namespace ArmPlugin {
static void normalize_corner(float* boxes, size_t size) {
    size_t total_num_of_boxes = size / 4;
    for (size_t i = 0; i < total_num_of_boxes; ++i) {
        float* current_box = boxes + 4 * i;

        float y1 = current_box[0];
        float x1 = current_box[1];
        float y2 = current_box[2];
        float x2 = current_box[3];

        float ymin = std::min(y1, y2);
        float ymax = std::max(y1, y2);
        float xmin = std::min(x1, x2);
        float xmax = std::max(x1, x2);

        current_box[0] = ymin;
        current_box[1] = xmin;
        current_box[2] = ymax;
        current_box[3] = xmax;
    }
}

static void normalize_center(float* boxes, size_t size) {
    size_t total_num_of_boxes = size / 4;
    for (size_t i = 0; i < total_num_of_boxes; ++i) {
        float* current_box = boxes + 4 * i;

        float x_center = current_box[0];
        float y_center = current_box[1];
        float width = current_box[2];
        float height = current_box[3];

        float y1 = y_center - height / 2.0;
        float x1 = x_center - width / 2.0;
        float y2 = y_center + height / 2.0;
        float x2 = x_center + width / 2.0;

        current_box[0] = y1;
        current_box[1] = x1;
        current_box[2] = y2;
        current_box[3] = x2;
    }
}

static std::vector<float> prepare_boxes_data(const float* boxes,
                                             const ngraph::Shape& boxes_shape,
                                             const opset::NonMaxSuppression::BoxEncodingType box_encoding) {
    auto size = ngraph::shape_size(boxes_shape);
    std::vector<float> normalized_boxes(boxes, boxes + size);
    if (box_encoding == opset::NonMaxSuppression::BoxEncodingType::CORNER) {
        normalize_corner(normalized_boxes.data(), size);
    } else {
        normalize_center(normalized_boxes.data(), size);
    }
    return normalized_boxes;
}

static void nms5(const float* boxes_data,
          const ngraph::Shape& boxes_data_shape,
          const float* scores_data,
          const ngraph::Shape& scores_data_shape,
          int64_t max_output_boxes_per_class,
          float iou_threshold,
          float score_threshold,
          float soft_nms_sigma,
          const ngraph::Shape& out_shape,
          const bool sort_result_descending,
          const ngraph::element::Type out_type,
          const ngraph::HostTensorVector& outputs,
          const ngraph::element::Type selected_scores_type,
          const opset::NonMaxSuppression::BoxEncodingType box_encoding) {
    std::vector<int64_t> selected_indices(ngraph::shape_size(out_shape));
    std::vector<float>   selected_scores(ngraph::shape_size(out_shape));
    int64_t valid_outputs = 0;
    std::vector<float> normalized_boxes = prepare_boxes_data(boxes_data, boxes_data_shape, box_encoding);

    ngraph::runtime::reference::non_max_suppression(normalized_boxes.data(),
                                                    boxes_data_shape,
                                                    scores_data,
                                                    scores_data_shape,
                                                    max_output_boxes_per_class,
                                                    iou_threshold,
                                                    score_threshold,
                                                    soft_nms_sigma,
                                                    selected_indices.data(),
                                                    out_shape,
                                                    selected_scores.data(),
                                                    out_shape,
                                                    &valid_outputs,
                                                    sort_result_descending);

    auto max_valid_outputs = out_shape[0];
    ngraph::runtime::reference::nms5_postprocessing(outputs,
                                                    out_type,
                                                    selected_indices,
                                                    selected_scores,
                                                    max_valid_outputs,
                                                    selected_scores_type);

    if (out_type == ngraph::element::i64) {
        int64_t* indices_ptr = outputs[0]->get_data_ptr<int64_t>();
        int64_t* valid_outputs_ptr = outputs[2]->get_data_ptr<int64_t>();
        *valid_outputs_ptr = valid_outputs;
        for (size_t i = 3 * valid_outputs; i < ngraph::shape_size(out_shape); i++) {
            indices_ptr[i] = -1;
        }
    } else {
        int32_t* indices_ptr = outputs[0]->get_data_ptr<int32_t>();
        int32_t* valid_outputs_ptr = outputs[2]->get_data_ptr<int32_t>();
        *valid_outputs_ptr = static_cast<int32_t>(valid_outputs);
        for (size_t i = 3 * valid_outputs; i < ngraph::shape_size(out_shape); i++) {
            indices_ptr[i] = -1;
        }
    }

    if (selected_scores_type == ngraph::element::f16) {
        ngraph::float16* scores_ptr = outputs[1]->get_data_ptr<ngraph::float16>();
        for (size_t i = 3 * valid_outputs; i < ngraph::shape_size(out_shape); i++) {
            scores_ptr[i] = ngraph::float16(-1);
        }
    } else {
        float* scores_ptr = outputs[1]->get_data_ptr<float>();
        for (size_t i = 3 * valid_outputs; i < ngraph::shape_size(out_shape); i++) {
            scores_ptr[i] = -1.f;
        }
    }
}

template<> Converter::Conversion::Ptr Converter::Convert(const opset::NonMaxSuppression& node) {
    auto make = [&] (auto refFunction) {
        ngraph::HostTensorVector hosts;
        for (auto output : node.outputs()) {
            auto tensor = std::make_shared<ngraph::HostTensor>(output.get_element_type(),
                                                               output.get_partial_shape().get_max_shape());
            hosts.push_back(tensor);
         }

        ngraph::element::Type selected_scores_type = node.get_input_size() < 4 ?
                                                     ngraph::element::f32 : node.get_input_element_type(3);
        return this->MakeConversion(refFunction,
                                    node.input(0),
                                    node.get_input_shape(0),
                                    node.input(1),
                                    node.get_input_shape(1),
                                    static_cast<int64_t>(node.max_boxes_output_from_input()),
                                    static_cast<float>(node.iou_threshold_from_input()),
                                    static_cast<float>(node.score_threshold_from_input()),
                                    static_cast<float>(node.soft_nms_sigma_from_input()),
                                    node.get_output_partial_shape(0).get_max_shape(),
                                    node.get_sort_result_descending(),
                                    node.output(0).get_element_type(),
                                    HostTensors{hosts, &node},
                                    selected_scores_type,
                                    node.get_box_encoding());
    };
    return make(nms5);
}
}  //  namespace ArmPlugin

