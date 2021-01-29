// Copyright (C) 2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0


#include "transformations/convert_prior_box_clustered_to_const.hpp"

#include <details/ie_exception.hpp>
#include <ngraph/rt_info.hpp>
#include "opset/opset.hpp"

ArmPlugin::pass::ConvertPriorBoxClustered::ConvertPriorBoxClustered() {
    auto prior_box = std::make_shared<opset::PriorBoxClustered>(ngraph::pattern::any_input(),
                                                                ngraph::pattern::any_input(),
                                                                ngraph::op::PriorBoxClusteredAttrs{});

    ngraph::matcher_pass_callback callback = [](ngraph::pattern::Matcher& m) {
        auto node = std::dynamic_pointer_cast<opset::PriorBoxClustered>(m.get_match_root());
        if (!node) {
            return false;
        }

        auto layer_shape = std::dynamic_pointer_cast<opset::Constant>(node->input_value(0).get_node_shared_ptr());
        auto image_node  = std::dynamic_pointer_cast<opset::Constant>(node->input_value(1).get_node_shared_ptr());

        if (!layer_shape || !image_node) {
            THROW_IE_EXCEPTION << "Unsupported PriorBoxClustered with inconstant inputs";
        }

        auto data  = layer_shape->cast_vector<int>();
        auto image = image_node->cast_vector<int>();

        auto attrs = node->get_attrs();
        size_t num_priors = attrs.widths.size();

        auto variances = attrs.variances;
        if (variances.empty())
            variances.push_back(0.1f);

        // Execute
        const int64_t layer_width = data[1];
        const int64_t layer_height = data[0];

        int64_t img_width  = image[1];
        int64_t img_height = image[0];

        auto out_shape = ngraph::Shape{2, 4 * static_cast<unsigned int>(layer_height * layer_width * num_priors)};
        // TODO: Uncomment after PriorBoxClustered is aligned with the specification.

        //                int img_width = img_w_ == 0 ? image[1] : img_w_;
        //                int img_height = img_h_ == 0 ? image[0] : img_h_;

        //                float step_w = attrs.step_widths == 0 ? step_ : attrs.step_widths;
        //                float step_h = attrs.step_heights == 0 ? step_ :
        //                attrs.step_heights;

        float step_w = attrs.step_widths;
        float step_h = attrs.step_heights;

        if (step_w == 0 && step_h == 0) {
            step_w = static_cast<float>(img_width) / layer_width;
            step_h = static_cast<float>(img_height) / layer_height;
        }

        std::vector<float> dst_data(ngraph::shape_size(out_shape));
        size_t var_size = variances.size();
        for (int64_t h = 0; h < layer_height; ++h) {
            for (int64_t w = 0; w < layer_width; ++w) {
                float center_x = (w + attrs.offset) * step_w;
                float center_y = (h + attrs.offset) * step_h;

                for (size_t s = 0; s < num_priors; ++s) {
                    float box_width = attrs.widths[s];
                    float box_height = attrs.heights[s];

                    float xmin = (center_x - box_width / 2.0f) / img_width;
                    float ymin = (center_y - box_height / 2.0f) / img_height;
                    float xmax = (center_x + box_width / 2.0f) / img_width;
                    float ymax = (center_y + box_height / 2.0f) / img_height;

                    if (attrs.clip) {
                        xmin = std::min(std::max(xmin, 0.0f), 1.0f);
                        ymin = std::min(std::max(ymin, 0.0f), 1.0f);
                        xmax = std::min(std::max(xmax, 0.0f), 1.0f);
                        ymax = std::min(std::max(ymax, 0.0f), 1.0f);
                    }

                    auto get_idx = [&] (uint64_t cnt) -> uint64_t {
                        return h * layer_width * num_priors * cnt + w * num_priors * cnt + s * cnt;
                    };

                    uint64_t idx = get_idx(4);
                    dst_data[idx + 0] = xmin;
                    dst_data[idx + 1] = ymin;
                    dst_data[idx + 2] = xmax;
                    dst_data[idx + 3] = ymax;

                    idx = get_idx(var_size);
                    for (size_t j = 0; j < var_size; j++)
                        dst_data[idx + j + out_shape[1]] = variances[j];
                }
            }
        }

        auto output = std::make_shared<opset::Constant>(ngraph::element::f32, out_shape, dst_data.data());
        output->set_friendly_name(node->get_friendly_name());
        ngraph::copy_runtime_info(node, output);
        ngraph::replace_node(node, output);
        return true;
    };

    auto m = std::make_shared<ngraph::pattern::Matcher>(prior_box, "ConvertPriorBoxClustered");
    register_matcher(m, callback);
}
