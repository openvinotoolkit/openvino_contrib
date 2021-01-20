// Copyright (C) 2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0


#include "transformations/convert_prior_box_to_const.hpp"

#include <details/ie_exception.hpp>
#include <ngraph/rt_info.hpp>
#include "opset/opset.hpp"

ArmPlugin::pass::ConvertPriorBox::ConvertPriorBox() : GraphRewrite() {
    auto prior_box = std::make_shared<opset::PriorBox>(ngraph::pattern::any_input(),
                                                       ngraph::pattern::any_input(),
                                                       ngraph::op::PriorBoxAttrs{});

    ngraph::graph_rewrite_callback callback = [](ngraph::pattern::Matcher& m) {
        auto node = std::dynamic_pointer_cast<opset::PriorBox>(m.get_match_root());
        if (!node) {
            return false;
        }

        auto data_node  = std::dynamic_pointer_cast<opset::Constant>(node->input_value(0).get_node_shared_ptr());
        auto image_node = std::dynamic_pointer_cast<opset::Constant>(node->input_value(1).get_node_shared_ptr());

        if (!data_node || !image_node) {
            THROW_IE_EXCEPTION << "Unsupported PriorBox with inconstant inputs";
        }

        auto data  = data_node->cast_vector<int>();
        auto image = image_node->cast_vector<int>();

        const int64_t layer_width  = data[1];
        const int64_t layer_height = data[0];
        const int64_t IW = image[1];
        const int64_t IH = image[0];

        auto attrs = node->get_attrs();
        int64_t num_priors = ngraph::op::PriorBox::number_of_priors(attrs);
        auto out_shape = ngraph::Shape{2, 4 * static_cast<unsigned int>(layer_height * layer_width * num_priors)};
        const int64_t OH = out_shape[1];
        const int64_t OW = 1;

        std::vector<float> aspect_ratios = {1.0f};
        for (const auto& aspect_ratio : attrs.aspect_ratio) {
            bool exist = false;
            for (const auto existed_value : aspect_ratios)
                exist |= std::fabs(aspect_ratio - existed_value) < 1e-6;

            if (!exist) {
                aspect_ratios.push_back(aspect_ratio);
                if (attrs.flip) {
                    aspect_ratios.push_back(1.0f / aspect_ratio);
                }
            }
        }

        std::vector<float> variance = attrs.variance;
        IE_ASSERT(variance.size() == 1 || variance.size() == 4 || variance.empty());
        if (variance.empty())
            variance.push_back(0.1f);


        float step = attrs.step;
        auto min_size = attrs.min_size;
        if (!attrs.scale_all_sizes) {
            // mxnet-like PriorBox
            if (step == -1)
                step = 1.f * IH / layer_height;
            else
                step *= IH;
            for (auto& size : min_size)
                size *= IH;
        }

        int64_t idx = 0;
        float center_x, center_y, box_width, box_height, step_x, step_y;
        float IWI = 1.0f / static_cast<float>(IW);
        float IHI = 1.0f / static_cast<float>(IH);

        if (step == 0) {
            step_x = static_cast<float>(IW) / layer_width;
            step_y = static_cast<float>(IH) / layer_height;
        } else {
            step_x = step;
            step_y = step;
        }

        std::vector<float> dst_data(ngraph::shape_size(out_shape));
        auto calculate_data = [&dst_data, &IWI, &IHI, &idx](
            float center_x, float center_y, float box_width, float box_height, bool clip) {
            if (clip) {
                // order: xmin, ymin, xmax, ymax
                dst_data[idx++] = std::max((center_x - box_width) * IWI, 0.f);
                dst_data[idx++] = std::max((center_y - box_height) * IHI, 0.f);
                dst_data[idx++] = std::min((center_x + box_width) * IWI, 1.f);
                dst_data[idx++] = std::min((center_y + box_height) * IHI, 1.f);
            } else {
                dst_data[idx++] = (center_x - box_width) * IWI;
                dst_data[idx++] = (center_y - box_height) * IHI;
                dst_data[idx++] = (center_x + box_width) * IWI;
                dst_data[idx++] = (center_y + box_height) * IHI;
            }
        };

        for (int64_t h = 0; h < layer_height; ++h) {
            for (int64_t w = 0; w < layer_width; ++w) {
                if (step == 0) {
                    center_x = (w + 0.5f) * step_x;
                    center_y = (h + 0.5f) * step_y;
                } else {
                    center_x = (attrs.offset + w) * step;
                    center_y = (attrs.offset + h) * step;
                }

                for (size_t s = 0; s < attrs.fixed_size.size(); ++s) {
                    auto fixed_size_ = static_cast<size_t>(attrs.fixed_size[s]);
                    box_width = box_height = fixed_size_ * 0.5f;

                    if (!attrs.fixed_ratio.empty()) {
                        for (float ar : attrs.fixed_ratio) {
                            auto density_ = static_cast<int64_t>(attrs.density[s]);
                            auto shift =
                                static_cast<int64_t>(attrs.fixed_size[s] / density_);
                            ar = std::sqrt(ar);
                            float box_width_ratio = attrs.fixed_size[s] * 0.5f * ar;
                            float box_height_ratio = attrs.fixed_size[s] * 0.5f / ar;
                            for (size_t r = 0; r < density_; ++r) {
                                for (size_t c = 0; c < density_; ++c) {
                                    float center_x_temp = center_x - fixed_size_ / 2 +
                                                            shift / 2.f + c * shift;
                                    float center_y_temp = center_y - fixed_size_ / 2 +
                                                            shift / 2.f + r * shift;
                                    calculate_data(center_x_temp,
                                                    center_y_temp,
                                                    box_width_ratio,
                                                    box_height_ratio,
                                                    true);
                                }
                            }
                        }
                    } else {
                        if (!attrs.density.empty()) {
                            auto density_ = static_cast<int64_t>(attrs.density[s]);
                            auto shift = static_cast<int64_t>(attrs.fixed_size[s] / density_);
                            for (int64_t r = 0; r < density_; ++r) {
                                for (int64_t c = 0; c < density_; ++c) {
                                    float center_x_temp = center_x - fixed_size_ / 2 + shift / 2.f + c * shift;
                                    float center_y_temp = center_y - fixed_size_ / 2 + shift / 2.f + r * shift;
                                    calculate_data(center_x_temp,
                                                    center_y_temp,
                                                    box_width,
                                                    box_height,
                                                    true);
                                }
                            }
                        }
                        //  Rest of priors
                        for (float ar : aspect_ratios) {
                            if (fabs(ar - 1.) < 1e-6) {
                                continue;
                            }

                            auto density_ = static_cast<int64_t>(attrs.density[s]);
                            auto shift = static_cast<int64_t>(attrs.fixed_size[s] / density_);
                            ar = std::sqrt(ar);
                            float box_width_ratio = attrs.fixed_size[s] * 0.5f * ar;
                            float box_height_ratio = attrs.fixed_size[s] * 0.5f / ar;
                            for (int64_t r = 0; r < density_; ++r) {
                                for (int64_t c = 0; c < density_; ++c) {
                                    float center_x_temp = center_x - fixed_size_ / 2 +
                                                            shift / 2.f + c * shift;
                                    float center_y_temp = center_y - fixed_size_ / 2 +
                                                            shift / 2.f + r * shift;
                                    calculate_data(center_x_temp,
                                                    center_y_temp,
                                                    box_width_ratio,
                                                    box_height_ratio,
                                                    true);
                                }
                            }
                        }
                    }
                }
                for (size_t ms_idx = 0; ms_idx < min_size.size(); ms_idx++) {
                    box_width = min_size[ms_idx] * 0.5f;
                    box_height = min_size[ms_idx] * 0.5f;
                    calculate_data(center_x, center_y, box_width, box_height, false);

                    if (attrs.max_size.size() > ms_idx) {
                        box_width = box_height = std::sqrt(min_size[ms_idx] * attrs.max_size[ms_idx]) * 0.5f;
                        calculate_data(center_x, center_y, box_width, box_height, false);
                    }

                    if (attrs.scale_all_sizes || (!attrs.scale_all_sizes && (ms_idx == min_size.size() - 1))) {
                        size_t s_idx = attrs.scale_all_sizes ? ms_idx : 0;
                        for (float ar : aspect_ratios) {
                            if (std::fabs(ar - 1.0f) < 1e-6) {
                                continue;
                            }

                            ar = std::sqrt(ar);
                            box_width = min_size[s_idx] * 0.5f * ar;
                            box_height = min_size[s_idx] * 0.5f / ar;
                            calculate_data(center_x, center_y, box_width, box_height, false);
                        }
                    }
                }
            }
        }

        if (attrs.clip) {
            for (uint64_t i = 0; i < layer_height * layer_width * num_priors * 4; ++i) {
                dst_data[i] = (std::min)((std::max)(dst_data[i], 0.0f), 1.0f);
            }
        }

        uint64_t channel_size = OH * OW;
        if (variance.size() == 1) {
            for (uint64_t i = 0; i < channel_size; ++i) {
                dst_data[i + channel_size] = variance[0];
            }
        } else {
            for (uint64_t i = 0; i < layer_height * layer_width * num_priors; ++i) {
                for (size_t j = 0; j < 4; ++j) {
                    dst_data[i * 4 + j + channel_size] = variance[j];
                }
            }
        }

        auto output = std::make_shared<opset::Constant>(ngraph::element::f32, out_shape, dst_data.data());
        output->set_friendly_name(node->get_friendly_name());
        ngraph::copy_runtime_info(node, output);
        ngraph::replace_node(node, output);
        return true;
    };

    auto m = std::make_shared<ngraph::pattern::Matcher>(prior_box, "ConvertPriorBox");
    this->add_matcher(m, callback, ngraph::pass::PassProperty::CHANGE_DYNAMIC_STATE);
}
