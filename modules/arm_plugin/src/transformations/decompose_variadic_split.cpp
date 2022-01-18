// Copyright (C) 2020-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0


#include "transformations/decompose_variadic_split.hpp"

#include "opset/opset.hpp"
#include <ngraph/rt_info.hpp>
#include <ngraph/pattern/op/wrap_type.hpp>

NGRAPH_RTTI_DEFINITION(ArmPlugin::pass::DecomposeVariadicSplit, "DecomposeVariadicSplit", 0);
ArmPlugin::pass::DecomposeVariadicSplit::DecomposeVariadicSplit() {
    auto split = ngraph::pattern::wrap_type<opset::VariadicSplit>();

    ngraph::matcher_pass_callback callback = [](ngraph::pattern::Matcher& m) {
        auto split = std::dynamic_pointer_cast<opset::VariadicSplit>(m.get_match_root());
        if (!split) {
            return false;
        }

        auto input = split->input_value(0).get_node_shared_ptr();
        auto axes = std::dynamic_pointer_cast<opset::Constant>(split->input_value(1).get_node_shared_ptr());
        auto split_lengths = std::dynamic_pointer_cast<opset::Constant>(split->input_value(2).get_node_shared_ptr());
        auto input_shape = input->get_shape();
        auto size = input_shape.size();


        if (!axes || !split_lengths) {
            IE_THROW() << "Unsupported VariadicSplit op with inconstant axes or split_lengths";
        }
        auto axis = axes->cast_vector<int64_t>()[0];

        axis = (axis < 0) ? (size + axis) : axis;

        auto splits = split_lengths->cast_vector<int64_t>();

        if (axis >= input_shape.size()) {
            IE_THROW() << "axis should be less than " << size;
        }

        auto stride = opset::Constant::create<int64_t>(ngraph::element::i64, ngraph::Shape{size}, std::vector<int64_t>(size, 1));
        std::vector<int64_t> begin_vec(size, 0);
        std::vector<int64_t> end_vec(input_shape.begin(), input_shape.end());
        end_vec[axis] = 0;

        ngraph::OutputVector slices;
        ngraph::NodeVector slice_nodes;
        std::string output_name = split->get_friendly_name();

        int count_useless_outs = 0;
        for (auto out : split->outputs()) {
            auto inputs = out.get_target_inputs();
            if (inputs.empty()) {
                ++count_useless_outs;
            }
        }

        for (size_t i = 0; i < splits.size(); i++) {
            begin_vec[axis] = end_vec[axis];
            end_vec[axis]  += splits[i];

            auto begin  = opset::Constant::create<int64_t>(ngraph::element::i64, ngraph::Shape{size}, begin_vec);
            auto end    = opset::Constant::create<int64_t>(ngraph::element::i64, ngraph::Shape{size}, end_vec);
            auto slice  = std::make_shared<opset::StridedSlice>(input, begin, end, stride, std::vector<int64_t>{}, std::vector<int64_t>{});
            std::string name = output_name;
            if (splits.size() - count_useless_outs > 1) {
                name += '.' + std::to_string(i);
            }
            slice->set_friendly_name(name);
            slice_nodes.push_back(slice);
            slices.push_back(slice->output(0));
        }

        ngraph::copy_runtime_info(split, slice_nodes);
        ngraph::replace_node(split, slices);
        return true;
    };

    auto m = std::make_shared<ngraph::pattern::Matcher>(split, "DecomposeVariadicSplit");
    register_matcher(m, callback);
}
