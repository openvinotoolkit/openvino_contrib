// Copyright (C) 2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include <ngraph/rt_info.hpp>
#include <details/ie_exception.hpp>

#include "opset/opset.hpp"
#include "transformations/convert_reduce_single_axis.hpp"


using namespace ArmPlugin;

template<typename T>
static auto addMatcher(ArmPlugin::pass::ConvertReduceSingleAxis* pass) {
    auto reduce = std::make_shared<T>(ngraph::pattern::any_input(), ngraph::pattern::any_input());

    auto m = std::make_shared<ngraph::pattern::Matcher>(reduce, ("Convert" + std::string {T::type_info.name}).c_str());
        pass->add_matcher(m, [] (ngraph::pattern::Matcher& m) {
        auto reduce = m.get_match_root();
        if (!std::dynamic_pointer_cast<T>(reduce)) {
            return false;
        }

        if (ngraph::shape_size(reduce->input_value(1).get_shape()) <= 1) {
            return false;
        }

        auto reduction_axes = std::dynamic_pointer_cast<opset::Constant>(reduce->input_value(1).get_node_shared_ptr());
        if (!reduction_axes) {
            THROW_IE_EXCEPTION << "Reduce op only supports constant multiple reduction axes.";
        }
        auto axes = reduction_axes->cast_vector<int64_t>();
        ngraph::NodeVector new_ops;
        std::shared_ptr<ngraph::Node> node = reduce->input_value(0).get_node_shared_ptr();
        for (auto axis : axes) {
            auto reduction_axis = opset::Constant::create<int64_t>(ngraph::element::i64, ngraph::Shape{}, {axis});
            node = std::make_shared<T>(node, reduction_axis, true);
            new_ops.push_back(node);
        }

        auto out_shape = reduce->get_output_shape(0);
        auto dst_shape = std::make_shared<opset::Constant>(ngraph::element::i64, ngraph::Shape{out_shape.size()},
                    std::vector<int64_t>(out_shape.begin(), out_shape.end()));
        auto reshape = std::make_shared<opset::Reshape>(node, dst_shape, true);

        reshape->set_friendly_name(reduce->get_friendly_name());
        ngraph::copy_runtime_info(reduce, new_ops);
        ngraph::replace_node(reduce, reshape);
        return true;
    });
}

ArmPlugin::pass::ConvertReduceSingleAxis::ConvertReduceSingleAxis() : GraphRewrite() {
    addMatcher<opset::ReduceProd>(this);
    addMatcher<opset::ReduceMin>(this);
}
