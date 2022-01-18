// Copyright (C) 2020-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0


#include "transformations/convert_tile_to_concats.hpp"
#include "opset/opset.hpp"

#include <numeric>

#include <ngraph/rt_info.hpp>
#include <ngraph/pattern/op/wrap_type.hpp>

NGRAPH_RTTI_DEFINITION(ArmPlugin::pass::ConvertTile, "ConvertTile", 0);
ArmPlugin::pass::ConvertTile::ConvertTile() {
    auto tile = ngraph::pattern::wrap_type<opset::Tile>();

    ngraph::matcher_pass_callback callback = [](ngraph::pattern::Matcher& m) {
        auto tile = std::dynamic_pointer_cast<opset::Tile>(m.get_match_root());
        if (!tile) {
            return false;
        }

        auto repeat_node = std::dynamic_pointer_cast<opset::Constant>(tile->input_value(1).get_node_shared_ptr());
        if (!repeat_node) {
            IE_THROW() << "Unsupported Tile with inconstant repeats.";
        }

        ngraph::NodeVector new_ops;
        auto input = tile->input_value(0).get_node_shared_ptr();
        auto repeats = repeat_node->cast_vector<int64_t>();

        std::vector<size_t> input_shape = tile->get_input_shape(0);
        size_t output_rank = std::max(input_shape.size(), repeats.size());
        repeats.insert(repeats.begin(), output_rank - repeats.size(), 1);
        input_shape.insert(input_shape.begin(), output_rank - input_shape.size(), 1);

        auto shape = std::make_shared<opset::Constant>(ngraph::element::i64,
                       ngraph::Shape{input_shape.size()}, std::vector<int64_t>(input_shape.begin(), input_shape.end()));
        input = std::make_shared<opset::Reshape>(input, shape, true);
        new_ops.push_back(input);
        for (size_t axis = 0; axis < repeats.size(); axis++) {
            input = std::make_shared<opset::Concat>(ngraph::NodeVector(repeats[axis], input), axis);
            new_ops.push_back(input);
        }

        auto last_node = new_ops.back();
        last_node->set_friendly_name(tile->get_friendly_name());
        ngraph::copy_runtime_info(tile, new_ops);
        ngraph::replace_node(tile, last_node);
        return true;
    };

    auto m = std::make_shared<ngraph::pattern::Matcher>(tile, "ConvertTile");
    register_matcher(m, callback);
}
