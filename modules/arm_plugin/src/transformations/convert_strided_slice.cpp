// Copyright (C) 2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0


#include "transformations/convert_strided_slice.hpp"
#include "opset/opset.hpp"

#include <numeric>

#include <details/ie_exception.hpp>
#include <ngraph/rt_info.hpp>

using namespace ArmPlugin;

ArmPlugin::pass::ConvertStridedSlice::ConvertStridedSlice() : GraphRewrite() {
    auto input  = std::make_shared<ngraph::pattern::op::Label>(ngraph::element::f32, ngraph::Shape{1, 1, 1, 1});
    auto begin  = std::make_shared<ngraph::pattern::op::Label>(ngraph::element::i64, ngraph::Shape{1});
    auto end    = std::make_shared<ngraph::pattern::op::Label>(ngraph::element::i64, ngraph::Shape{1});
    auto stride = std::make_shared<ngraph::pattern::op::Label>(ngraph::element::i64, ngraph::Shape{1});
    auto slice  = std::make_shared<opset::StridedSlice>(input, begin, end, stride, std::vector<int64_t>{}, std::vector<int64_t>{});

    ngraph::graph_rewrite_callback callback = [](ngraph::pattern::Matcher& m) {
        auto slice = std::dynamic_pointer_cast<opset::StridedSlice>(m.get_match_root());
        if (!slice) {
            return false;
        }
        auto&& inputShape = slice->get_input_shape(0);
        auto&& dims       = inputShape.size();

        if (dims > 4) {
            THROW_IE_EXCEPTION << "Unsupported StridedSlice with " << dims << " dimensions.";
        }

        auto&& begin  = std::dynamic_pointer_cast<ngraph::op::Constant>(
                            slice->input(1).get_source_output().get_node_shared_ptr())->cast_vector<int64_t>();
        auto&& end    = std::dynamic_pointer_cast<ngraph::op::Constant>(
                            slice->input(2).get_source_output().get_node_shared_ptr())->cast_vector<int64_t>();
        auto&& stride = std::dynamic_pointer_cast<ngraph::op::Constant>(
                            slice->input(3).get_source_output().get_node_shared_ptr())->cast_vector<int64_t>();

        begin.resize(dims, 0);
        std::copy(inputShape.begin() + end.size(), inputShape.end(), std::back_inserter(end));
        stride.resize(dims, 1);

        auto ellipsisMask = slice->get_ellipsis_mask();
        auto beginMask    = slice->get_begin_mask();
        auto endMask      = slice->get_end_mask();
        auto newAxis      = slice->get_new_axis_mask();
        auto shrinkAxis   = slice->get_shrink_axis_mask();

        ellipsisMask.resize(dims, 0);
        beginMask.resize(dims, 0);
        endMask.resize(dims, 0);
        newAxis.resize(dims, 0);
        shrinkAxis.resize(dims, 0);

        bool addOrReduceDims = std::find(newAxis.begin(), newAxis.end(), 1) != newAxis.end() ||
                               std::find(shrinkAxis.begin(), shrinkAxis.end(), 1) != shrinkAxis.end();

        if (addOrReduceDims && std::find(ellipsisMask.begin(), ellipsisMask.end(), 1) != ellipsisMask.end()) {
            THROW_IE_EXCEPTION << "Unsupported StridedSlice with ellipsis_mask and new_axis_mask or shrink_axis_mask";
        }

        for (size_t i = 0; i < begin.size(); i++) {
            if (beginMask[i] == 1 || ellipsisMask[i] == 1) {
                begin[i] = stride[i] > 0 ? 0 : inputShape[i];
            }
            if (endMask[i] == 1 || ellipsisMask[i] == 1) {
                end[i] = stride[i] > 0 ? inputShape[i] : -(begin[i] + 2);
            }
        }

        std::vector<int64_t> new_begin = begin, new_end = end, new_stride = stride;
        if (addOrReduceDims) {
            std::vector<int> shift;
            int cum_shift = 0;
            // calculating shift for begin and end according to newAxisMask
            for (size_t i = 0; i < newAxis.size(); ++i) {
                if (newAxis[i] == 1) {
                    ++cum_shift;
                    shrinkAxis[i] = 0; // shrinkAxis ignored if set newAxis
                } else {
                    shift.push_back(cum_shift);
                }
            }
            shift.resize(inputShape.size(), cum_shift);
            begin.resize(inputShape.size() + cum_shift, 0);
            end.resize(inputShape.size() + cum_shift, std::numeric_limits<std::int64_t>::max());
            stride.resize(inputShape.size() + cum_shift, 1);

            // if shrinkAxis[i] == 1, then we squeeze corresponding dimension to 1
            // and remove it in reshape after slice
            for (size_t i = 0; i < begin.size(); ++i) {
                if (shrinkAxis[i] == 1) {
                    if (begin[i] < 0) {
                        begin[i] = inputShape[i] + begin[i];
                    }
                    end[i]    = begin[i] + 1;
                    stride[i] = 1;
                }
            }
            for (size_t i = 0; i < inputShape.size(); ++i) {
                new_begin[i]  = begin[i + shift[i]];
                new_end[i]    = std::min(end[i + shift[i]], (int64_t)inputShape[i]);
                new_stride[i] = stride[i + shift[i]];
            }
        }
        auto begin_node  = opset::Constant::create<int64_t>(ngraph::element::i64, ngraph::Shape{dims}, new_begin);
        auto end_node    = opset::Constant::create<int64_t>(ngraph::element::i64, ngraph::Shape{dims}, new_end);
        auto stride_node = opset::Constant::create<int64_t>(ngraph::element::i64, ngraph::Shape{dims}, new_stride);

        auto optimized_slice = std::make_shared<opset::StridedSlice>(slice->input(0).get_source_output(),
                                     begin_node, end_node, stride_node, std::vector<int64_t>{}, std::vector<int64_t>{});
        auto shape = std::make_shared<ngraph::op::Constant>(ngraph::element::i64,
                       ngraph::Shape{slice->get_shape().size()}, std::vector<int64_t>(slice->get_shape().begin(), slice->get_shape().end()));
        auto reshape = std::make_shared<ngraph::op::v1::Reshape>(optimized_slice, shape, true);

        reshape->set_friendly_name(slice->get_friendly_name());
        ngraph::copy_runtime_info(slice, {optimized_slice, reshape});
        ngraph::replace_node(slice, reshape);
        return true;
    };

    auto m = std::make_shared<ngraph::pattern::Matcher>(slice, "ConvertStridedSlice");
    this->add_matcher(m, callback);
}
