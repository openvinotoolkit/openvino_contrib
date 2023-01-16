// Copyright (C) 2020-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "interpolate_arm.hpp"
#include "ngraph/coordinate_index.hpp"

using namespace ngraph;
using namespace ArmPlugin;

opset::ArmInterpolate::ArmInterpolate(const ngraph::Output<ngraph::Node>& image,
                                      const ngraph::Output<ngraph::Node>& output_shape,
                                      const ngraph::Output<ngraph::Node>& scales,
                                      const Interpolate::InterpolateAttrs& attrs)
    : Interpolate{
        image,
        output_shape,
        scales,
        attrs}, m_attrs(attrs) {
    constructor_validate_and_infer_types();
}

opset::ArmInterpolate::ArmInterpolate(const ngraph::Output<ngraph::Node>& image,
                                      const ngraph::Output<ngraph::Node>& output_shape,
                                      const ngraph::Output<ngraph::Node>& scales,
                                      const ngraph::Output<ngraph::Node>& axes,
                                      const Interpolate::InterpolateAttrs& attrs)
    : Interpolate{
        image,
        output_shape,
        scales,
        axes,
        attrs}, m_attrs(attrs) {
    constructor_validate_and_infer_types();
}

std::shared_ptr<ngraph::Node> ArmPlugin::opset::ArmInterpolate::clone_with_new_inputs(const ngraph::OutputVector& new_args) const {
    auto num_args = new_args.size();
    if (num_args == 3) {
        return std::make_shared<ArmInterpolate>(new_args.at(0),
                                                new_args.at(1),
                                                new_args.at(2),
                                                m_attrs);
    } else if (num_args == 4) {
        return std::make_shared<ArmInterpolate>(new_args.at(0),
                                                new_args.at(1),
                                                new_args.at(2),
                                                new_args.at(3),
                                                m_attrs);
    } else {
        throw ngraph_error("Unsupported number of arguments for ArmInterpolate operation");
    }
}

namespace {
    Strides default_strides(size_t n_axes) {
        return Strides(n_axes, 1);
    }
    CoordinateDiff default_padding(size_t n_axes) {
        return CoordinateDiff(n_axes, 0);
    }
    AxisVector default_axis_order(size_t n_axes) {
        AxisVector result(n_axes);
        std::iota(result.begin(), result.end(), 0);
        return result;
    }

    Coordinate default_source_start_corner(size_t n_axes) {
        return Coordinate(n_axes, 0);
    }
    Coordinate default_source_end_corner(const Shape& source_shape) {
        return source_shape;
    }
}  // namespace

ArmPlugin::opset::CoordinateTransform::CoordinateTransform(const Shape& source_shape,
                                         const Coordinate& source_start_corner,
                                         const Coordinate& source_end_corner,
                                         const Strides& source_strides,
                                         const AxisVector& source_axis_order,
                                         const CoordinateDiff& target_padding_below,
                                         const CoordinateDiff& target_padding_above,
                                         const Strides& target_dilation_strides)
        : CoordinateTransformBasic(source_shape),
          m_source_start_corner(source_start_corner),
          m_source_end_corner(source_end_corner),
          m_source_strides(source_strides),
          m_source_axis_order(source_axis_order),
          m_target_padding_below(target_padding_below),
          m_target_padding_above(target_padding_above),
          m_target_dilation_strides(target_dilation_strides) {
    m_n_axes = source_shape.size();

    if (m_n_axes != source_start_corner.size()) {
        throw std::domain_error("Source start corner does not have the same number of axes as the source space shape");
    }
    if (m_n_axes != source_end_corner.size()) {
        throw std::domain_error("Source end corner does not have the same number of axes as the source space shape");
    }
    if (m_n_axes != source_strides.size()) {
        throw std::domain_error("Source strides do not have the same number of axes as the source space shape");
    }
    if (m_n_axes != source_axis_order.size()) {
        // Note: this check is NOT redundant with the is_permutation check below, though you might
        // think it is. If the lengths don't match then is_permutation won't catch that; it'll
        // either stop short or walk off the end of source_axis_order.
        throw std::domain_error("Source axis order does not have the same number of axes as the source space shape");
    }
    if (m_n_axes != target_padding_below.size()) {
        throw std::domain_error("Padding-below shape does not have the same number of axes as the source space shape");
    }
    if (m_n_axes != target_padding_above.size()) {
        throw std::domain_error("Padding-above shape does not have the same number of axes as the source space shape");
    }
    if (m_n_axes != target_dilation_strides.size()) {
        throw std::domain_error("Target dilation strides do not have the same number of axes as the source shape");
    }

    AxisVector all_axes(m_n_axes);
    for (size_t i = 0; i < all_axes.size(); i++) {
        all_axes[i] = i;
    }

    if (!std::is_permutation(all_axes.begin(), all_axes.end(), source_axis_order.begin())) {
        throw std::domain_error("Source axis order is not a permutation of {0,...,n-1} where n is the number of axes "
                                "in the source space shape");
    }

    for (size_t i = 0; i < m_n_axes; i++) {
        if (target_dilation_strides[i] == 0) {
            std::stringstream ss;

            ss << "The target dilation stride is 0 at axis " << i;
            throw std::domain_error(ss.str());
        }
    }

    std::vector<std::ptrdiff_t> padded_upper_bounds;

    for (size_t i = 0; i < m_n_axes; i++) {
        NGRAPH_SUPPRESS_DEPRECATED_START
        std::ptrdiff_t padded_upper_bound = subtract_or_zero(source_shape[i], size_t(1)) * target_dilation_strides[i] +
                                            1 + target_padding_below[i] + target_padding_above[i];
        NGRAPH_SUPPRESS_DEPRECATED_END

        if (padded_upper_bound < 0) {
            std::stringstream ss;

            ss << "The end corner is out of bounds at axis " << i;
            throw std::domain_error(ss.str());
        }

        padded_upper_bounds.push_back(padded_upper_bound);
    }

    for (size_t i = 0; i < m_n_axes; i++) {
        if (static_cast<int64_t>(source_start_corner[i]) >= padded_upper_bounds[i] &&
            source_start_corner[i] != source_shape[i]) {
            std::stringstream ss;

            ss << "The start corner is out of bounds at axis " << i;
            throw std::domain_error(ss.str());
        }

        if (static_cast<int64_t>(source_end_corner[i]) > padded_upper_bounds[i]) {
            std::stringstream ss;

            ss << "The end corner is out of bounds at axis " << i;
            throw std::domain_error(ss.str());
        }
    }

    for (size_t i = 0; i < m_n_axes; i++) {
        if (source_strides[i] == 0) {
            std::stringstream ss;

            ss << "The source stride is 0 at axis " << i;
            throw std::domain_error(ss.str());
        }
    }

    for (size_t axis = 0; axis < m_n_axes; axis++) {
        m_target_shape.push_back(
                ceil_div(source_end_corner[source_axis_order[axis]] - source_start_corner[source_axis_order[axis]],
                         source_strides[source_axis_order[axis]]));
    }
}

ArmPlugin::opset::CoordinateTransform::CoordinateTransform(const Shape& source_shape)
        : CoordinateTransform(source_shape,
                              default_source_start_corner(source_shape.size()),
                              default_source_end_corner(source_shape),
                              default_strides(source_shape.size()),
                              default_axis_order(source_shape.size()),
                              default_padding(source_shape.size()),
                              default_padding(source_shape.size()),
                              default_strides(source_shape.size())) {}

// Compute the index of a target-space coordinate in thebuffer.
size_t ArmPlugin::opset::CoordinateTransform::index(const Coordinate& c) const {
    return coordinate_index(to_source_coordinate(c), m_source_shape);
}

// Convert a target-space coordinate to a source-space coordinate.
Coordinate ArmPlugin::opset::CoordinateTransform::to_source_coordinate(const Coordinate& c_target) const {
    if (c_target.size() != m_n_axes) {
        throw std::domain_error("Target coordinate rank does not match the coordinate transform rank");
    }

    Coordinate c_source(c_target.size());

    for (size_t target_axis = 0; target_axis < m_n_axes; target_axis++) {
        size_t source_axis = m_source_axis_order[target_axis];

        size_t target_pos = c_target[target_axis];
        size_t pos_destrided = target_pos * m_source_strides[source_axis];
        size_t pos_deshifted = pos_destrided + m_source_start_corner[source_axis];
        size_t pos_depadded = pos_deshifted - m_target_padding_below[target_axis];
        size_t pos_dedilated = pos_depadded / m_target_dilation_strides[target_axis];
        c_source[source_axis] = pos_dedilated;
    }

    return c_source;
}

CoordinateIterator ArmPlugin::opset::CoordinateTransform::begin() const noexcept {
    return CoordinateIterator(m_target_shape);
}

const CoordinateIterator& ArmPlugin::opset::CoordinateTransform::end() const noexcept {
    return CoordinateIterator::end();
}
