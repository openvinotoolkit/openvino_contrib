// Copyright (C) 2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "ngraph_opset.hpp"
#include "utils.hpp"

namespace ArmPlugin {
namespace opset {

namespace util {
class PoolBase : public ov::op::Op {
public:
    OPENVINO_OP("PoolBase", "arm_opset_util");
    PoolBase(const ov::Output<Node>& arg,
            const ov::Strides& strides,
            const ov::Shape& pads_begin,
            const ov::Shape& pads_end,
            const ov::Shape& kernel,
            const ov::op::RoundingType& rounding_type = ov::op::RoundingType::FLOOR,
            const ov::op::PadType& auto_pad = ov::op::PadType::EXPLICIT,
            const DataLayout& layout = DataLayout::NCHW);

    const ov::Strides& get_strides() const {
        return m_strides;
    }

    void set_strides(const ov::Strides& strides) {
        m_strides = strides;
    }

    const ov::Shape& get_pads_begin() const {
        return m_pads_begin;
    }

    void set_pads_begin(const ov::Shape& pads_begin) {
        m_pads_begin = pads_begin;
    }

    const ov::Shape& get_pads_end() const {
        return m_pads_end;
    }

    void set_pads_end(const ov::Shape& pads_end) {
        m_pads_end = pads_end;
    }

    const ov::Shape& get_kernel() const {
        return m_kernel;
    }

    void set_kernel(const ov::Shape& kernel) {
        m_kernel = kernel;
    }

    const ov::op::RoundingType& get_rounding_type() const {
        return m_rounding_type;
    }

    void set_rounding_type(const ov::op::RoundingType& rounding_type) {
        m_rounding_type = rounding_type;
    }

    const ov::op::PadType& get_auto_pad() const {
        return m_auto_pad;
    }

    void set_auto_pad(const ov::op::PadType& auto_pad) {
        m_auto_pad = auto_pad;
    }

    const DataLayout& get_layout() const {
        return m_layout;
    }

    void set_layout(const DataLayout& layout) {
        m_layout = layout;
    }

protected:
    ov::Strides m_strides;
    ov::Shape m_pads_begin;
    ov::Shape m_pads_end;
    ov::Shape m_kernel;
    ov::op::RoundingType m_rounding_type;
    ov::op::PadType m_auto_pad;
    DataLayout m_layout;

    ov::PartialShape infer_shape(const ov::Strides& dilations, bool exclude_pad);
};
} // namespace util

namespace v1 {
class ArmMaxPool : public util::PoolBase {
public:
    OPENVINO_OP("ArmMaxPool", "arm_opset", util::PoolBase, 1);
    ArmMaxPool(const ov::Output<Node>& arg,
               const ov::Strides& strides,
               const ov::Shape& pads_begin,
               const ov::Shape& pads_end,
               const ov::Shape& kernel,
               const ov::op::RoundingType& rounding_type = ov::op::RoundingType::FLOOR,
               const ov::op::PadType& auto_pad = ov::op::PadType::EXPLICIT,
               const DataLayout& layout = DataLayout::NCHW);

    std::shared_ptr<ov::Node> clone_with_new_inputs(const ov::OutputVector& new_args) const override;
    void validate_and_infer_types() override;
};


class ArmAvgPool : public util::PoolBase {
public:
    OPENVINO_OP("ArmAvgPool", "arm_opset", util::PoolBase, 1);
    ArmAvgPool(const ov::Output<Node>& arg,
               const ov::Strides& strides,
               const ov::Shape& pads_begin,
               const ov::Shape& pads_end,
               const ov::Shape& kernel,
               bool exclude_pad,
               const ov::op::RoundingType& rounding_type = ov::op::RoundingType::FLOOR,
               const ov::op::PadType& auto_pad = ov::op::PadType::EXPLICIT,
               const DataLayout& layout = DataLayout::NCHW);

    std::shared_ptr<ov::Node> clone_with_new_inputs(const ov::OutputVector& new_args) const override;
    void validate_and_infer_types() override;

    const bool get_exclude_pad() const {
        return m_exclude_pad;
    }

    void set_exclude_pad(const bool exclude_pad) {
        m_exclude_pad = exclude_pad;
    }

private:
    bool m_exclude_pad;
};
} // namespace v1

namespace v8 {
class ArmMaxPool : public util::PoolBase {
public:
    OPENVINO_OP("ArmMaxPool", "arm_opset", util::PoolBase, 8);
    ArmMaxPool(const ov::Output<Node>& arg,
               const ov::Strides& strides,
               const ov::Strides& dilations,
               const ov::Shape& pads_begin,
               const ov::Shape& pads_end,
               const ov::Shape& kernel,
               const ov::op::RoundingType& rounding_type = ov::op::RoundingType::FLOOR,
               const ov::op::PadType& auto_pad = ov::op::PadType::EXPLICIT,
               const ov::element::Type& index_element_type = ov::element::i64,
               int64_t axis = 0,
               const DataLayout& layout = DataLayout::NCHW);

    std::shared_ptr<ov::Node> clone_with_new_inputs(const ov::OutputVector& new_args) const override;
    void validate_and_infer_types() override;

    const ov::Strides& get_dilations() const {
        return m_dilations;
    }

    void set_dilations(const ov::Strides& dilations) {
        m_dilations = dilations;
    }

    const ov::element::Type& get_index_element_type() const {
        return m_index_element_type;
    }

    void set_index_element_type(const ov::element::Type& index_element_type) {
        m_index_element_type = index_element_type;
    }

    const int64_t get_axis() const {
        return m_axis;
    }

    void set_axis(const int64_t axis) {
        m_axis = axis;
    }

private:
    ov::Strides m_dilations;
    ov::element::Type m_index_element_type;
    int64_t m_axis;
};
} // namespace v8

}  // namespace opset
}  // namespace ArmPlugin
