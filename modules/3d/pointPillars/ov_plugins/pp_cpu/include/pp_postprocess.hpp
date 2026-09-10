// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#ifndef __PP_POSTPROCESS_HPP__
#define __PP_POSTPROCESS_HPP__

#include <openvino/op/op.hpp>

#include "pp_decoder_config_generated.hpp"

namespace PpExtension {

// Stage 1: score filter.
// Computes the best class score for each anchor and writes a selection bitmap.
// Input: classification logits [PP_ANCHOR_COUNT * PP_NUM_CLASSES] f32.
// Output: selection bitmap [PP_SELECT_WORDS] i32.
class PPPostprocScores : public ov::op::Op {
public:
    OPENVINO_OP("PPPostprocScores");

    PPPostprocScores() = default;
    explicit PPPostprocScores(const ov::Output<ov::Node>& classification_logits);

    void validate_and_infer_types() override;
    std::shared_ptr<ov::Node> clone_with_new_inputs(const ov::OutputVector& new_args) const override;
    bool visit_attributes(ov::AttributeVisitor& visitor) override;
    bool evaluate(ov::TensorVector& outputs, const ov::TensorVector& inputs) const override;
    bool has_evaluate() const override;
};

// Stage 2: compact, decode and sort.
// Inputs:
//   selection bitmap [PP_SELECT_WORDS] i32
//   classification logits [PP_ANCHOR_COUNT * PP_NUM_CLASSES] f32
//   box regression [PP_ANCHOR_COUNT * PP_NUM_BOX_VALUES] f32
//   direction logits [PP_ANCHOR_COUNT * 2] f32
// Output: candidate records [PP_MAX_CANDIDATES + 1, PP_DET_CHANNELS] f32.
// The header row stores the candidate count as integer bits.
class PPPostprocCandidates : public ov::op::Op {
public:
    OPENVINO_OP("PPPostprocCandidates");

    PPPostprocCandidates() = default;
    PPPostprocCandidates(const ov::Output<ov::Node>& selection_bitmap,
                         const ov::Output<ov::Node>& classification_logits,
                         const ov::Output<ov::Node>& box_regression,
                         const ov::Output<ov::Node>& direction_logits);

    void validate_and_infer_types() override;
    std::shared_ptr<ov::Node> clone_with_new_inputs(const ov::OutputVector& new_args) const override;
    bool visit_attributes(ov::AttributeVisitor& visitor) override;
    bool evaluate(ov::TensorVector& outputs, const ov::TensorVector& inputs) const override;
    bool has_evaluate() const override;
};

// Stage 3: rotated-BEV IoU bitmask.
// Input: candidate records [PP_MAX_CANDIDATES + 1, PP_DET_CHANNELS] f32.
// Output: NMS mask [PP_MAX_CANDIDATES, PP_NMS_COL_BLOCKS] i32.
class PPPostprocNms : public ov::op::Op {
public:
    OPENVINO_OP("PPPostprocNms");

    PPPostprocNms() = default;
    explicit PPPostprocNms(const ov::Output<ov::Node>& candidate_records);

    void validate_and_infer_types() override;
    std::shared_ptr<ov::Node> clone_with_new_inputs(const ov::OutputVector& new_args) const override;
    bool visit_attributes(ov::AttributeVisitor& visitor) override;
    bool evaluate(ov::TensorVector& outputs, const ov::TensorVector& inputs) const override;
    bool has_evaluate() const override;
};

// Stage 4: greedy suppression.
// Inputs:
//   candidate records [PP_MAX_CANDIDATES + 1, PP_DET_CHANNELS] f32
//   NMS mask [PP_MAX_CANDIDATES, PP_NMS_COL_BLOCKS] i32
// Output: detections [PP_MAX_DETECTIONS, PP_DET_CHANNELS] f32.
// Rows after the detection count are zero.
class PPPostprocGather : public ov::op::Op {
public:
    OPENVINO_OP("PPPostprocGather");

    PPPostprocGather() = default;
    PPPostprocGather(const ov::Output<ov::Node>& candidate_records,
                     const ov::Output<ov::Node>& nms_mask);

    void validate_and_infer_types() override;
    std::shared_ptr<ov::Node> clone_with_new_inputs(const ov::OutputVector& new_args) const override;
    bool visit_attributes(ov::AttributeVisitor& visitor) override;
    bool evaluate(ov::TensorVector& outputs, const ov::TensorVector& inputs) const override;
    bool has_evaluate() const override;
};

}  // namespace PpExtension

#endif  // __PP_POSTPROCESS_HPP__
