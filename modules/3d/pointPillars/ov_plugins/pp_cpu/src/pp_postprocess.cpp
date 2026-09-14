// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

// CPU implementation of detection-head postprocessing.

#include "pp_postprocess.hpp"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <vector>

namespace PpExtension {
namespace {

enum AnchorField : int {
    kAnchorDx = 0,
    kAnchorDy,
    kAnchorDz,
    kAnchorRotation,
};

enum BoxField : int {
    kBoxX = 0,
    kBoxY,
    kBoxZ,
    kBoxLength,
    kBoxWidth,
    kBoxHeight,
    kBoxYaw,
};

enum DetectionField : int {
    kDetectionX = 0,
    kDetectionY,
    kDetectionZ,
    kDetectionLength,
    kDetectionWidth,
    kDetectionHeight,
    kDetectionYaw,
};

enum DirectionClass : int {
    kForwardDirection = 0,
    kBackwardDirection,
};

constexpr int kFirstClassIndex = 0;
constexpr int kFirstCornerIndex = 0;
constexpr float kHalfExtent = 0.5f;
constexpr float kBoxContainmentMargin = 1e-2f;
constexpr float kIntersectionEpsilon = 1e-8f;
constexpr double kDirectionWrapEpsilon = 1e-8;
constexpr float kFullTurnRadians = 2.0f * static_cast<float>(M_PI);
constexpr int kRectangleCornerCount = 4;
constexpr int kClosedRectangleCornerCount = kRectangleCornerCount + 1;
constexpr int kIntersectionPointCapacity = 16;
constexpr int kCandidateHeaderRows = 1;
constexpr int kSkipCurrentCandidate = 1;

enum ScoresInputPort : int {
    kScoresClassificationInput = 0,
};

enum CandidatesInputPort : int {
    kCandidatesSelectionInput = 0,
    kCandidatesClassificationInput,
    kCandidatesBoxInput,
    kCandidatesDirectionInput,
};

enum NmsInputPort : int {
    kNmsCandidatesInput = 0,
};

enum GatherInputPort : int {
    kGatherCandidatesInput = 0,
    kGatherNmsMaskInput,
};

constexpr int kOutputPort = 0;
constexpr size_t kOutputCount = 1;
constexpr size_t kScoresInputCount = 1;
constexpr size_t kCandidatesInputCount = 4;
constexpr size_t kNmsInputCount = 1;
constexpr size_t kGatherInputCount = 2;

float sigmoid(float x) { return 1.0f / (1.0f + std::exp(-x)); }

// Return the largest class sigmoid and its class index for one anchor.
float anchor_score(const float* classification_logits, int64_t anchor, int* class_id) {
    const float* scores = classification_logits + anchor * PP_NUM_CLASSES;
    float best = sigmoid(scores[kFirstClassIndex]);
    int best_id = kFirstClassIndex;
    for (int class_index = kFirstClassIndex + 1; class_index < PP_NUM_CLASSES; ++class_index) {
        const float value = sigmoid(scores[class_index]);
        if (value > best) {
            best = value;
            best_id = class_index;
        }
    }
    if (class_id) *class_id = best_id;
    return best;
}

// Decode one surviving anchor into a detection record.
void decode_anchor(const float* box_regression, const float* direction_logits, int64_t anchor, int class_id,
                   float score, float* detection_record) {
    const int64_t loc = anchor / PP_NUM_ANCHORS;
    const int64_t ith = anchor % PP_NUM_ANCHORS;
    const int64_t col = loc % PP_FEATURE_X;
    const int64_t row = loc / PP_FEATURE_X;

    // Place each anchor at the center of its feature-map cell.
    const float x_offset = PP_MIN_X + (col + kHalfExtent) * (PP_MAX_X - PP_MIN_X) / PP_FEATURE_X;
    const float y_offset = PP_MIN_Y + (row + kHalfExtent) * (PP_MAX_Y - PP_MIN_Y) / PP_FEATURE_Y;

    const float* anchor_ptr = PP_ANCHORS + ith * PP_ANCHOR_VALUES;
    const float z_offset = anchor_ptr[kAnchorDz] * kHalfExtent + PP_ANCHOR_BOTTOM_HEIGHTS[ith / PP_NUM_ROTATIONS];

    const float anchor_dx = anchor_ptr[kAnchorDx];
    const float anchor_dy = anchor_ptr[kAnchorDy];
    const float anchor_dz = anchor_ptr[kAnchorDz];
    const float anchor_rotation = anchor_ptr[kAnchorRotation];
    const float diagonal = std::sqrt(anchor_dx * anchor_dx + anchor_dy * anchor_dy);

    const float* e = box_regression + anchor * PP_NUM_BOX_VALUES;
    const float bx = e[kBoxX] * diagonal + x_offset;
    const float by = e[kBoxY] * diagonal + y_offset;
    const float bz = e[kBoxZ] * anchor_dz + z_offset;
    const float bl = std::exp(e[kBoxLength]) * anchor_dx;
    const float bw = std::exp(e[kBoxWidth]) * anchor_dy;
    const float bh = std::exp(e[kBoxHeight]) * anchor_dz;
    const float br = e[kBoxYaw] + anchor_rotation;

    const float* direction_scores = direction_logits + anchor * PP_NUM_DIR_VALUES;
    const int direction_label = direction_scores[kForwardDirection] > direction_scores[kBackwardDirection]
                                    ? kForwardDirection
                                    : kBackwardDirection;
    const float period = kFullTurnRadians / PP_NUM_DIR_VALUES;
    const float val = br - PP_DIR_OFFSET;
    const float dir_rot =
        static_cast<float>(val - std::floor(val / (period + kDirectionWrapEpsilon)) * period);
    const float yaw = dir_rot + PP_DIR_OFFSET + period * direction_label;

    detection_record[kDetectionX] = bx;
    detection_record[kDetectionY] = by;
    detection_record[kDetectionZ] = bz;
    detection_record[kDetectionLength] = bl;
    detection_record[kDetectionWidth] = bw;
    detection_record[kDetectionHeight] = bh;
    detection_record[kDetectionYaw] = yaw;

    std::memcpy(&detection_record[PP_CLASS_INDEX], &class_id, sizeof(class_id));
    detection_record[PP_SCORE_INDEX] = score;
}

// Rotated BEV IoU.
struct Point2D {
    float x, y;
};

float cross(const Point2D& p1, const Point2D& p2, const Point2D& p0) {
    return (p1.x - p0.x) * (p2.y - p0.y) - (p2.x - p0.x) * (p1.y - p0.y);
}

int check_box2d(const float* detection_record, const Point2D& point) {
    const float center_x = detection_record[kDetectionX];
    const float center_y = detection_record[kDetectionY];
    const float angle_cos = std::cos(-detection_record[kDetectionYaw]);
    const float angle_sin = std::sin(-detection_record[kDetectionYaw]);
    const float rot_x = (point.x - center_x) * angle_cos + (point.y - center_y) * (-angle_sin);
    const float rot_y = (point.x - center_x) * angle_sin + (point.y - center_y) * angle_cos;
    const float half_length = detection_record[kDetectionLength] * kHalfExtent;
    const float half_width = detection_record[kDetectionWidth] * kHalfExtent;
    return std::fabs(rot_x) < half_length + kBoxContainmentMargin &&
           std::fabs(rot_y) < half_width + kBoxContainmentMargin;
}

bool intersection(const Point2D& p1, const Point2D& p0, const Point2D& q1, const Point2D& q0, Point2D& ans) {
        if (!(std::fmin(p0.x, p1.x) <= std::fmax(q0.x, q1.x) &&
            std::fmin(q0.x, q1.x) <= std::fmax(p0.x, p1.x) &&
            std::fmin(p0.y, p1.y) <= std::fmax(q0.y, q1.y) &&
            std::fmin(q0.y, q1.y) <= std::fmax(p0.y, p1.y)))
        return false;

    const float s1 = cross(q0, p1, p0);
    const float s2 = cross(p1, q1, p0);
    const float s3 = cross(p0, q1, q0);
    const float s4 = cross(q1, p1, q0);
    if (!(s1 * s2 > 0 && s3 * s4 > 0)) return false;

    const float s5 = cross(q1, p1, p0);
    if (std::fabs(s5 - s1) > kIntersectionEpsilon) {
        ans.x = (s5 * q0.x - s1 * q1.x) / (s5 - s1);
        ans.y = (s5 * q0.y - s1 * q1.y) / (s5 - s1);
    } else {
        const float a0 = p0.y - p1.y, b0 = p1.x - p0.x, c0 = p0.x * p1.y - p1.x * p0.y;
        const float a1 = q0.y - q1.y, b1 = q1.x - q0.x, c1 = q0.x * q1.y - q1.x * q0.y;
        const float determinant = a0 * b1 - a1 * b0;
        ans.x = (b0 * c1 - b1 * c0) / determinant;
        ans.y = (a1 * c0 - a0 * c1) / determinant;
    }
    return true;
}

void rotate_around_center(const Point2D& center, float angle_cos, float angle_sin, Point2D& p) {
    const float new_x = (p.x - center.x) * angle_cos + (p.y - center.y) * (-angle_sin) + center.x;
    const float new_y = (p.x - center.x) * angle_sin + (p.y - center.y) * angle_cos + center.y;
    p = Point2D{new_x, new_y};
}

float rotated_iou(const float* box_a, const float* box_b) {
    const float a_angle = box_a[kDetectionYaw], b_angle = box_b[kDetectionYaw];
    const float a_length_half = box_a[kDetectionLength] * kHalfExtent;
    const float b_length_half = box_b[kDetectionLength] * kHalfExtent;
    const float a_width_half = box_a[kDetectionWidth] * kHalfExtent;
    const float b_width_half = box_b[kDetectionWidth] * kHalfExtent;
    const float a_x1 = box_a[kDetectionX] - a_length_half;
    const float a_y1 = box_a[kDetectionY] - a_width_half;
    const float a_x2 = box_a[kDetectionX] + a_length_half;
    const float a_y2 = box_a[kDetectionY] + a_width_half;
    const float b_x1 = box_b[kDetectionX] - b_length_half;
    const float b_y1 = box_b[kDetectionY] - b_width_half;
    const float b_x2 = box_b[kDetectionX] + b_length_half;
    const float b_y2 = box_b[kDetectionY] + b_width_half;

    const Point2D center_a{box_a[kDetectionX], box_a[kDetectionY]};
    const Point2D center_b{box_b[kDetectionX], box_b[kDetectionY]};

    Point2D cross_points[kIntersectionPointCapacity];
    Point2D polygon_center{};
    int intersection_point_count = 0;

    Point2D box_a_corners[kClosedRectangleCornerCount] = {
        {a_x1, a_y1}, {a_x2, a_y1}, {a_x2, a_y2}, {a_x1, a_y2}, {}};
    Point2D box_b_corners[kClosedRectangleCornerCount] = {
        {b_x1, b_y1}, {b_x2, b_y1}, {b_x2, b_y2}, {b_x1, b_y2}, {}};

    const float a_angle_cos = std::cos(a_angle), a_angle_sin = std::sin(a_angle);
    const float b_angle_cos = std::cos(b_angle), b_angle_sin = std::sin(b_angle);
    for (int corner_index = 0; corner_index < kRectangleCornerCount; ++corner_index) {
        rotate_around_center(center_a, a_angle_cos, a_angle_sin, box_a_corners[corner_index]);
        rotate_around_center(center_b, b_angle_cos, b_angle_sin, box_b_corners[corner_index]);
    }
    box_a_corners[kRectangleCornerCount] = box_a_corners[kFirstCornerIndex];
    box_b_corners[kRectangleCornerCount] = box_b_corners[kFirstCornerIndex];

    for (int edge_a = 0; edge_a < kRectangleCornerCount; ++edge_a) {
        for (int edge_b = 0; edge_b < kRectangleCornerCount; ++edge_b) {
            if (intersection(box_a_corners[edge_a + 1], box_a_corners[edge_a], box_b_corners[edge_b + 1],
                             box_b_corners[edge_b], cross_points[intersection_point_count])) {
                polygon_center.x += cross_points[intersection_point_count].x;
                polygon_center.y += cross_points[intersection_point_count].y;
                intersection_point_count++;
            }
        }
    }

    for (int corner_index = 0; corner_index < kRectangleCornerCount; ++corner_index) {
        if (check_box2d(box_a, box_b_corners[corner_index])) {
            polygon_center.x += box_b_corners[corner_index].x;
            polygon_center.y += box_b_corners[corner_index].y;
            cross_points[intersection_point_count] = box_b_corners[corner_index];
            intersection_point_count++;
        }
        if (check_box2d(box_b, box_a_corners[corner_index])) {
            polygon_center.x += box_a_corners[corner_index].x;
            polygon_center.y += box_a_corners[corner_index].y;
            cross_points[intersection_point_count] = box_a_corners[corner_index];
            intersection_point_count++;
        }
    }

    polygon_center.x /= intersection_point_count;
    polygon_center.y /= intersection_point_count;

    for (int pass = 0; pass < intersection_point_count - 1; ++pass) {
        for (int point_index = 0; point_index < intersection_point_count - pass - 1; ++point_index) {
            if (std::atan2(cross_points[point_index].y - polygon_center.y,
                           cross_points[point_index].x - polygon_center.x) >
                std::atan2(cross_points[point_index + 1].y - polygon_center.y,
                           cross_points[point_index + 1].x - polygon_center.x)) {
                const Point2D temp = cross_points[point_index];
                cross_points[point_index] = cross_points[point_index + 1];
                cross_points[point_index + 1] = temp;
            }
        }
    }

    float area = 0.0f;
    for (int point_index = 0; point_index < intersection_point_count - 1; ++point_index) {
        const Point2D a{cross_points[point_index].x - cross_points[kFirstCornerIndex].x,
                   cross_points[point_index].y - cross_points[kFirstCornerIndex].y};
        const Point2D b{cross_points[point_index + 1].x - cross_points[kFirstCornerIndex].x,
                   cross_points[point_index + 1].y - cross_points[kFirstCornerIndex].y};
        area += (a.x * b.y - a.y * b.x);
    }

    const float overlap_area = std::fabs(area) * kHalfExtent;
    const float area_a = box_a[kDetectionLength] * box_a[kDetectionWidth];
    const float area_b = box_b[kDetectionLength] * box_b[kDetectionWidth];
    return overlap_area / std::fmax(area_a + area_b - overlap_area, kIntersectionEpsilon);
}

int32_t header_count(const ov::Tensor& candidate_records) {
    int32_t count = 0;
    std::memcpy(&count, candidate_records.data<const float>(), sizeof(int32_t));
    return std::clamp<int32_t>(count, 0, static_cast<int32_t>(PP_MAX_CANDIDATES));
}

}  // namespace

// ---------------------------------------------------------------------------
// Stage 1: score filter.
// ---------------------------------------------------------------------------
PPPostprocScores::PPPostprocScores(const ov::Output<ov::Node>& classification_logits)
    : Op({classification_logits}) {
    constructor_validate_and_infer_types();
}

void PPPostprocScores::validate_and_infer_types() {
    set_output_type(kOutputPort, ov::element::i32, ov::PartialShape{PP_SELECT_WORDS});
}

std::shared_ptr<ov::Node> PPPostprocScores::clone_with_new_inputs(const ov::OutputVector& new_args) const {
    OPENVINO_ASSERT(new_args.size() == 1, "PPPostprocScores expects 1 input");
    return std::make_shared<PPPostprocScores>(new_args[kScoresClassificationInput]);
}

bool PPPostprocScores::visit_attributes(ov::AttributeVisitor&) { return true; }

bool PPPostprocScores::has_evaluate() const { return true; }

bool PPPostprocScores::evaluate(ov::TensorVector& outputs, const ov::TensorVector& inputs) const {
    OPENVINO_ASSERT(inputs.size() == kScoresInputCount && outputs.size() == kOutputCount,
                    "PPPostprocScores expects 1 input and 1 output");
    outputs[kOutputPort].set_shape({static_cast<size_t>(PP_SELECT_WORDS)});

    const float* classification_logits = inputs[kScoresClassificationInput].data<const float>();
    auto* words = reinterpret_cast<uint32_t*>(outputs[kOutputPort].data<int32_t>());

    for (int64_t word = 0; word < PP_SELECT_WORDS; ++word) {
        uint32_t bits = 0;
        for (int bit = 0; bit < PP_SELECT_BITS; ++bit) {
            if (anchor_score(classification_logits, word * PP_SELECT_BITS + bit, nullptr) >= PP_SCORE_THRESH)
                bits |= 1u << bit;
        }
        words[word] = bits;
    }
    return true;
}

// ---------------------------------------------------------------------------
// Stage 2: compact, decode, and sort.
// ---------------------------------------------------------------------------
PPPostprocCandidates::PPPostprocCandidates(const ov::Output<ov::Node>& selection_bitmap,
                                           const ov::Output<ov::Node>& classification_logits,
                                           const ov::Output<ov::Node>& box_regression,
                                           const ov::Output<ov::Node>& direction_logits)
    : Op({selection_bitmap, classification_logits, box_regression, direction_logits}) {
    constructor_validate_and_infer_types();
}

void PPPostprocCandidates::validate_and_infer_types() {
    set_output_type(kOutputPort, ov::element::f32,
                    ov::PartialShape{PP_MAX_CANDIDATES + kCandidateHeaderRows, PP_DET_CHANNELS});
}

std::shared_ptr<ov::Node> PPPostprocCandidates::clone_with_new_inputs(const ov::OutputVector& new_args) const {
    OPENVINO_ASSERT(new_args.size() == 4, "PPPostprocCandidates expects 4 inputs");
    return std::make_shared<PPPostprocCandidates>(new_args[kCandidatesSelectionInput],
                                                  new_args[kCandidatesClassificationInput],
                                                  new_args[kCandidatesBoxInput],
                                                  new_args[kCandidatesDirectionInput]);
}

bool PPPostprocCandidates::visit_attributes(ov::AttributeVisitor&) { return true; }

bool PPPostprocCandidates::has_evaluate() const { return true; }

bool PPPostprocCandidates::evaluate(ov::TensorVector& outputs, const ov::TensorVector& inputs) const {
    OPENVINO_ASSERT(inputs.size() == kCandidatesInputCount && outputs.size() == kOutputCount,
                    "PPPostprocCandidates expects 4 inputs and 1 output");
    outputs[kOutputPort].set_shape(
        {static_cast<size_t>(PP_MAX_CANDIDATES + kCandidateHeaderRows), static_cast<size_t>(PP_DET_CHANNELS)});

    const auto* selection_words =
        reinterpret_cast<const uint32_t*>(inputs[kCandidatesSelectionInput].data<const int32_t>());
    const float* classification_logits = inputs[kCandidatesClassificationInput].data<const float>();
    const float* box_regression = inputs[kCandidatesBoxInput].data<const float>();
    const float* direction_logits = inputs[kCandidatesDirectionInput].data<const float>();
    float* candidate_records = outputs[kOutputPort].data<float>();
    std::memset(candidate_records, 0, outputs[kOutputPort].get_byte_size());

    // Ascending anchor order here; the sort below is stable, so ties end up
    // ordered by anchor index.
    std::vector<int64_t> surviving_anchor_indices;
    for (int64_t word = 0; word < PP_SELECT_WORDS; ++word) {
        uint32_t bits = selection_words[word];
        while (bits) {
            const int bit = __builtin_ctz(bits);
            bits &= bits - 1;
            surviving_anchor_indices.push_back(word * PP_SELECT_BITS + bit);
        }
    }

    const size_t candidate_count =
        std::min<size_t>(surviving_anchor_indices.size(), static_cast<size_t>(PP_MAX_CANDIDATES));
    std::vector<float> decoded_records(candidate_count * PP_DET_CHANNELS);
    for (size_t i = 0; i < candidate_count; ++i) {
        int class_id = 0;
        const float score = anchor_score(classification_logits, surviving_anchor_indices[i], &class_id);
        decode_anchor(box_regression, direction_logits, surviving_anchor_indices[i], class_id, score,
                      decoded_records.data() + i * PP_DET_CHANNELS);
    }

    std::vector<size_t> order(candidate_count);
    for (size_t i = 0; i < candidate_count; ++i) order[i] = i;
    std::stable_sort(order.begin(), order.end(), [&](size_t a, size_t b) {
        return decoded_records[a * PP_DET_CHANNELS + PP_SCORE_INDEX] >
               decoded_records[b * PP_DET_CHANNELS + PP_SCORE_INDEX];
    });

    const int32_t candidate_count_i32 = static_cast<int32_t>(candidate_count);
    std::memcpy(candidate_records, &candidate_count_i32, sizeof(int32_t));
    for (size_t i = 0; i < candidate_count; ++i) {
        std::memcpy(candidate_records + (i + kCandidateHeaderRows) * PP_DET_CHANNELS,
                    decoded_records.data() + order[i] * PP_DET_CHANNELS,
                    PP_DET_CHANNELS * sizeof(float));
    }
    return true;
}

// ---------------------------------------------------------------------------
// Stage 3: rotated-BEV IoU bitmask.
// ---------------------------------------------------------------------------
PPPostprocNms::PPPostprocNms(const ov::Output<ov::Node>& candidate_records) : Op({candidate_records}) {
    constructor_validate_and_infer_types();
}

void PPPostprocNms::validate_and_infer_types() {
    set_output_type(kOutputPort, ov::element::i32, ov::PartialShape{PP_MAX_CANDIDATES, PP_NMS_COL_BLOCKS});
}

std::shared_ptr<ov::Node> PPPostprocNms::clone_with_new_inputs(const ov::OutputVector& new_args) const {
    OPENVINO_ASSERT(new_args.size() == 1, "PPPostprocNms expects 1 input");
    return std::make_shared<PPPostprocNms>(new_args[kNmsCandidatesInput]);
}

bool PPPostprocNms::visit_attributes(ov::AttributeVisitor&) { return true; }

bool PPPostprocNms::has_evaluate() const { return true; }

bool PPPostprocNms::evaluate(ov::TensorVector& outputs, const ov::TensorVector& inputs) const {
    OPENVINO_ASSERT(inputs.size() == kNmsInputCount && outputs.size() == kOutputCount,
                    "PPPostprocNms expects 1 input and 1 output");
    outputs[kOutputPort].set_shape({static_cast<size_t>(PP_MAX_CANDIDATES), static_cast<size_t>(PP_NMS_COL_BLOCKS)});

    const float* candidate_records =
        inputs[kNmsCandidatesInput].data<const float>() + kCandidateHeaderRows * PP_DET_CHANNELS;
    auto* nms_mask = reinterpret_cast<uint32_t*>(outputs[kOutputPort].data<int32_t>());
    std::memset(nms_mask, 0, outputs[kOutputPort].get_byte_size());

    const int32_t candidate_count = header_count(inputs[kNmsCandidatesInput]);
    for (int32_t row = 0; row < candidate_count; ++row) {
        const float* current_candidate = candidate_records + static_cast<int64_t>(row) * PP_DET_CHANNELS;
        for (int32_t col_block = row / PP_NMS_BLOCK;
             col_block < (candidate_count + PP_NMS_BLOCK - kSkipCurrentCandidate) / PP_NMS_BLOCK;
             ++col_block) {
            uint32_t bits = 0;
            const int32_t begin = col_block == row / PP_NMS_BLOCK
                                      ? row % PP_NMS_BLOCK + kSkipCurrentCandidate
                                      : 0;
            const int32_t block_size =
                std::min<int32_t>(PP_NMS_BLOCK, candidate_count - col_block * PP_NMS_BLOCK);
            for (int32_t i = begin; i < block_size; ++i) {
                const float* other_candidate =
                    candidate_records + static_cast<int64_t>(col_block * PP_NMS_BLOCK + i) * PP_DET_CHANNELS;
                if (rotated_iou(current_candidate, other_candidate) >= PP_NMS_THRESH) bits |= 1u << i;
            }
            nms_mask[static_cast<int64_t>(row) * PP_NMS_COL_BLOCKS + col_block] = bits;
        }
    }
    return true;
}

// ---------------------------------------------------------------------------
// Stage 4: greedy suppression.
// ---------------------------------------------------------------------------
PPPostprocGather::PPPostprocGather(const ov::Output<ov::Node>& candidate_records,
                                   const ov::Output<ov::Node>& nms_mask)
    : Op({candidate_records, nms_mask}) {
    constructor_validate_and_infer_types();
}

void PPPostprocGather::validate_and_infer_types() {
    set_output_type(kOutputPort, ov::element::f32, ov::PartialShape{PP_MAX_DETECTIONS, PP_DET_CHANNELS});
}

std::shared_ptr<ov::Node> PPPostprocGather::clone_with_new_inputs(const ov::OutputVector& new_args) const {
    OPENVINO_ASSERT(new_args.size() == 2, "PPPostprocGather expects 2 inputs");
    return std::make_shared<PPPostprocGather>(new_args[kGatherCandidatesInput], new_args[kGatherNmsMaskInput]);
}

bool PPPostprocGather::visit_attributes(ov::AttributeVisitor&) { return true; }

bool PPPostprocGather::has_evaluate() const { return true; }

bool PPPostprocGather::evaluate(ov::TensorVector& outputs, const ov::TensorVector& inputs) const {
    OPENVINO_ASSERT(inputs.size() == kGatherInputCount && outputs.size() == kOutputCount,
                    "PPPostprocGather expects 2 inputs and 1 output");
    outputs[kOutputPort].set_shape({static_cast<size_t>(PP_MAX_DETECTIONS), static_cast<size_t>(PP_DET_CHANNELS)});

    const float* candidate_records =
        inputs[kGatherCandidatesInput].data<const float>() + kCandidateHeaderRows * PP_DET_CHANNELS;
    const auto* nms_mask = reinterpret_cast<const uint32_t*>(inputs[kGatherNmsMaskInput].data<const int32_t>());
    float* detections = outputs[kOutputPort].data<float>();
    std::memset(detections, 0, outputs[kOutputPort].get_byte_size());

    const int32_t candidate_count = header_count(inputs[kGatherCandidatesInput]);
    const int32_t col_blocks =
        (candidate_count + PP_NMS_BLOCK - kSkipCurrentCandidate) / PP_NMS_BLOCK;
    std::vector<uint32_t> removed(static_cast<size_t>(col_blocks), 0);

    int32_t kept = 0;
    for (int32_t i = 0; i < candidate_count && kept < PP_MAX_DETECTIONS; ++i) {
        const int32_t block = i / PP_NMS_BLOCK;
        if (removed[static_cast<size_t>(block)] & (1u << (i % PP_NMS_BLOCK))) continue;
        std::memcpy(detections + static_cast<int64_t>(kept) * PP_DET_CHANNELS,
                candidate_records + static_cast<int64_t>(i) * PP_DET_CHANNELS,
                    PP_DET_CHANNELS * sizeof(float));
        kept++;
        const uint32_t* nms_row = nms_mask + static_cast<int64_t>(i) * PP_NMS_COL_BLOCKS;
        for (int32_t j = block; j < col_blocks; ++j) removed[static_cast<size_t>(j)] |= nms_row[j];
    }
    return true;
}

}  // namespace PpExtension
