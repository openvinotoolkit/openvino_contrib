// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma OPENCL FP_CONTRACT OFF

// Shared helpers for detection-head postprocessing.
//
// The pipeline contains score filtering, candidate decoding and sorting,
// rotated-BEV IoU masking, and greedy suppression.

#ifdef cl_khr_fp64
#pragma OPENCL EXTENSION cl_khr_fp64 : enable
#endif

// Decoder geometry, record layout, and mask sizes come from the generated
// decoder configuration concatenated before this file.

inline float pp_sigmoid(float x) { return 1.0f / (1.0f + exp(-x)); }

// Return the largest class sigmoid for one anchor. The lowest class index wins ties.
inline float pp_anchor_score(__global const float* cls, int anchor, int* class_id) {
    __global const float* scores = cls + (long)anchor * PP_NUM_CLASSES;
    float best = pp_sigmoid(scores[0]);
    int best_id = 0;
    for (int i = 1; i < PP_NUM_CLASSES; ++i) {
        const float value = pp_sigmoid(scores[i]);
        if (value > best) {
            best = value;
            best_id = i;
        }
    }
    if (class_id) *class_id = best_id;
    return best;
}

// Anchor deltas -> {x, y, z, w, l, h, yaw, class_id bits, score}.
inline void pp_decode(__global const float* box, __global const float* dir, int anchor, int class_id,
                      float score, float* out) {
    const int loc = anchor / PP_NUM_ANCHORS;
    const int ith = anchor % PP_NUM_ANCHORS;
    const int col = loc % PP_FEATURE_X;
    const int row = loc / PP_FEATURE_X;

    // Place each anchor at the center of its feature-map cell.
    const float x_offset = PP_MIN_X + (col + 0.5f) * (PP_MAX_X - PP_MIN_X) / PP_FEATURE_X;
    const float y_offset = PP_MIN_Y + (row + 0.5f) * (PP_MAX_Y - PP_MIN_Y) / PP_FEATURE_Y;

    const float dxa = PP_ANCHORS[ith * PP_ANCHOR_VALUES + 0];
    const float dya = PP_ANCHORS[ith * PP_ANCHOR_VALUES + 1];
    const float dza = PP_ANCHORS[ith * PP_ANCHOR_VALUES + 2];
    const float ra = PP_ANCHORS[ith * PP_ANCHOR_VALUES + 3];
    const float za = dza / 2 + PP_ANCHOR_BOTTOM_HEIGHTS[ith / PP_NUM_ROTATIONS];
    const float diagonal = sqrt(dxa * dxa + dya * dya);

    __global const float* e = box + (long)anchor * PP_NUM_BOX_VALUES;
    const float bx = e[0] * diagonal + x_offset;
    const float by = e[1] * diagonal + y_offset;
    const float bz = e[2] * dza + za;
    const float bw = exp(e[3]) * dxa;
    const float bl = exp(e[4]) * dya;
    const float bh = exp(e[5]) * dza;
    const float br = e[6] + ra;

    __global const float* d = dir + (long)anchor * PP_NUM_DIR_VALUES;
    const int dir_label = d[0] > d[1] ? 0 : 1;

    // Wrap the direction-adjusted yaw to one direction period.
    const float period = 2.0f * (float)M_PI / PP_NUM_DIR_VALUES;
    const float val = br - PP_DIR_OFFSET;
#ifdef cl_khr_fp64
    const float dir_rot = (float)((double)val - floor((double)val / ((double)period + 1e-8)) * (double)period);
#else
    const float dir_rot = val - floor(val / (period + 1e-8f)) * period;
#endif
    const float yaw = dir_rot + PP_DIR_OFFSET + period * dir_label;

    out[0] = bx;
    out[1] = by;
    out[2] = bz;
    out[3] = bw;
    out[4] = bl;
    out[5] = bh;
    out[6] = yaw;
    out[PP_CLASS_INDEX] = as_float(class_id);
    out[PP_SCORE_INDEX] = score;
}

// ---------------------------------------------------------------------------
// Rotated BEV IoU helpers.
// ---------------------------------------------------------------------------
inline float pp_cross(float2 p1, float2 p2, float2 p0) {
    return (p1.x - p0.x) * (p2.y - p0.y) - (p2.x - p0.x) * (p1.y - p0.y);
}

inline int pp_check_box2d(const float* box, float2 p) {
    const float MARGIN = 1e-2f;
    const float center_x = box[0];
    const float center_y = box[1];
    const float angle_cos = cos(-box[6]);
    const float angle_sin = sin(-box[6]);
    const float rot_x = (p.x - center_x) * angle_cos + (p.y - center_y) * (-angle_sin);
    const float rot_y = (p.x - center_x) * angle_sin + (p.y - center_y) * angle_cos;
    return fabs(rot_x) < box[3] / 2 + MARGIN && fabs(rot_y) < box[4] / 2 + MARGIN;
}

inline bool pp_intersection(float2 p1, float2 p0, float2 q1, float2 q0, float2* ans) {
    if ((fmin(p0.x, p1.x) <= fmax(q0.x, q1.x) && fmin(q0.x, q1.x) <= fmax(p0.x, p1.x) &&
         fmin(p0.y, p1.y) <= fmax(q0.y, q1.y) && fmin(q0.y, q1.y) <= fmax(p0.y, p1.y)) == 0)
        return false;

    const float s1 = pp_cross(q0, p1, p0);
    const float s2 = pp_cross(p1, q1, p0);
    const float s3 = pp_cross(p0, q1, q0);
    const float s4 = pp_cross(q1, p1, q0);
    if (!(s1 * s2 > 0 && s3 * s4 > 0)) return false;

    const float s5 = pp_cross(q1, p1, p0);
    if (fabs(s5 - s1) > 1e-8f) {
        (*ans).x = (s5 * q0.x - s1 * q1.x) / (s5 - s1);
        (*ans).y = (s5 * q0.y - s1 * q1.y) / (s5 - s1);
    } else {
        const float a0 = p0.y - p1.y, b0 = p1.x - p0.x, c0 = p0.x * p1.y - p1.x * p0.y;
        const float a1 = q0.y - q1.y, b1 = q1.x - q0.x, c1 = q0.x * q1.y - q1.x * q0.y;
        const float D = a0 * b1 - a1 * b0;
        (*ans).x = (b0 * c1 - b1 * c0) / D;
        (*ans).y = (a1 * c0 - a0 * c1) / D;
    }
    return true;
}

inline float2 pp_rotate_around_center(float2 center, float angle_cos, float angle_sin, float2 p) {
    const float new_x = (p.x - center.x) * angle_cos + (p.y - center.y) * (-angle_sin) + center.x;
    const float new_y = (p.x - center.x) * angle_sin + (p.y - center.y) * angle_cos + center.y;
    return (float2)(new_x, new_y);
}

// Return the circumradius of a box for the current length and width.
inline float pp_box_reach(float w, float l) { return 0.5f * sqrt(w * w + l * l); }

inline float pp_rotated_iou(const float* box_a, const float* box_b) {
    const float a_dx_half = box_a[3] / 2, b_dx_half = box_b[3] / 2;
    const float a_dy_half = box_a[4] / 2, b_dy_half = box_b[4] / 2;
    const float a_x1 = box_a[0] - a_dx_half, a_y1 = box_a[1] - a_dy_half;
    const float a_x2 = box_a[0] + a_dx_half, a_y2 = box_a[1] + a_dy_half;
    const float b_x1 = box_b[0] - b_dx_half, b_y1 = box_b[1] - b_dy_half;
    const float b_x2 = box_b[0] + b_dx_half, b_y2 = box_b[1] + b_dy_half;

    float2 box_a_corners[5];
    float2 box_b_corners[5];
    const float2 center_a = (float2)(box_a[0], box_a[1]);
    const float2 center_b = (float2)(box_b[0], box_b[1]);

    // Two convex quadrilaterals produce at most 16 intersection points.
    float2 cross_points[16];
    float2 poly_center = (float2)(0.0f, 0.0f);
    int cnt = 0;

    box_a_corners[0] = (float2)(a_x1, a_y1);
    box_a_corners[1] = (float2)(a_x2, a_y1);
    box_a_corners[2] = (float2)(a_x2, a_y2);
    box_a_corners[3] = (float2)(a_x1, a_y2);

    box_b_corners[0] = (float2)(b_x1, b_y1);
    box_b_corners[1] = (float2)(b_x2, b_y1);
    box_b_corners[2] = (float2)(b_x2, b_y2);
    box_b_corners[3] = (float2)(b_x1, b_y2);

    const float a_angle_cos = cos(box_a[6]), a_angle_sin = sin(box_a[6]);
    const float b_angle_cos = cos(box_b[6]), b_angle_sin = sin(box_b[6]);
    for (int k = 0; k < 4; k++) {
        box_a_corners[k] = pp_rotate_around_center(center_a, a_angle_cos, a_angle_sin, box_a_corners[k]);
        box_b_corners[k] = pp_rotate_around_center(center_b, b_angle_cos, b_angle_sin, box_b_corners[k]);
    }
    box_a_corners[4] = box_a_corners[0];
    box_b_corners[4] = box_b_corners[0];

    for (int i = 0; i < 4; i++) {
        for (int j = 0; j < 4; j++) {
            float2 point;
            if (pp_intersection(box_a_corners[i + 1], box_a_corners[i], box_b_corners[j + 1],
                                box_b_corners[j], &point)) {
                poly_center += point;
                cross_points[cnt] = point;
                cnt++;
            }
        }
    }

    for (int k = 0; k < 4; k++) {
        if (pp_check_box2d(box_a, box_b_corners[k])) {
            poly_center += box_b_corners[k];
            cross_points[cnt] = box_b_corners[k];
            cnt++;
        }
        if (pp_check_box2d(box_b, box_a_corners[k])) {
            poly_center += box_a_corners[k];
            cross_points[cnt] = box_a_corners[k];
            cnt++;
        }
    }

    poly_center.x /= cnt;
    poly_center.y /= cnt;

    // Compute each point angle once and reuse it during sorting.
    float angle[16];
    for (int k = 0; k < cnt; k++)
        angle[k] = atan2(cross_points[k].y - poly_center.y, cross_points[k].x - poly_center.x);

    for (int j = 0; j < cnt - 1; j++) {
        for (int i = 0; i < cnt - j - 1; i++) {
            if (angle[i] > angle[i + 1]) {
                const float2 temp = cross_points[i];
                cross_points[i] = cross_points[i + 1];
                cross_points[i + 1] = temp;
                const float temp_angle = angle[i];
                angle[i] = angle[i + 1];
                angle[i + 1] = temp_angle;
            }
        }
    }

    float area = 0.0f;
    for (int k = 0; k < cnt - 1; k++) {
        const float2 a = cross_points[k] - cross_points[0];
        const float2 b = cross_points[k + 1] - cross_points[0];
        area += (a.x * b.y - a.y * b.x);
    }

    const float s_overlap = fabs(area) / 2.0f;
    const float sa = box_a[3] * box_a[4];
    const float sb = box_b[3] * box_b[4];
    return s_overlap / fmax(sa + sb - s_overlap, 1e-8f);
}
