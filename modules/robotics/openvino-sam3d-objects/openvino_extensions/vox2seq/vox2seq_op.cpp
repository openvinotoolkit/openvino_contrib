/*
 * Copyright (C) 2018-2026 Intel Corporation
 * SPDX-License-Identifier: Apache-2.0
 */


// Vox2Seq OpenVINO Extension — Full Implementation
//
// Z-order (Morton) encoding: bit-interleave x, y, z coordinates.
// Hilbert curve encoding: LUT-based state machine for 3D Hilbert curve.
// Both are trivially parallel — one code per voxel.

#include "vox2seq_op.hpp"
#include <openvino/core/type.hpp>
#include <openvino/core/shape.hpp>

#include <cstring>
#include <cstdint>
#include <algorithm>
#include <vector>
#include <iostream>
#ifdef _OPENMP
#include <omp.h>
#endif

namespace SAM3DExtension {

// ============================================================
// Z-order (Morton) encoding
// ============================================================
static inline uint64_t spread_bits_3(uint32_t v) {
    // Spread bits of v: insert two zero bits between each original bit
    // For 21-bit input → 63-bit output (fits in int64)
    uint64_t x = v & 0x1FFFFF;  // 21 bits max
    x = (x | (x << 32)) & 0x1F00000000FFFF;
    x = (x | (x << 16)) & 0x1F0000FF0000FF;
    x = (x | (x << 8))  & 0x100F00F00F00F00F;
    x = (x | (x << 4))  & 0x10C30C30C30C30C3;
    x = (x | (x << 2))  & 0x1249249249249249;
    return x;
}

static inline int64_t morton_encode_3d(int32_t x, int32_t y, int32_t z) {
    return (int64_t)(spread_bits_3((uint32_t)x) |
                     (spread_bits_3((uint32_t)y) << 1) |
                     (spread_bits_3((uint32_t)z) << 2));
}

// ============================================================
// Hilbert curve encoding (3D)
// ============================================================
// LUT-based 3D Hilbert encoding using the state-machine approach.
// Each octant maps to a Hilbert index + new state.

// Hilbert state transition table for 3D
// State × Octant → (hilbert_index, next_state)
// 12 states, 8 octants each
static const int HILBERT_TABLE[12][8][2] = {
    // State 0
    {{0,1},{1,2},{3,2},{2,5},{7,4},{6,5},{4,4},{5,1}},
    // State 1
    {{0,0},{7,6},{1,6},{6,11},{3,8},{4,11},{2,8},{5,0}},
    // State 2
    {{0,0},{3,7},{7,7},{4,9},{1,10},{2,9},{6,10},{5,0}},
    // State 3
    {{2,1},{3,2},{1,2},{0,5},{5,4},{4,5},{6,4},{7,1}},
    // State 4
    {{6,0},{1,6},{7,6},{0,11},{5,8},{2,11},{4,8},{3,0}},
    // State 5
    {{2,0},{5,7},{3,7},{4,9},{1,10},{6,9},{0,10},{7,0}},
    // State 6
    {{6,3},{7,2},{5,2},{4,5},{1,4},{0,5},{2,4},{3,3}},
    // State 7
    {{0,3},{1,7},{3,7},{2,9},{7,10},{6,9},{4,10},{5,3}},
    // State 8
    {{4,3},{7,7},{3,7},{0,9},{5,10},{6,9},{2,10},{1,3}},
    // State 9
    {{6,3},{1,2},{7,2},{0,5},{5,4},{2,5},{4,4},{3,3}},
    // State 10
    {{4,0},{3,6},{7,6},{0,11},{5,8},{4,11},{2,8},{1,0}},
    // State 11
    {{4,0},{5,7},{1,7},{0,9},{7,10},{6,9},{2,10},{3,0}},
};

static int64_t hilbert_encode_3d(int32_t x, int32_t y, int32_t z, int bits = 10) {
    int64_t code = 0;
    int state = 0;

    for (int i = bits - 1; i >= 0; i--) {
        int octant = ((x >> i) & 1) |
                     (((y >> i) & 1) << 1) |
                     (((z >> i) & 1) << 2);
        int hilbert_idx = HILBERT_TABLE[state][octant][0];
        int next_state  = HILBERT_TABLE[state][octant][1];
        code = (code << 3) | hilbert_idx;
        state = next_state;
    }
    return code;
}

// ============================================================
// OV Operation Interface
// ============================================================

Vox2SeqOp::Vox2SeqOp(
    const ov::Output<ov::Node>& coords,
    const ov::Output<ov::Node>& num_pts)
    : Op({coords, num_pts})
{
    constructor_validate_and_infer_types();
}

void Vox2SeqOp::validate_and_infer_types() {
    set_output_type(0, ov::element::i64,
        ov::Shape{(size_t)VOX2SEQ_MAX_N});
}

std::shared_ptr<ov::Node> Vox2SeqOp::clone_with_new_inputs(
    const ov::OutputVector& new_args) const
{
    auto op = std::make_shared<Vox2SeqOp>(new_args[0], new_args[1]);
    op->m_mode = m_mode;
    op->m_permute = m_permute;
    return op;
}

bool Vox2SeqOp::visit_attributes(ov::AttributeVisitor& visitor) {
    visitor.on_attribute("mode", m_mode);
    visitor.on_attribute("permute", m_permute);
    return true;
}

bool Vox2SeqOp::has_evaluate() const { return true; }

bool Vox2SeqOp::evaluate(
    ov::TensorVector& outputs,
    const ov::TensorVector& inputs) const
{
    const int32_t* coords_ptr   = inputs[0].data<int32_t>();
    const int32_t* num_pts_ptr  = inputs[1].data<int32_t>();
    int N = num_pts_ptr[0];
    if (N <= 0) N = 0;
    if (N > VOX2SEQ_MAX_N) N = VOX2SEQ_MAX_N;

    outputs[0].set_shape(ov::Shape{(size_t)VOX2SEQ_MAX_N});
    int64_t* codes = outputs[0].data<int64_t>();
    std::memset(codes, 0, VOX2SEQ_MAX_N * sizeof(int64_t));

    int px = (m_permute.size() > 0) ? (int)m_permute[0] : 0;
    int py = (m_permute.size() > 1) ? (int)m_permute[1] : 1;
    int pz = (m_permute.size() > 2) ? (int)m_permute[2] : 2;

    if (m_mode == "z_order" || m_mode == "morton") {
        #ifdef _OPENMP
        #pragma omp parallel for
        #endif
        for (int i = 0; i < N; i++) {
            int32_t x = coords_ptr[i * 3 + px];
            int32_t y = coords_ptr[i * 3 + py];
            int32_t z = coords_ptr[i * 3 + pz];
            codes[i] = morton_encode_3d(x, y, z);
        }
    } else if (m_mode == "hilbert") {
        // Determine number of bits from max coordinate
        int max_coord = 0;
        for (int i = 0; i < N; i++) {
            max_coord = std::max(max_coord, (int)coords_ptr[i * 3 + 0]);
            max_coord = std::max(max_coord, (int)coords_ptr[i * 3 + 1]);
            max_coord = std::max(max_coord, (int)coords_ptr[i * 3 + 2]);
        }
        int bits = 1;
        while ((1 << bits) <= max_coord) bits++;

        #ifdef _OPENMP
        #pragma omp parallel for
        #endif
        for (int i = 0; i < N; i++) {
            int32_t x = coords_ptr[i * 3 + px];
            int32_t y = coords_ptr[i * 3 + py];
            int32_t z = coords_ptr[i * 3 + pz];
            codes[i] = hilbert_encode_3d(x, y, z, bits);
        }
    }

    return true;
}

}  // namespace SAM3DExtension
