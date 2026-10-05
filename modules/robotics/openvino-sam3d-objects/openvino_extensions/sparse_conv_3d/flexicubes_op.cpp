/*
 * Copyright (C) 2018-2026 Intel Corporation
 * SPDX-License-Identifier: Apache-2.0
 */

// FlexiCubes mesh extraction
//
// Inputs:
//   0: features   [MAX_N, 101] FP32 — per-voxel mesh attrs (res 256)
//   1: coords     [MAX_N, 4]   I32  — [batch, z, y, x], res 256
//   2: num_voxels [1]          I32  — actual voxel count
// Outputs:
//   0: vertices [V, 3] FP32
//   1: faces    [F, 3] I32
//   2: colors   [V, 6] FP32
//   3: counts   [2]    I32  — [V, F]

#include <openvino/op/op.hpp>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <unordered_map>
#include <vector>
#include <algorithm>
#include <array>

#include "flexicubes_op.hpp"
#include "flexicubes_tables.hpp"

namespace SAM3DExtension {

// ── static geometry tables (match FlexiCubes / utils_cube exactly) ────────────
// utils_cube.cube_corners == FlexiCubes.cube_corners
static const int FC_CUBE_CORNERS[8][3] = {
    {0, 0, 0}, {1, 0, 0}, {0, 1, 0}, {1, 1, 0},
    {0, 0, 1}, {1, 0, 1}, {0, 1, 1}, {1, 1, 1}};
// FlexiCubes.cube_edges (24 entries = 12 edges × 2 corner indices)
static const int FC_CUBE_EDGES[24] = {
    0, 1, 1, 5, 4, 5, 0, 4, 2, 3, 3, 7, 6, 7, 2, 6, 2, 0, 3, 1, 7, 5, 6, 4};
// cube_corners_idx = 2^j
static const int FC_CORNER_IDX[8] = {1, 2, 4, 8, 16, 32, 64, 128};
static const int FC_QUAD_SPLIT_1[6] = {0, 1, 2, 0, 2, 3};
static const int FC_QUAD_SPLIT_2[6] = {0, 1, 3, 3, 1, 2};

static inline float fc_sigmoid(float x) { return 1.0f / (1.0f + std::exp(-x)); }

// Per-vertex sparse attribute (active-voxel corners only). Defaults: sdf=1,
// deform=0, color=0 (=> voxelgrid_color sigmoid(0)=0.5).
struct FCVAttr {
    float sdf;
    float deform[3];
    float color[6];
};

// ── the full extraction ───────────────────────────────────────────────────────
// coords_zyx: N rows of (z,y,x) int; feats: N×101 float. res=256.
void flexicubes_extract(const int32_t* coords_zyx, const float* feats, int N, int res,
                        std::vector<float>& out_vertices,
                        std::vector<int32_t>& out_faces,
                        std::vector<float>& out_colors) {
    const int C = 6;                       // color channels
    const int res_v = res + 1;             // vertex grid dimension
    const int64_t RV = (int64_t)res_v;
    const int64_t RC = (int64_t)res;
    const float sdf_bias = -1.0f / (float)res;
    const float defk = (1.0f - 1e-8f) / ((float)res * 2.0f);
    const float weight_scale = 0.99f;

    auto vid_of = [&](int64_t a, int64_t b, int64_t c) -> int64_t {
        return (a * RV + b) * RV + c;
    };
    auto cube_flat = [&](int64_t i, int64_t j, int64_t k) -> int64_t {
        return (i * RC + j) * RC + k;
    };

    // ── 1. sparse_cube2verts: scatter voxel-corner attrs to unique vertices, mean ──
    // Accumulate sum + count per dense vertex id.
    struct Acc { float sdf; float deform[3]; float color[6]; int cnt; };
    std::unordered_map<int64_t, Acc> vacc;
    vacc.reserve((size_t)N * 8 * 2);
    // weights per active voxel (for beta/alpha/gamma), keyed by cube flat index.
    std::unordered_map<int64_t, const float*> vox_weights;
    vox_weights.reserve((size_t)N * 2);

    for (int i = 0; i < N; ++i) {
        int64_t z = coords_zyx[(size_t)i * 3 + 0];
        int64_t y = coords_zyx[(size_t)i * 3 + 1];
        int64_t x = coords_zyx[(size_t)i * 3 + 2];
        const float* fr = feats + (size_t)i * 101;
        vox_weights[cube_flat(z, y, x)] = fr + 32;  // weights = feats[32:53]
        for (int j = 0; j < 8; ++j) {
            int64_t vz = z + FC_CUBE_CORNERS[j][0];
            int64_t vy = y + FC_CUBE_CORNERS[j][1];
            int64_t vx = x + FC_CUBE_CORNERS[j][2];
            int64_t vid = vid_of(vz, vy, vx);
            float sdf_j = fr[j] + sdf_bias;
            const float* def_j = fr + 8 + j * 3;
            const float* col_j = fr + 53 + j * 6;
            auto it = vacc.find(vid);
            if (it == vacc.end()) {
                Acc a;
                a.sdf = sdf_j;
                a.deform[0] = def_j[0]; a.deform[1] = def_j[1]; a.deform[2] = def_j[2];
                for (int c = 0; c < 6; ++c) a.color[c] = col_j[c];
                a.cnt = 1;
                vacc.emplace(vid, a);
            } else {
                Acc& a = it->second;
                a.sdf += sdf_j;
                a.deform[0] += def_j[0]; a.deform[1] += def_j[1]; a.deform[2] += def_j[2];
                for (int c = 0; c < 6; ++c) a.color[c] += col_j[c];
                a.cnt += 1;
            }
        }
    }

    // Finalise mean → sparse vertex attribute map.
    std::unordered_map<int64_t, FCVAttr> vattr;
    vattr.reserve(vacc.size() * 2);
    for (auto& kv : vacc) {
        const Acc& a = kv.second;
        float inv = 1.0f / (float)a.cnt;
        FCVAttr v;
        v.sdf = a.sdf * inv;
        v.deform[0] = a.deform[0] * inv;
        v.deform[1] = a.deform[1] * inv;
        v.deform[2] = a.deform[2] * inv;
        for (int c = 0; c < 6; ++c) v.color[c] = a.color[c] * inv;
        vattr.emplace(kv.first, v);
    }

    // Accessors with defaults for the dense grid.
    auto get_sdf = [&](int64_t vid) -> float {
        auto it = vattr.find(vid);
        return it == vattr.end() ? 1.0f : it->second.sdf;  // outside default
    };
    auto get_occ = [&](int64_t vid) -> bool { return get_sdf(vid) < 0.0f; };
    // deformed vertex position x_nx3[vid]
    auto get_xpos = [&](int64_t vid, float out[3]) {
        int64_t a = vid / (RV * RV);
        int64_t b = (vid / RV) % RV;
        int64_t c = vid % RV;
        float dz = 0, dy = 0, dx = 0;
        auto it = vattr.find(vid);
        if (it != vattr.end()) { dz = it->second.deform[0]; dy = it->second.deform[1]; dx = it->second.deform[2]; }
        out[0] = (float)a / (float)res - 0.5f + defk * std::tanh(dz);
        out[1] = (float)b / (float)res - 0.5f + defk * std::tanh(dy);
        out[2] = (float)c / (float)res - 0.5f + defk * std::tanh(dx);
    };
    auto get_color = [&](int64_t vid, float out[6]) {  // sigmoid(color_d)
        auto it = vattr.find(vid);
        if (it == vattr.end()) { for (int c = 0; c < 6; ++c) out[c] = 0.5f; return; }
        for (int c = 0; c < 6; ++c) out[c] = fc_sigmoid(it->second.color[c]);
    };

    // ── 2. identify surface cubes (in ascending flat cube-index order) ────────
    // Candidate cubes = all cubes touching any occupied vertex.
    std::vector<int64_t> cand;
    cand.reserve(vattr.size() * 8);
    for (auto& kv : vattr) {
        if (!(kv.second.sdf < 0.0f)) continue;  // occupied vertex
        int64_t vid = kv.first;
        int64_t a = vid / (RV * RV);
        int64_t b = (vid / RV) % RV;
        int64_t c = vid % RV;
        // cubes containing this vertex: cube index (i,j,k) with corner == vertex.
        for (int di = 0; di <= 1; ++di)
            for (int dj = 0; dj <= 1; ++dj)
                for (int dk = 0; dk <= 1; ++dk) {
                    int64_t ci = a - di, cj = b - dj, ck = c - dk;
                    if (ci < 0 || cj < 0 || ck < 0 || ci >= res || cj >= res || ck >= res) continue;
                    cand.push_back(cube_flat(ci, cj, ck));
                }
    }
    std::sort(cand.begin(), cand.end());
    cand.erase(std::unique(cand.begin(), cand.end()), cand.end());

    // Filter candidates to genuine surface cubes; record corner vids + occ.
    int NS = 0;
    std::vector<int64_t> sc_flat;          // surf cube flat index (sorted)
    std::vector<std::array<int64_t, 8>> sc_vids;  // 8 corner vertex ids
    std::vector<int> sc_caseid;
    sc_flat.reserve(cand.size());
    sc_vids.reserve(cand.size());
    for (int64_t p : cand) {
        int64_t i = p / (RC * RC);
        int64_t j = (p / RC) % RC;
        int64_t k = p % RC;
        std::array<int64_t, 8> vids;
        int occ_sum = 0, caseid = 0;
        for (int corner = 0; corner < 8; ++corner) {
            int64_t vz = i + FC_CUBE_CORNERS[corner][0];
            int64_t vy = j + FC_CUBE_CORNERS[corner][1];
            int64_t vx = k + FC_CUBE_CORNERS[corner][2];
            int64_t vid = vid_of(vz, vy, vx);
            vids[corner] = vid;
            bool occ = get_occ(vid);
            if (occ) { occ_sum += 1; caseid += FC_CORNER_IDX[corner]; }
        }
        if (occ_sum > 0 && occ_sum < 8) {
            sc_flat.push_back(p);
            sc_vids.push_back(vids);
            sc_caseid.push_back(caseid);
        }
    }
    NS = (int)sc_flat.size();
    if (NS == 0) { out_vertices.clear(); out_faces.clear(); out_colors.clear(); return; }

    // ── 3. per-surf-cube weights → beta(12), alpha(8), gamma(1) ───────────────
    std::vector<std::array<float, 12>> beta(NS);
    std::vector<std::array<float, 8>> alpha(NS);
    std::vector<float> gamma_f(NS);
    for (int s = 0; s < NS; ++s) {
        int64_t i = sc_flat[s] / (RC * RC);
        int64_t j = (sc_flat[s] / RC) % RC;
        int64_t k = sc_flat[s] % RC;
        auto it = vox_weights.find(cube_flat(i, j, k));
        const float* w = (it == vox_weights.end()) ? nullptr : it->second;  // 21 values or 0
        for (int t = 0; t < 12; ++t) {
            float wv = w ? w[t] : 0.0f;
            beta[s][t] = std::tanh(wv) * weight_scale + 1.0f;
        }
        for (int t = 0; t < 8; ++t) {
            float wv = w ? w[12 + t] : 0.0f;
            alpha[s][t] = std::tanh(wv) * weight_scale + 1.0f;
        }
        float gw = w ? w[20] : 0.0f;
        gamma_f[s] = fc_sigmoid(gw) * weight_scale + (1.0f - weight_scale) / 2.0f;
    }

    // ── 4. _get_case_id: resolve ambiguous DMC configs via adjacency ──────────
    // Build map of "to_check" surf cubes (check_table[caseid][0]==1): cube3d→(s, pc[5]).
    struct PC { int s; int pc[5]; };
    std::unordered_map<int64_t, PC> tocheck;
    tocheck.reserve(NS);
    std::vector<int> tocheck_s;
    for (int s = 0; s < NS; ++s) {
        const int* pc = CHECK_TABLE + (size_t)sc_caseid[s] * 5;
        if (pc[0] == 1) {
            PC e; e.s = s;
            for (int t = 0; t < 5; ++t) e.pc[t] = pc[t];
            tocheck[sc_flat[s]] = e;
            tocheck_s.push_back(s);
        }
    }
    // Apply inversions (two-pass: map already built from original case ids).
    for (int s : tocheck_s) {
        const int* pc = CHECK_TABLE + (size_t)sc_caseid[s] * 5;
        int64_t i = sc_flat[s] / (RC * RC);
        int64_t j = (sc_flat[s] / RC) % RC;
        int64_t k = sc_flat[s] % RC;
        int64_t ai = i + pc[1], aj = j + pc[2], ak = k + pc[3];
        if (ai < 0 || ai >= res || aj < 0 || aj >= res || ak < 0 || ak >= res) continue;
        auto it = tocheck.find(cube_flat(ai, aj, ak));
        if (it != tocheck.end() && it->second.pc[0] == 1) {
            sc_caseid[s] = pc[4];  // invert to problem_config[...,-1]
        }
    }

    // ── 5. _identify_surf_edges: unique edges (torch.unique dim=0 semantics) ──
    // all_edges rows: per surf cube, 12 edges (v_a, v_b) using cube_edges corner
    // index pairs. Key = a*(res_v^3) + b keeps lexicographic (a,b) order.
    const int64_t VBIG = RV * RV * RV;  // 257^3, bounds vertex ids
    int64_t nrows = (int64_t)NS * 12;
    std::vector<int64_t> row_a(nrows), row_b(nrows), row_key(nrows);
    for (int s = 0; s < NS; ++s) {
        for (int e = 0; e < 12; ++e) {
            int ca = FC_CUBE_EDGES[2 * e];
            int cb = FC_CUBE_EDGES[2 * e + 1];
            int64_t a = sc_vids[s][ca];
            int64_t b = sc_vids[s][cb];
            int64_t r = (int64_t)s * 12 + e;
            row_a[r] = a; row_b[r] = b;
            row_key[r] = a * VBIG + b;
        }
    }
    // sort rows by key → unique ids + counts + inverse map
    std::vector<int64_t> order(nrows);
    for (int64_t r = 0; r < nrows; ++r) order[r] = r;
    std::sort(order.begin(), order.end(),
              [&](int64_t p, int64_t q) { return row_key[p] < row_key[q]; });
    std::vector<int> idx_unique(nrows);       // row → unique id
    std::vector<int64_t> uniq_a, uniq_b;      // unique edges
    std::vector<int> uniq_count;
    {
        int uid = -1;
        int64_t prev_key = -1;
        for (int64_t t = 0; t < nrows; ++t) {
            int64_t r = order[t];
            if (t == 0 || row_key[r] != prev_key) {
                ++uid;
                uniq_a.push_back(row_a[r]);
                uniq_b.push_back(row_b[r]);
                uniq_count.push_back(0);
                prev_key = row_key[r];
            }
            idx_unique[r] = uid;
            uniq_count[uid] += 1;
        }
    }
    int NU = (int)uniq_a.size();
    // mask_edges[u] = (occ(a)+occ(b))==1 ; mapping u→surf-edge index (or -1)
    std::vector<int> mask_edges(NU), mapping(NU, -1);
    std::vector<int64_t> surf_edges_a, surf_edges_b;
    for (int u = 0; u < NU; ++u) {
        int sm = (get_occ(uniq_a[u]) ? 1 : 0) + (get_occ(uniq_b[u]) ? 1 : 0);
        mask_edges[u] = (sm == 1) ? 1 : 0;
        if (mask_edges[u]) {
            mapping[u] = (int)surf_edges_a.size();
            surf_edges_a.push_back(uniq_a[u]);
            surf_edges_b.push_back(uniq_b[u]);
        }
    }
    int NSE = (int)surf_edges_a.size();
    // per-row (n_surf*12): idx_map (surf-edge idx or -1), counts, surf_edges_mask
    std::vector<int> idx_map(nrows), row_count(nrows), row_semask(nrows);
    for (int64_t r = 0; r < nrows; ++r) {
        int u = idx_unique[r];
        idx_map[r] = mapping[u];
        row_count[r] = uniq_count[u];
        row_semask[r] = mask_edges[u];
    }

    // Precompute per-surf-edge geometry: x (2×3), s (2×1), color (2×C), zero_crossing (3)
    std::vector<std::array<float, 6>> se_x(NSE);     // [x0(3), x1(3)]
    std::vector<std::array<float, 2>> se_s(NSE);
    std::vector<std::array<float, 12>> se_c(NSE);    // [c0(6), c1(6)]
    for (int se = 0; se < NSE; ++se) {
        int64_t a = surf_edges_a[se], b = surf_edges_b[se];
        float xa[3], xb[3], ca[6], cb[6];
        get_xpos(a, xa); get_xpos(b, xb);
        get_color(a, ca); get_color(b, cb);
        for (int t = 0; t < 3; ++t) { se_x[se][t] = xa[t]; se_x[se][3 + t] = xb[t]; }
        se_s[se][0] = get_sdf(a); se_s[se][1] = get_sdf(b);
        for (int t = 0; t < 6; ++t) { se_c[se][t] = ca[t]; se_c[se][6 + t] = cb[t]; }
    }

    // ── 6. _compute_vd: dual vertices as beta-weighted edge crossings ─────────
    // Assign vd indices grouped by num_vd ascending, then surf-cube order.
    std::vector<int> num_vd(NS);
    for (int s = 0; s < NS; ++s) num_vd[s] = NUM_VD_TABLE[sc_caseid[s]];
    // unique num values present, ascending
    std::vector<int> nums;
    {
        std::vector<int> tmp(num_vd);
        std::sort(tmp.begin(), tmp.end());
        tmp.erase(std::unique(tmp.begin(), tmp.end()), tmp.end());
        nums = tmp;
    }
    // flat edge-group arrays
    std::vector<int> eg_edge, eg_vd, eg_cube;   // per emitted (dmc edge slot)
    eg_edge.reserve((size_t)NS * 7);
    eg_vd.reserve((size_t)NS * 7);
    eg_cube.reserve((size_t)NS * 7);
    std::vector<float> vd_gamma;                // per dual vertex
    // vd_idx_map[s*12 + e] = vd index (last write wins)
    std::vector<int> vd_idx_map((size_t)NS * 12, 0);
    int total_num_vd = 0;
    for (int num : nums) {
        for (int s = 0; s < NS; ++s) {
            if (num_vd[s] != num) continue;
            const int* dmc = DMC_TABLE + (size_t)sc_caseid[s] * 28;  // [4][7]
            for (int g = 0; g < num; ++g) {
                int vd_index = total_num_vd++;
                vd_gamma.push_back(gamma_f[s]);
                const int* grp = dmc + g * 7;
                for (int e = 0; e < 7; ++e) {
                    int eg = grp[e];
                    if (eg == -1) continue;
                    eg_edge.push_back(eg);
                    eg_vd.push_back(vd_index);
                    eg_cube.push_back(s);
                    vd_idx_map[(size_t)s * 12 + eg] = vd_index;
                }
            }
        }
    }
    int TV = total_num_vd;
    std::vector<float> vd((size_t)TV * 3, 0.0f);
    std::vector<float> vd_color((size_t)TV * C, 0.0f);
    std::vector<float> beta_sum((size_t)TV, 0.0f);

    size_t M = eg_edge.size();
    for (size_t m = 0; m < M; ++m) {
        int s = eg_cube[m];
        int e = eg_edge[m];
        int se = idx_map[(size_t)s * 12 + e];   // surf-edge index (assumed >=0)
        // alpha at edge endpoints
        float a0 = alpha[s][FC_CUBE_EDGES[2 * e]];
        float a1 = alpha[s][FC_CUBE_EDGES[2 * e + 1]];
        float s0 = se_s[se][0], s1 = se_s[se][1];
        float w0 = s0 * a0, w1 = s1 * a1;
        // _linear_interp: (x0*w1' - x1*w0') / (w1'-w0') with w' = [w1, -w0]
        // => new_w0 = w1, new_w1 = -w0; denom = w1 - w0; ue = (x0*w1 + x1*(-w0))/denom
        float denom = w1 - w0;
        float beta_e = beta[s][e];
        int vidx = eg_vd[m];
        for (int t = 0; t < 3; ++t) {
            float x0 = se_x[se][t], x1 = se_x[se][3 + t];
            float ue = (x0 * w1 - x1 * w0) / denom;
            vd[(size_t)vidx * 3 + t] += ue * beta_e;
        }
        for (int c = 0; c < C; ++c) {
            float c0 = se_c[se][c], c1 = se_c[se][6 + c];
            float uc = (c0 * w1 - c1 * w0) / denom;
            vd_color[(size_t)vidx * C + c] += uc * beta_e;
        }
        beta_sum[vidx] += beta_e;
    }
    for (int v = 0; v < TV; ++v) {
        float inv = 1.0f / beta_sum[v];
        for (int t = 0; t < 3; ++t) vd[(size_t)v * 3 + t] *= inv;
        for (int c = 0; c < C; ++c) vd_color[(size_t)v * C + c] *= inv;
    }

    // ── 7. _triangulate: connect 4 dual verts around each interior surf edge ──
    // group_mask = (edge_counts==4) & surf_edges_mask, over the n_surf*12 rows.
    // group = idx_map[group_mask] (surf-edge idx); sort stable → quads of 4.
    std::vector<int64_t> gm_rows;  // rows passing mask
    gm_rows.reserve(nrows);
    for (int64_t r = 0; r < nrows; ++r) {
        if (row_count[r] == 4 && row_semask[r]) gm_rows.push_back(r);
    }
    // stable sort rows by group value (idx_map[r]); ties keep original row order.
    std::stable_sort(gm_rows.begin(), gm_rows.end(),
                     [&](int64_t p, int64_t q) { return idx_map[p] < idx_map[q]; });
    int64_t NQ4 = (int64_t)gm_rows.size();
    // group into quads of 4 (each surf edge shared by exactly 4 cubes)
    int64_t nquads = NQ4 / 4;
    // For each quad: 4 vd indices (vd_idx_map at masked rows), plus edge's sdf.
    std::vector<std::array<int, 4>> quads(nquads);
    std::vector<int> quad_flip(nquads);
    for (int64_t qi = 0; qi < nquads; ++qi) {
        for (int t = 0; t < 4; ++t) {
            int64_t r = gm_rows[qi * 4 + t];
            quads[qi][t] = vd_idx_map[r];  // vd_idx_map indexed by row (s*12+e)
        }
        // s_edges: scalar_field at surf_edges[edge_indices.reshape(-1,4)[:,0]] endpoints
        int se = idx_map[gm_rows[qi * 4 + 0]];
        float s0 = se_s[se][0];  // endpoint 0 sdf
        quad_flip[qi] = (s0 > 0.0f) ? 1 : 0;
    }
    // reorder: flipped quads first (with [0,1,3,2]) then non-flipped ([2,3,1,0])
    std::vector<std::array<int, 4>> quad_ordered;
    quad_ordered.reserve(nquads);
    for (int64_t qi = 0; qi < nquads; ++qi) {
        if (quad_flip[qi]) {
            auto& q = quads[qi];
            quad_ordered.push_back({q[0], q[1], q[3], q[2]});
        }
    }
    for (int64_t qi = 0; qi < nquads; ++qi) {
        if (!quad_flip[qi]) {
            auto& q = quads[qi];
            quad_ordered.push_back({q[2], q[3], q[1], q[0]});
        }
    }
    // split each quad into 2 triangles by gamma
    out_faces.clear();
    out_faces.reserve((size_t)quad_ordered.size() * 6);
    for (auto& q : quad_ordered) {
        float g02 = vd_gamma[q[0]] * vd_gamma[q[2]];
        float g13 = vd_gamma[q[1]] * vd_gamma[q[3]];
        const int* split = (g02 > g13) ? FC_QUAD_SPLIT_1 : FC_QUAD_SPLIT_2;
        for (int t = 0; t < 6; ++t) out_faces.push_back(q[split[t]]);
    }

    // ── outputs ───────────────────────────────────────────────────────────────
    out_vertices = std::move(vd);
    out_colors = std::move(vd_color);
}

// ── OV op boilerplate ─────────────────────────────────────────────────────────
FlexiCubesOp::FlexiCubesOp(const ov::Output<ov::Node>& features,
                           const ov::Output<ov::Node>& coords,
                           const ov::Output<ov::Node>& num_voxels)
    : Op({features, coords, num_voxels}) {
    constructor_validate_and_infer_types();
}

FlexiCubesOp::FlexiCubesOp(const ov::OutputVector& args) : Op(args) {
    constructor_validate_and_infer_types();
}

void FlexiCubesOp::validate_and_infer_types() {
    set_output_type(0, ov::element::f32, ov::PartialShape{ov::Dimension::dynamic(), 3});
    set_output_type(1, ov::element::i32, ov::PartialShape{ov::Dimension::dynamic(), 3});
    set_output_type(2, ov::element::f32, ov::PartialShape{ov::Dimension::dynamic(), 6});
    set_output_type(3, ov::element::i32, ov::PartialShape{2});
}

std::shared_ptr<ov::Node> FlexiCubesOp::clone_with_new_inputs(
    const ov::OutputVector& new_args) const {
    auto op = std::make_shared<FlexiCubesOp>(new_args);
    op->m_res = m_res;
    return op;
}

bool FlexiCubesOp::visit_attributes(ov::AttributeVisitor& visitor) {
    visitor.on_attribute("res", m_res);
    return true;
}

bool FlexiCubesOp::has_evaluate() const { return true; }

bool FlexiCubesOp::evaluate(ov::TensorVector& outputs,
                            const ov::TensorVector& inputs) const {
    const float* feats = inputs[0].data<float>();
    const int32_t* coords4 = inputs[1].data<int32_t>();
    const int32_t* num_ptr = inputs[2].data<int32_t>();
    int N = num_ptr[0];

    // Extract (z,y,x) from [b,z,y,x].
    std::vector<int32_t> coords_zyx((size_t)N * 3);
    for (int i = 0; i < N; ++i) {
        coords_zyx[(size_t)i * 3 + 0] = coords4[(size_t)i * 4 + 1];
        coords_zyx[(size_t)i * 3 + 1] = coords4[(size_t)i * 4 + 2];
        coords_zyx[(size_t)i * 3 + 2] = coords4[(size_t)i * 4 + 3];
    }

    std::vector<float> verts, colors;
    std::vector<int32_t> faces;
    flexicubes_extract(coords_zyx.data(), feats, N, (int)m_res, verts, faces, colors);

    int64_t V = (int64_t)verts.size() / 3;
    int64_t F = (int64_t)faces.size() / 3;
    outputs[0].set_shape(ov::Shape{(size_t)V, 3});
    outputs[1].set_shape(ov::Shape{(size_t)F, 3});
    outputs[2].set_shape(ov::Shape{(size_t)V, 6});
    outputs[3].set_shape(ov::Shape{2});
    if (V > 0) std::memcpy(outputs[0].data<float>(), verts.data(), verts.size() * sizeof(float));
    if (F > 0) std::memcpy(outputs[1].data<int32_t>(), faces.data(), faces.size() * sizeof(int32_t));
    if (V > 0) std::memcpy(outputs[2].data<float>(), colors.data(), colors.size() * sizeof(float));
    outputs[3].data<int32_t>()[0] = (int32_t)V;
    outputs[3].data<int32_t>()[1] = (int32_t)F;
    return true;
}

}  // namespace SAM3DExtension
