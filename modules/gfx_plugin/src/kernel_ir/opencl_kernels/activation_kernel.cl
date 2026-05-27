// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

static inline float gfx_gelu_tanh_f32(float x) {
    return 0.5f * x * (1.0f + tanh(0.7978845608f * (x + 0.044715f * x * x * x)));
}

static inline float gfx_softplus_f32(float x) {
    return log(1.0f + exp(x));
}

static inline float gfx_activation_f32(float x, uint op, float alpha, float beta) {
    switch (op) {
    case 16u: return fmax(x, 0.0f);
    case 17u: return 1.0f / (1.0f + exp(-x));
    case 18u: return tanh(x);
    case 19u: return fabs(x);
    case 20u: return -x;
    case 21u: return exp(x);
    case 22u: return log(x);
    case 23u: return sqrt(x);
    case 24u: return floor(x);
    case 25u: return ceil(x);
    case 26u: return x > 0.0f ? x : (exp(x) - 1.0f) * alpha;
    case 27u: return 0.5f * x * (1.0f + erf(x * 0.70710678118f));
    case 28u: return gfx_gelu_tanh_f32(x);
    case 29u: return x * fmin(fmax(x + 3.0f, 0.0f), 6.0f) / 6.0f;
    case 30u: return fmin(fmax(x + 3.0f, 0.0f), 6.0f) / 6.0f;
    case 31u: return gfx_softplus_f32(x);
    case 80u: return x * tanh(gfx_softplus_f32(x));
    case 81u: return x / (1.0f + fabs(x));
    case 82u: return x > 0.0f ? 1.0f : (x < 0.0f ? -1.0f : 0.0f);
    case 83u: return fmin(fmax(x, alpha), beta);
    case 84u: return sin(x);
    case 85u: return cos(x);
    case 86u: return tan(x);
    case 87u: return erf(x);
    case 88u: return asin(x);
    case 89u: return acos(x);
    case 90u: return atan(x);
    case 91u: return asinh(x);
    case 92u: return acosh(x);
    case 93u: return atanh(x);
    case 94u: return sinh(x);
    case 95u: return cosh(x);
    case 96u: return rint(x);
    case 97u: return round(x);
    default: return x;
    }
}

__kernel void gfx_opencl_generated_activation_f32(__global const float* src,
                                                  __global float* dst,
                                                  uint count,
                                                  uint op,
                                                  float alpha,
                                                  float beta) {
    const uint gid = get_global_id(0);
    if (gid >= count) {
        return;
    }
    dst[gid] = gfx_activation_f32(src[gid], op, alpha, beta);
}

static inline float gfx_f16_bits_to_f32(uint bits) {
    const uint sign = (bits >> 15) & 1u;
    const uint exponent = (bits >> 10) & 31u;
    const uint fraction = bits & 1023u;
    float value = 0.0f;
    if (exponent == 0u) {
        value = ldexp((float)fraction, -24);
    } else if (exponent == 31u) {
        value = fraction == 0u ? INFINITY : NAN;
    } else {
        value = ldexp(1.0f + (float)fraction / 1024.0f,
                      (int)exponent - 15);
    }
    return sign ? -value : value;
}

static inline uint gfx_f32_to_f16_bits(float value) {
    if (isnan(value)) {
        return 0x7e00u;
    }
    const uint sign = signbit(value) ? 0x8000u : 0u;
    float abs_value = fabs(value);
    if (isinf(abs_value)) {
        return sign | 0x7c00u;
    }
    if (abs_value == 0.0f) {
        return sign;
    }
    int exponent = 0;
    float normalized = frexp(abs_value, &exponent);
    exponent -= 1;
    const int half_exponent = exponent + 15;
    if (half_exponent >= 31) {
        return sign | 0x7c00u;
    }
    if (half_exponent <= 0) {
        const float scaled = ldexp(abs_value, 24);
        uint mantissa = (uint)floor(scaled + 0.5f);
        return sign | min(mantissa, 1023u);
    }
    normalized = normalized * 2.0f - 1.0f;
    uint mantissa = (uint)floor(normalized * 1024.0f + 0.5f);
    uint exponent_bits = (uint)half_exponent;
    if (mantissa == 1024u) {
        mantissa = 0u;
        ++exponent_bits;
        if (exponent_bits >= 31u) {
            return sign | 0x7c00u;
        }
    }
    return sign | (exponent_bits << 10) | (mantissa & 1023u);
}

__kernel void gfx_opencl_generated_activation_f16(__global const uint* src,
                                                  __global uint* dst,
                                                  uint count,
                                                  uint op,
                                                  float alpha,
                                                  float beta) {
    const uint word_idx = get_global_id(0);
    const uint elem0 = word_idx * 2u;
    if (elem0 >= count) {
        return;
    }
    const uint word = src[word_idx];
    const uint lo =
        gfx_f32_to_f16_bits(gfx_activation_f32(gfx_f16_bits_to_f32(word & 0xffffu), op, alpha, beta));
    uint hi = 0u;
    if (elem0 + 1u < count) {
        hi = gfx_f32_to_f16_bits(
            gfx_activation_f32(gfx_f16_bits_to_f32((word >> 16) & 0xffffu), op, alpha, beta));
    }
    dst[word_idx] = (hi << 16) | (lo & 0xffffu);
}
