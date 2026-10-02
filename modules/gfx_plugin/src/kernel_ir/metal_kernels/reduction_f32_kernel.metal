#include <metal_stdlib>
using namespace metal;

constant uint kReduceSum = 0u;
constant uint kReduceMean = 1u;
constant uint kReduceMax = 2u;
constant uint kReduceMin = 3u;
constant uint kReduceProd = 4u;
constant uint kReduceL1 = 5u;
constant uint kReduceL2 = 6u;

kernel void gfx_metal_generated_reduction_f32(
    device const float* input [[buffer(0)]],
    device float* output [[buffer(1)]],
    constant uint& num_elements [[buffer(2)]],
    constant uint& rank [[buffer(3)]],
    constant uint& op_code [[buffer(4)]],
    constant int* out_dims [[buffer(5)]],
    constant int* in_dims [[buffer(6)]],
    constant int* in_strides [[buffer(7)]],
    constant int* axis_mask [[buffer(8)]],
    constant int* reduce_dims [[buffer(9)]],
    uint gid [[thread_position_in_grid]]) {
    if (gid >= num_elements) {
        return;
    }

    uint idx = gid;
    int coords[8];
    for (uint d = rank; d-- > 0;) {
        coords[d] = idx % static_cast<uint>(out_dims[d]);
        idx /= static_cast<uint>(out_dims[d]);
    }

    uint reduce_size = 1u;
    for (uint d = 0; d < rank; ++d) {
        if (axis_mask[d]) {
            reduce_size *= static_cast<uint>(reduce_dims[d]);
        }
    }

    float acc = 0.0f;
    bool first = true;
    if (op_code == kReduceProd) {
        acc = 1.0f;
    }

    for (uint r = 0; r < reduce_size; ++r) {
        uint tmp = r;
        int in_idx = 0;
        for (uint d = rank; d-- > 0;) {
            int coord = coords[d];
            if (axis_mask[d]) {
                coord = static_cast<int>(tmp % static_cast<uint>(reduce_dims[d]));
                tmp /= static_cast<uint>(reduce_dims[d]);
            }
            in_idx += coord * in_strides[d];
        }

        const float value = input[in_idx];
        if (op_code == kReduceMax) {
            if (first || value > acc) {
                acc = value;
                first = false;
            }
        } else if (op_code == kReduceMin) {
            if (first || value < acc) {
                acc = value;
                first = false;
            }
        } else if (op_code == kReduceProd) {
            acc *= value;
        } else if (op_code == kReduceL1) {
            acc += fabs(value);
        } else if (op_code == kReduceL2) {
            acc += value * value;
        } else {
            acc += value;
        }
    }

    if (op_code == kReduceMean) {
        acc /= static_cast<float>(reduce_size);
    } else if (op_code == kReduceL2) {
        acc = sqrt(acc);
    }
    output[gid] = acc;
}
