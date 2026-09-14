#include <metal_stdlib>
using namespace metal;

constant uint kReduceLogicalAnd = 7u;
constant uint kReduceLogicalOr = 8u;

kernel void gfx_metal_generated_reduction_logical_bool(
    device const uchar* input [[buffer(0)]],
    device uchar* output [[buffer(1)]],
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

    uchar acc = op_code == kReduceLogicalAnd ? static_cast<uchar>(1)
                                             : static_cast<uchar>(0);
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

        const bool value = input[in_idx] != 0;
        if (op_code == kReduceLogicalAnd) {
            acc = static_cast<uchar>((acc != 0 && value) ? 1 : 0);
        } else {
            acc = static_cast<uchar>((acc != 0 || value) ? 1 : 0);
        }
    }
    output[gid] = acc;
}
