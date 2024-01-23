// Copyright (C) 2022-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <cuda_operation_base.hpp>
#include <kernels/convert_color_i420.hpp>
#include <kernels/details/cuda_type_traits.hpp>

#include "converters.hpp"
#include "openvino/op/i420_to_bgr.hpp"
#include "openvino/op/i420_to_rgb.hpp"

namespace ov {
namespace nvidia_gpu {

template <typename TNGraphNode, typename TKernel>
class I420ConvertColorBase : public OperationBase {
public:
    static_assert(std::is_same<TNGraphNode, ov::op::v8::I420toRGB>::value ||
                      std::is_same<TNGraphNode, ov::op::v8::I420toBGR>::value,
                  "TNGraphNode should be either NV12toRGB or NV12toBGR");

    using NodeOp = TNGraphNode;
    I420ConvertColorBase(const CreationContext& context,
                         const NodeOp& node,
                         IndexCollection&& inputIds,
                         IndexCollection&& outputIds)
        : OperationBase{context, node, move(inputIds), move(outputIds)} {
        constexpr const size_t N_DIM = 0;
        constexpr const size_t H_DIM = 1;
        constexpr const size_t W_DIM = 2;
        OPENVINO_ASSERT(node.get_input_size() == 1 || node.get_input_size() == 3,
                        "I420 conversion shall have one or 3 inputs, but it is ",
                        node.get_input_size());
        auto single_plane = node.get_input_size() == 1;

        const auto& in_tensor_shape = node.get_input_shape(0);
        auto batch_size = in_tensor_shape[N_DIM];
        auto image_w = in_tensor_shape[W_DIM];
        auto image_h = in_tensor_shape[H_DIM];
        if (single_plane) {
            image_h = image_h * 2 / 3;
        }

        OPENVINO_ASSERT(node.get_input_size() == 1 || node.get_input_size() == 3, "Node name: ", GetName());
        OPENVINO_ASSERT(node.get_output_size() == 1, "Node name: ", GetName());

        const auto element_type = node.get_output_element_type(0);

        const size_t max_threads_per_block = context.device().props().maxThreadsPerBlock;
        if (single_plane) {
            kernel_ = TKernel{convertDataType<ov::nvidia_gpu::kernel::Type_t>(element_type),
                              max_threads_per_block,
                              batch_size,
                              image_h,
                              image_w,
                              image_w * image_h * 3 / 2,
                              image_w * image_h * 3 / 2};
        } else {
            kernel_ = TKernel{convertDataType<ov::nvidia_gpu::kernel::Type_t>(element_type),
                              max_threads_per_block,
                              batch_size,
                              image_h,
                              image_w,
                              image_w * image_h,
                              image_w * image_h / 4};
        }
    }

    void Execute(const InferenceRequestContext& context,
                 Inputs inputTensors,
                 Outputs outputTensors,
                 const Workbuffers& workbuffers) const override {
        OPENVINO_ASSERT(kernel_, "Node name: ", GetName());
        OPENVINO_ASSERT(inputTensors.size() == 1 || inputTensors.size() == 3, "Node name: ", GetName());
        OPENVINO_ASSERT(outputTensors.size() == 1, "Node name: ", GetName());
        auto& stream = context.getThreadContext().stream();

        if (inputTensors.size() == 1) {
            (*kernel_)(stream.get(),
                       static_cast<const void*>(inputTensors[0].get()),
                       static_cast<void*>(outputTensors[0].get()));
        } else {
            (*kernel_)(stream.get(),
                       static_cast<const void*>(inputTensors[0].get()),
                       static_cast<const void*>(inputTensors[1].get()),
                       static_cast<const void*>(inputTensors[2].get()),
                       static_cast<void*>(outputTensors[0].get()));
        }
    }

    CudaGraphCompatibility GetCudaGraphCompatibility() const override { return CudaGraphCompatibility::FULL; }

private:
    std::optional<TKernel> kernel_;
};

class I420toRGBOp
    : public I420ConvertColorBase<ov::op::v8::I420toRGB, kernel::I420ColorConvert<kernel::ColorConversion::RGB>> {
public:
    using I420ConvertColorBase::I420ConvertColorBase;
};

class I420toBGROp
    : public I420ConvertColorBase<ov::op::v8::I420toBGR, kernel::I420ColorConvert<kernel::ColorConversion::BGR>> {
public:
    using I420ConvertColorBase::I420ConvertColorBase;
};

}  // namespace nvidia_gpu
}  // namespace ov
