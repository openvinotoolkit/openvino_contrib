// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <openvino/core/extension.hpp>
#include <openvino/core/op_extension.hpp>

#include "transformer/nodes/concat_optimized.hpp"
#include "transformer/nodes/fully_connected.hpp"
#include "transformer/nodes/fused_convolution.hpp"
#include "transformer/nodes/fused_convolution_backprop_data.hpp"
#include "transformer/nodes/lstm_sequence_optimized.hpp"

OPENVINO_CREATE_EXTENSIONS(
    std::vector<ov::Extension::Ptr>({
        std::make_shared<ov::OpExtension<ov::nvidia_gpu::nodes::ConcatOptimized>>(),
        std::make_shared<ov::OpExtension<ov::nvidia_gpu::nodes::FullyConnected>>(),
        std::make_shared<ov::OpExtension<ov::nvidia_gpu::nodes::FusedConvBackpropData>>(),
        std::make_shared<ov::OpExtension<ov::nvidia_gpu::nodes::FusedConvolution>>(),
        std::make_shared<ov::OpExtension<ov::nvidia_gpu::nodes::FusedGroupConvolution>>(),
        std::make_shared<ov::OpExtension<ov::nvidia_gpu::nodes::LSTMSequenceOptimized>>()
}));
