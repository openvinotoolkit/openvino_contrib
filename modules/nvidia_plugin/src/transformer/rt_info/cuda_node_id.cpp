// Copyright (C) 2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "cuda_node_id.hpp"

void ov::nvidia_gpu::rt_info::set_node_id(const std::shared_ptr<Node>& node, uint64_t id) {
    auto& rt_info = node->get_rt_info();
    rt_info[CudaNodeId::get_type_info_static()] = id;
}

void ov::nvidia_gpu::rt_info::remove_node_id(const std::shared_ptr<Node>& node) {
    auto& rt_info = node->get_rt_info();
    rt_info.erase(CudaNodeId::get_type_info_static());
}

uint64_t ov::nvidia_gpu::rt_info::get_node_id(const std::shared_ptr<Node>& node) {
    const auto& rt_info = node->get_rt_info();
    return rt_info.at(CudaNodeId::get_type_info_static()).as<uint64_t>();
}
