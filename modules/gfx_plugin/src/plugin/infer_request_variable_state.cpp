// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "plugin/infer_request_variable_state.hpp"

#include <cstring>
#include <memory>
#include <unordered_set>
#include <utility>

#include "openvino/core/except.hpp"
#include "openvino/core/model.hpp"
#include "openvino/op/util/multi_subgraph_base.hpp"
#include "openvino/op/util/variable.hpp"
#include "openvino/runtime/make_tensor.hpp"
#include "openvino/runtime/tensor.hpp"
#include "plugin/infer_request_state.hpp"
#include "runtime/backend_runtime.hpp"
#include "runtime/stateful_execution.hpp"

namespace ov {
namespace gfx_plugin {
namespace {

ov::Shape default_state_shape(const ov::PartialShape& shape) {
    return shape.is_static() ? shape.to_shape() : ov::Shape{0};
}

void ensure_host_tensor(StatefulVariableTensorState& slot,
                        const ov::element::Type& type,
                        const ov::Shape& shape) {
    OPENVINO_ASSERT(type != ov::element::dynamic,
                    "GFX: variable state element type is dynamic");
    if (!slot.host_tensor ||
        slot.host_tensor.get_element_type() != type ||
        slot.host_tensor.get_shape() != shape) {
        slot.host_tensor = ov::Tensor(type, shape);
    }
}

void zero_host_tensor(ov::Tensor& tensor) {
    if (tensor && tensor.get_byte_size() != 0) {
        std::memset(tensor.data(), 0, tensor.get_byte_size());
    }
}

class GfxVariableState final : public ov::IVariableState {
public:
    GfxVariableState(InferRequestState& request_state,
                     std::string variable_id,
                     ov::PartialShape expected_shape,
                     ov::element::Type expected_type,
                     BackendResources resources)
        : ov::IVariableState(variable_id),
          m_request_state(&request_state),
          m_expected_shape(std::move(expected_shape)),
          m_expected_type(expected_type),
          m_resources(resources) {}

    void reset() override {
        auto& slot = slot_ref();
        const auto shape = slot.host_tensor ? slot.host_tensor.get_shape()
                                            : default_state_shape(m_expected_shape);
        ensure_host_tensor(slot, resolved_type(slot), shape);
        zero_host_tensor(slot.host_tensor);
        slot.host_dirty = true;
        slot.host_stale = false;
        slot.initialized = false;
    }

    void set_state(const ov::SoPtr<ov::ITensor>& state) override {
        OPENVINO_ASSERT(state._ptr, "GFX: variable state tensor is null");
        const ov::Tensor tensor = ov::make_tensor(state);
        OPENVINO_ASSERT(m_expected_shape.compatible(ov::PartialShape(tensor.get_shape())),
                        "Wrong tensor shape: ",
                        tensor.get_shape(),
                        " is not compatible with expected: ",
                        m_expected_shape,
                        " in a variable with ID: ",
                        get_name());
        OPENVINO_ASSERT(m_expected_type.compatible(tensor.get_element_type()),
                        "Wrong tensor type: ",
                        tensor.get_element_type(),
                        " expected: ",
                        m_expected_type,
                        " in a variable with ID: ",
                        get_name());
        auto& slot = slot_ref();
        ensure_host_tensor(slot, tensor.get_element_type(), tensor.get_shape());
        OPENVINO_ASSERT(tensor.get_byte_size() == slot.host_tensor.get_byte_size(),
                        "Blob size of tensors are not equal. Variable with ID: ",
                        get_name());
        if (tensor.get_byte_size() != 0) {
            std::memcpy(slot.host_tensor.data(), tensor.data(), tensor.get_byte_size());
        }
        slot.expected_shape = m_expected_shape;
        slot.expected_type = m_expected_type;
        slot.host_dirty = true;
        slot.host_stale = false;
        slot.initialized = false;
    }

    ov::SoPtr<ov::ITensor> get_state() const override {
        auto& slot = slot_ref();
        if (!slot.host_tensor) {
            ensure_host_tensor(slot, resolved_type(slot), default_state_shape(m_expected_shape));
            zero_host_tensor(slot.host_tensor);
        }
        sync_stateful_variable_host(slot, get_name(), m_resources);
        return ov::get_tensor_impl(slot.host_tensor);
    }

private:
    StatefulVariableTensorState& slot_ref() const {
        OPENVINO_ASSERT(m_request_state, "GFX: variable state request is gone");
        auto& slot = m_request_state->variable_states[get_name()];
        if (slot.expected_type == ov::element::dynamic) {
            slot.expected_type = m_expected_type;
        }
        if (slot.expected_shape.is_dynamic() && m_expected_shape.is_static()) {
            slot.expected_shape = m_expected_shape;
        }
        return slot;
    }

    ov::element::Type resolved_type(const StatefulVariableTensorState& slot) const {
        if (slot.host_tensor) {
            return slot.host_tensor.get_element_type();
        }
        if (slot.expected_type != ov::element::dynamic) {
            return slot.expected_type;
        }
        return m_expected_type;
    }

    InferRequestState* m_request_state = nullptr;
    ov::PartialShape m_expected_shape;
    ov::element::Type m_expected_type = ov::element::dynamic;
    BackendResources m_resources;
};

void collect_variables(const std::shared_ptr<const ov::Model>& model,
                       InferRequestState& state,
                       std::unordered_set<std::string>& seen,
                       const BackendResources& resources) {
    if (!model) {
        return;
    }
    for (const auto& op : model->get_ordered_ops()) {
        if (auto multi = ov::as_type_ptr<const ov::op::util::MultiSubGraphOp>(op)) {
            for (const auto& subgraph : multi->get_functions()) {
                collect_variables(subgraph, state, seen, resources);
            }
        }
    }
    for (const auto& variable : model->get_variables()) {
        const auto& info = variable->get_info();
        if (!seen.insert(info.variable_id).second) {
            continue;
        }
        auto& slot = state.variable_states[info.variable_id];
        slot.expected_shape = info.data_shape;
        slot.expected_type = info.data_type;
        ensure_host_tensor(slot, info.data_type, default_state_shape(info.data_shape));
        zero_host_tensor(slot.host_tensor);
        slot.host_dirty = true;
        slot.host_stale = false;
        slot.initialized = false;

        auto variable_state = std::make_shared<GfxVariableState>(
            state, info.variable_id, info.data_shape, info.data_type, resources);
        state.variable_state_objects.emplace_back(variable_state, nullptr);
    }
}

}  // namespace

void initialize_variable_states(InferRequestState& state,
                                const std::shared_ptr<const ov::Model>& model,
                                const BackendResources& resources) {
    state.variable_state_objects.clear();
    std::unordered_set<std::string> seen;
    collect_variables(model, state, seen, resources);
}

std::vector<ov::SoPtr<ov::IVariableState>>
query_variable_states(const InferRequestState& state) {
    return state.variable_state_objects;
}

}  // namespace gfx_plugin
}  // namespace ov
