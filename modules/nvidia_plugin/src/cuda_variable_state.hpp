// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <mutex>

#include "cuda/runtime.hpp"
#include "openvino/op/util/variable.hpp"
#include "openvino/runtime/ivariable_state.hpp"
#include "openvino/runtime/make_tensor.hpp"

namespace ov {
namespace nvidia_gpu {

/**
 * @brief GPU-resident implementation of ov::IVariableState.
 *
 * State buffer lives on the CUDA device to avoid H2D/D2H copies during inference.
 * ReadValue/Assign ops access it via device_buffer() (fast D2D path).
 * get_state()/set_state() copy to/from host (slow path, for user API only).
 */
class CudaVariableState : public ov::IVariableState {
public:
    using Ptr = std::shared_ptr<CudaVariableState>;

    explicit CudaVariableState(const ov::op::util::VariableInfo& variable_info)
        : ov::IVariableState(variable_info.variable_id),
          data_shape_(variable_info.data_shape),
          element_type_(variable_info.data_type),
          is_reset_(true) {
        if (!data_shape_.is_dynamic()) {
            current_shape_ = data_shape_.to_shape();
            auto byte_size = element_type_.size() * ov::shape_size(current_shape_);
            if (byte_size > 0) {
                device_buffer_ = CUDA::DefaultStream::stream().malloc(byte_size);
                current_byte_size_ = byte_size;
                CUDA::DefaultStream::stream().memset(device_buffer_, 0, byte_size);
            }
        }
        // Initialize m_state with an empty host tensor
        auto shape = data_shape_.is_dynamic() ? ov::Shape{0} : current_shape_;
        m_state = ov::SoPtr<ov::ITensor>{ov::make_tensor(element_type_, shape), nullptr};
    }

    void set_state(const ov::SoPtr<ov::ITensor>& state) override {
        std::lock_guard<std::mutex> lock(mutex_);
        OPENVINO_ASSERT(data_shape_.compatible(state->get_shape()),
                        "Wrong tensor shape: ", state->get_shape(),
                        " is not compatible with expected: ", data_shape_,
                        " in variable: ", get_name());
        OPENVINO_ASSERT(element_type_.compatible(state->get_element_type()),
                        "Wrong tensor type: ", state->get_element_type(),
                        " expected: ", element_type_,
                        " in variable: ", get_name());
        current_shape_ = state->get_shape();
        auto byte_size = state->get_byte_size();
        ensure_device_buffer(byte_size);
        if (byte_size > 0) {
            CUDA::DefaultStream::stream().upload(
                CUDA::DevicePointer<void*>{device_buffer_.get()},
                state->data(), byte_size);
        }
        is_reset_ = false;
    }

    ov::SoPtr<ov::ITensor> get_state() const override {
        std::lock_guard<std::mutex> lock(mutex_);
        auto tensor_ptr = ov::make_tensor(element_type_, current_shape_);
        if (current_byte_size_ > 0 && device_buffer_.get()) {
            CUDA::DefaultStream::stream().download(
                tensor_ptr->data(),
                CUDA::DevicePointer<const void*>{device_buffer_.get()},
                current_byte_size_);
        }
        return ov::SoPtr<ov::ITensor>{tensor_ptr, nullptr};
    }

    void reset() override {
        std::lock_guard<std::mutex> lock(mutex_);
        if (current_byte_size_ > 0 && device_buffer_.get()) {
            CUDA::DefaultStream::stream().memset(device_buffer_, 0, current_byte_size_);
        }
        is_reset_ = true;
    }

    // --- Fast path for ReadValue/Assign (GPU-to-GPU, no host involvement) ---

    void* device_buffer_ptr() const { return device_buffer_.get(); }
    std::size_t device_buffer_byte_size() const { return current_byte_size_; }
    const ov::Shape& shape() const { return current_shape_; }
    ov::element::Type element_type() const { return element_type_; }
    bool is_reset_state() const { return is_reset_; }

    /**
     * @brief Update state from a device pointer (D2D copy). Called by AssignOp.
     * Reallocates if the new shape requires more memory.
     */
    void update_device_state(const CUDA::Stream& stream,
                             CUDA::DevicePointer<const void*> src,
                             const ov::Shape& new_shape) {
        std::lock_guard<std::mutex> lock(mutex_);
        auto byte_size = element_type_.size() * ov::shape_size(new_shape);
        ensure_device_buffer(byte_size);
        current_shape_ = new_shape;
        if (byte_size > 0) {
            stream.transfer(CUDA::DevicePointer<void*>{device_buffer_.get()}, src, byte_size);
        }
        is_reset_ = false;
    }

    /**
     * @brief Copy state to a device pointer (D2D copy). Called by ReadValueOp.
     */
    void read_device_state(const CUDA::Stream& stream,
                           CUDA::DevicePointer<void*> dst,
                           std::size_t dst_byte_size) const {
        std::lock_guard<std::mutex> lock(mutex_);
        if (current_byte_size_ > 0 && device_buffer_.get()) {
            auto copy_size = std::min(dst_byte_size, current_byte_size_);
            stream.transfer(dst, CUDA::DevicePointer<const void*>{device_buffer_.get()}, copy_size);
        }
    }

private:
    void ensure_device_buffer(std::size_t required_bytes) {
        if (required_bytes > current_byte_size_ || !device_buffer_.get()) {
            device_buffer_ = CUDA::DefaultStream::stream().malloc(required_bytes);
            current_byte_size_ = required_bytes;
        }
    }

    mutable std::mutex mutex_;
    ov::PartialShape data_shape_;
    ov::element::Type element_type_;
    ov::Shape current_shape_;
    std::size_t current_byte_size_ = 0;
    CUDA::DefaultAllocation device_buffer_{nullptr};
    bool is_reset_ = true;
};

}  // namespace nvidia_gpu
}  // namespace ov
