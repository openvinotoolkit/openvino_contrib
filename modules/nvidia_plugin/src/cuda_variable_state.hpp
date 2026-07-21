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
 * ReadValue/Assign ops access it via device_buffer() (fast D2D path) on the
 * per-request stream. get_state()/set_state()/reset() copy to/from host (slow
 * path, for the user API only).
 *
 * Thread-safety: every access to the mutable fields (device_buffer_,
 * current_shape_, capacity_bytes_, is_reset_) is guarded by mutex_, including the
 * fast-path accessors. The slow-path host copies run on the synchronizing default
 * stream, which is ordered against the (blocking) per-request streams, so a
 * set_state/reset cannot tear an in-flight D2D copy on the device side.
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
        } else {
            // For dynamic shapes, create an initial shape with 0 for dynamic dims.
            // This preserves the rank (e.g. KV-cache: {1,32,?,64} → {1,32,0,64}).
            current_shape_.resize(data_shape_.rank().is_static() ? data_shape_.rank().get_length() : 0);
            for (size_t i = 0; i < current_shape_.size(); ++i) {
                current_shape_[i] = data_shape_[i].is_static() ? data_shape_[i].get_length() : 0;
            }
        }
        // The shape the state returns to on reset() (empty KV-cache for dynamic vars).
        initial_shape_ = current_shape_;
        auto byte_size = logical_byte_size();
        if (byte_size > 0) {
            device_buffer_ = CUDA::DefaultStream::stream().malloc(byte_size);
            capacity_bytes_ = byte_size;
            CUDA::DefaultStream::stream().memset(device_buffer_, 0, byte_size);
        }
        // Initialize m_state with a host tensor matching current shape
        m_state = ov::SoPtr<ov::ITensor>{ov::make_tensor(element_type_, current_shape_), nullptr};
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
        auto byte_size = logical_byte_size();
        if (byte_size > 0 && device_buffer_.get()) {
            CUDA::DefaultStream::stream().download(
                tensor_ptr->data(),
                CUDA::DevicePointer<const void*>{device_buffer_.get()},
                byte_size);
        }
        return ov::SoPtr<ov::ITensor>{tensor_ptr, nullptr};
    }

    void reset() override {
        std::lock_guard<std::mutex> lock(mutex_);
        // Return to the initial (post-reset) shape — e.g. an empty KV-cache —
        // so get_state()/ReadValue observe the correct size, not the grown one.
        current_shape_ = initial_shape_;
        if (capacity_bytes_ > 0 && device_buffer_.get()) {
            CUDA::DefaultStream::stream().memset(device_buffer_, 0, capacity_bytes_);
        }
        is_reset_ = true;
    }

    // --- Fast path for ReadValue/Assign (GPU-to-GPU, no host involvement) ---
    // These accessors are mutex-guarded: ReadValue/Assign read them on the
    // inference thread while the user-facing set_state/reset/update_device_state
    // may mutate the same fields from another thread. shape() returns BY VALUE so
    // the caller never reads current_shape_'s storage after the lock is released.

    void* device_buffer_ptr() const {
        std::lock_guard<std::mutex> lock(mutex_);
        return device_buffer_.get();
    }
    std::size_t device_buffer_byte_size() const {
        std::lock_guard<std::mutex> lock(mutex_);
        return logical_byte_size();
    }
    ov::Shape shape() const {
        std::lock_guard<std::mutex> lock(mutex_);
        return current_shape_;
    }
    ov::element::Type element_type() const { return element_type_; }  // immutable after construction
    bool is_reset_state() const {
        std::lock_guard<std::mutex> lock(mutex_);
        return is_reset_;
    }

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
        auto byte_size = logical_byte_size();
        if (byte_size > 0 && device_buffer_.get()) {
            auto copy_size = std::min(dst_byte_size, byte_size);
            stream.transfer(dst, CUDA::DevicePointer<const void*>{device_buffer_.get()}, copy_size);
        }
    }

private:
    // Logical size of the state as currently shaped (distinct from the allocation
    // capacity, which only ever grows). All copies use this so a shrunk shape never
    // over-reads the buffer.
    std::size_t logical_byte_size() const { return element_type_.size() * ov::shape_size(current_shape_); }

    void ensure_device_buffer(std::size_t required_bytes) {
        if (required_bytes > capacity_bytes_ || !device_buffer_.get()) {
            device_buffer_ = CUDA::DefaultStream::stream().malloc(required_bytes);
            capacity_bytes_ = required_bytes;
        }
    }

    mutable std::mutex mutex_;
    ov::PartialShape data_shape_;
    ov::element::Type element_type_;
    ov::Shape initial_shape_;  // shape to restore on reset()
    ov::Shape current_shape_;
    std::size_t capacity_bytes_ = 0;  // allocated device buffer size (high-water mark)
    CUDA::DefaultAllocation device_buffer_{nullptr};
    bool is_reset_ = true;
};

}  // namespace nvidia_gpu
}  // namespace ov
