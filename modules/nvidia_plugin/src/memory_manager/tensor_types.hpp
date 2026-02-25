// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <cstdint>
#include <error.hpp>
#include <iostream>
#include <memory>
#include <optional>
#include <sstream>
#include <unordered_map>
#include <utility>

namespace ov {
namespace nvidia_gpu {

using BufferID = unsigned;

class TensorID {
public:
    using Ptr = std::shared_ptr<TensorID>;

    explicit TensorID(BufferID buffer_id) : id_{buffer_id} {}

    /**
     * Returns tensor id
     */
    [[nodiscard]] BufferID GetId() const { return id_; }

    /**
     * Returns offset of the current tensor within buffer (root tensor)
     */
    [[nodiscard]] unsigned GetOffset() const {
        unsigned offset = offset_;
        if (parent_tensor_) {
            offset += parent_tensor_->GetOffset();
        }
        return offset;
    }

    /**
     * Returns root tensor (buffer, allocation object)
     */
    [[nodiscard]] const TensorID& GetBuffer() const {
        if (parent_tensor_) {
            return parent_tensor_->GetBuffer();
        }
        return *this;
    }

    /**
     * Sets parent tensor (parent buffer) and offset for it
     * @param parent_tensor Parent tensor
     * @param offset Offset within parent tensor
     */
    void SetParent(std::shared_ptr<TensorID> parent_tensor, unsigned offset) {
        parent_tensor_ = std::move(parent_tensor);
        offset_ = offset;
    }

    bool operator==(const TensorID& t) const { return id_ == t.id_; }

    bool operator!=(const TensorID& t) const { return !(operator==(t)); }

private:
    BufferID id_{};
    std::shared_ptr<TensorID> parent_tensor_;
    unsigned offset_{};
};

inline std::ostream& operator<<(std::ostream& s, const TensorID& t) {
    s << "ID: " << t.GetId() << ", ";
    s << "BufferID: " << t.GetBuffer().GetId() << ", ";
    s << "Offset: " << t.GetOffset();
    return s;
}

inline std::string to_string(const TensorID& x) {
    std::ostringstream ss;
    ss << x;
    return ss.str();
}

}  // namespace nvidia_gpu
}  // namespace ov
