// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <cstdint>
#include <memory>
#include <unordered_map>
#include <iostream>

namespace CUDAPlugin {

using BufferID = unsigned;

struct TensorID {
    BufferID buffer_id{};
    unsigned offset{};

    TensorID() = default;

    TensorID(BufferID buffer_id)
        : buffer_id{buffer_id} {
    }

    TensorID(BufferID buffer_id, unsigned offset)
        : buffer_id{buffer_id}
        , offset{offset} {
    }

    bool operator==(const TensorID& t) const {
        return buffer_id == t.buffer_id && offset == t.offset;
    }

    bool operator!=(const TensorID& t) const {
        return !(operator==(t));
    }
};

inline
std::ostream& operator<<(std::ostream& s, const TensorID& t) {
    s << "Id: " << t.buffer_id << ", ";
    s << "Offset: " << t.offset;
    return s;
}

}

namespace std {

template <>
struct hash<CUDAPlugin::TensorID> {
    std::size_t operator()(const CUDAPlugin::TensorID& t) const {
        return std::hash<unsigned>()(t.buffer_id) ^ std::hash<unsigned>()(t.offset);
    }
};

}
