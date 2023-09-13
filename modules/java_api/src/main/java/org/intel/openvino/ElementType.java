// Copyright (C) 2020-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

package org.intel.openvino;

import java.util.HashMap;
import java.util.Map;

public enum ElementType {
    undefined(0),
    dynamic(1),
    bool(2),
    bf16(3),
    f16(4),
    f32(5),
    f64(6),
    i4(7),
    i8(8),
    i16(9),
    i32(10),
    i64(11),
    u1(12),
    u4(13),
    u8(14),
    u16(15),
    u32(16),
    u64(17);

    private int value;
    private static Map<Integer, ElementType> map = new HashMap<Integer, ElementType>();

    static {
        for (ElementType type : ElementType.values()) {
            map.put(type.value, type);
        }
    }

    private ElementType(int value) {
        this.value = value;
    }

    public int getValue() {
        return value;
    }

    static ElementType valueOf(int value) {
        return map.get(value);
    }
}
