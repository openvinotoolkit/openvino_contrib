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
    u2(13),
    u3(14),
    u4(15),
    u6(16),
    u8(17),
    u16(18),
    u32(19),
    u64(20),
    nf4(21),
    f8e4m3(22),
    f8e5m2(23),
    string(24),
    f4e2m1(25);

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
