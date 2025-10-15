// Copyright (C) 2020-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

package org.intel.openvino;

import java.util.HashMap;
import java.util.Map;

public enum ElementType {
    dynamic(0),
    undefined(dynamic.value),
    bool(1),
    bf16(2),
    f16(3),
    f32(4),
    f64(5),
    i4(6),
    i8(7),
    i16(8),
    i32(9),
    i64(10),
    u1(11),
    u2(12),
    u3(13),
    u4(14),
    u6(15),
    u8(16),
    u16(17),
    u32(18),
    u64(19),
    nf4(20),
    f8e4m3(21),
    f8e5m2(22),
    string(23),
    f4e2m1(24),
    f8e8m0(25);

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
