// Copyright (C) 2020-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

package org.intel.openvino;

public enum ResizeAlgorithm {
    RESIZE_LINEAR(0),
    RESIZE_CUBIC(1),
    RESIZE_NEAREST(2);

    private int value;

    private ResizeAlgorithm(int value) {
        this.value = value;
    }

    public int getValue() {
        return value;
    }
}
