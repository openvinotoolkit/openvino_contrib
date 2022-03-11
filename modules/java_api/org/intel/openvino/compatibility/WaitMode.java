// Copyright (C) 2020-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

package org.intel.openvino.compatibility;

public enum WaitMode {
    RESULT_READY(-1),
    STATUS_ONLY(0);

    private int value;

    private WaitMode(int value) {
        this.value = value;
    }

    public int getValue() {
        return value;
    }
}
