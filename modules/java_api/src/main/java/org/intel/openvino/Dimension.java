// Copyright (C) 2020-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

package org.intel.openvino;

/**
 * Class representing a dimension, which may be dynamic (undetermined until runtime), in a shape or
 * shape-like object.
 */
public class Dimension extends Wrapper {

    protected Dimension(long addr) {
        super(addr);
    }

    /**
     * Convert this dimension to int value. This dimension must be static and non-negative. Throws
     * std::invalid_argument If this dimension is dynamic or negative.
     */
    public int get_length() {
        return getLength(nativeObj);
    }

    /*----------------------------------- native methods -----------------------------------*/
    private static native int getLength(long addr);
}
