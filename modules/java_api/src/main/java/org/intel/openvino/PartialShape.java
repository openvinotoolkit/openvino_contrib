// Copyright (C) 2020-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

package org.intel.openvino;

/** This class represents the definitions and operations about partial shape. */
public class PartialShape extends Wrapper {

    public PartialShape(long addr) {
        super(addr);
    }

    /**
     * Get the dimension at specified index of a partial shape.
     *
     * @param index The index of dimension.
     * @return The particular dimension of partial shape.
     */
    public Dimension get_dimension(int index) {
        return new Dimension(GetDimension(nativeObj, index));
    }

    /** Returns the max bounding shape. */
    public int[] get_max_shape() {
        return GetMaxShape(nativeObj);
    }

    /** Returns the min bounding shape. */
    public int[] get_min_shape() {
        return GetMinShape(nativeObj);
    }

    /*----------------------------------- native methods -----------------------------------*/
    private static native long GetDimension(long addr, int index);

    private static native int[] GetMaxShape(long addr);

    private static native int[] GetMinShape(long addr);
}
