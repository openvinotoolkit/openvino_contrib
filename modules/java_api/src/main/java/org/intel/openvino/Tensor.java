// Copyright (C) 2020-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

package org.intel.openvino;

/**
 * Tensor API holding host memory
 *
 * <p>It can throw exceptions safely for the application, where it is properly handled.
 */
public class Tensor extends Wrapper {

    public Tensor(long addr) {
        super(addr);
    }

    public Tensor(ElementType type, int[] dims, long cArray) {
        super(TensorCArray(type.getValue(), dims, cArray));
    }

    public Tensor(int[] dims, float[] data) {
        super(TensorFloat(dims, data));
    }

    /**
     * Returns the total number of elements (a product of all the dims or 1 for scalar)
     *
     * @return The total number of elements
     */
    public int get_size() {
        return GetSize(nativeObj);
    }

    /** Returns a tensor shape */
    public int[] get_shape() {
        return GetShape(nativeObj);
    }

    /** Returns a tensor data as floating point array. */
    public float[] data() {
        return asFloat(nativeObj);
    }

    /*----------------------------------- native methods -----------------------------------*/
    private static native long TensorCArray(int type, int[] shape, long cArray);

    private static native long TensorFloat(int[] shape, float[] data);

    private static native int[] GetShape(long addr);

    private static native float[] asFloat(long addr);

    private static native int GetSize(long addr);

    @Override
    protected native void delete(long nativeObj);
}
