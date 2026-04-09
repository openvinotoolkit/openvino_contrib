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
     * Constructs an Integer {@link Tensor} from the given int array.
     *
     * @param dims shape of the tensor
     * @param data an integer array containing the tensor data
     */
    public Tensor(int[] dims, int[] data) {
        super(TensorInt(dims, data));
    }

    /**
     * Constructs a Long {@link Tensor} from the given long array.
     *
     * @param dims shape of the tensor
     * @param data a long array containing the tensor data
     */
    public Tensor(int[] dims, long[] data) {
        super(TensorLong(dims, data));
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

    /** Returns the tensor element type. */
    public ElementType get_element_type() {
        return ElementType.valueOf(GetElementType(nativeObj));
    }

    /** Returns a tensor data as floating point array. */
    public float[] data() {
        return asFloat(nativeObj);
    }

    /** Returns the tensor data as an integer array. */
    public int[] as_int() {
        return asInt(nativeObj);
    }

    /*----------------------------------- native methods -----------------------------------*/
    private static native long TensorCArray(int type, int[] shape, long cArray);

    private static native long TensorFloat(int[] shape, float[] data);

    private static native long TensorInt(int[] shape, int[] data);

    private static native long TensorLong(int[] shape, long[] data);

    private static native int[] GetShape(long addr);

    private static native int GetElementType(long addr);

    private static native float[] asFloat(long addr);

    private static native int[] asInt(long addr);

    private static native int GetSize(long addr);

    @Override
    protected native void delete(long nativeObj);
}
