// Copyright (C) 2018-2026 Intel Corporation
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
     * Constructs a {@link Tensor} of a byte-wide element type from the given byte array.
     *
     * <p>This is the OpenCV-free way to feed a CPU-side image buffer (for example a {@code u8} RGB
     * frame produced from an Android {@code ImageProxy}) into inference: the data is copied into
     * the tensor, so the source array can be reused right after. Only byte-wide element types are
     * accepted, namely {@link ElementType#u8} and {@link ElementType#i8}.
     *
     * @param type element type of the tensor, must be {@link ElementType#u8} or {@link
     *     ElementType#i8}
     * @param dims shape of the tensor
     * @param data a byte array containing the tensor data; its length must equal the product of
     *     {@code dims}
     */
    public Tensor(ElementType type, int[] dims, byte[] data) {
        super(TensorByte(type.getValue(), dims, data));
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

    /**
     * Returns the tensor data as a byte array.
     *
     * <p>Supported only for byte-wide element types ({@link ElementType#u8} and {@link
     * ElementType#i8}), which cannot be read back through {@link #data()} or {@link #as_int()}.
     * {@code u8} values are returned in their raw two's-complement byte form, so a pixel value of
     * {@code 200} reads back as {@code (byte) 200 == -56}; mask with {@code & 0xFF} to recover the
     * unsigned value.
     *
     * @return the tensor data as a byte array
     */
    public byte[] as_byte() {
        return asByte(nativeObj);
    }

    /*----------------------------------- native methods -----------------------------------*/
    private static native long TensorCArray(int type, int[] shape, long cArray);

    private static native long TensorFloat(int[] shape, float[] data);

    private static native long TensorByte(int type, int[] shape, byte[] data);

    private static native long TensorInt(int[] shape, int[] data);

    private static native long TensorLong(int[] shape, long[] data);

    private static native int[] GetShape(long addr);

    private static native int GetElementType(long addr);

    private static native float[] asFloat(long addr);

    private static native int[] asInt(long addr);

    private static native byte[] asByte(long addr);

    private static native int GetSize(long addr);

    @Override
    protected native void delete(long nativeObj);
}
