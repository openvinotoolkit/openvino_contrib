// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

package org.intel.openvino;

/**
 * Preprocessing steps. Each step typically intends adding of some operation to input parameter User
 * application can specify sequence of preprocessing steps in a builder-like manner.
 */
public class PreProcessSteps extends Wrapper {

    public PreProcessSteps(long addr) {
        super(addr);
    }

    /**
     * Add resize operation to model's dimensions.
     *
     * @param alg Resize algorithm.
     * @return Reference to 'this' to allow chaining with other calls in a builder-like manner.
     */
    public PreProcessSteps resize(ResizeAlgorithm alg) {
        Resize(nativeObj, alg.getValue());
        return this;
    }

    /**
     * Add a scale preprocessing operation: divide each element by the given value.
     *
     * <p>For example, {@code scale(255.0f)} maps {@code u8} pixels in {@code [0, 255]} to floats in
     * {@code [0, 1]}. Scaling happens inside the compiled model, so the application can keep
     * feeding raw {@code u8} pixels.
     *
     * @param value Scale value applied to every element.
     * @return Reference to 'this' to allow chaining with other calls in a builder-like manner.
     */
    public PreProcessSteps scale(float value) {
        Scale(nativeObj, value);
        return this;
    }

    /**
     * Add a per-channel scale preprocessing operation: divide each channel by its own value.
     *
     * <p>Requires the 'C' dimension to be set in the input tensor layout. The number of values must
     * match the number of channels.
     *
     * @param values Per-channel scale values.
     * @return Reference to 'this' to allow chaining with other calls in a builder-like manner.
     */
    public PreProcessSteps scale(float[] values) {
        ScaleValues(nativeObj, values);
        return this;
    }

    /**
     * Add a mean preprocessing operation: subtract the given value from each element.
     *
     * @param value Mean value subtracted from every element.
     * @return Reference to 'this' to allow chaining with other calls in a builder-like manner.
     */
    public PreProcessSteps mean(float value) {
        Mean(nativeObj, value);
        return this;
    }

    /**
     * Add a per-channel mean preprocessing operation: subtract each channel's own value.
     *
     * <p>Requires the 'C' dimension to be set in the input tensor layout. The number of values must
     * match the number of channels. When combined with {@link #scale(float[])}, mean subtraction is
     * applied before scaling, matching the {@code (x - mean) / scale} normalization convention.
     *
     * @param values Per-channel mean values.
     * @return Reference to 'this' to allow chaining with other calls in a builder-like manner.
     */
    public PreProcessSteps mean(float[] values) {
        MeanValues(nativeObj, values);
        return this;
    }

    /**
     * Add an element-type conversion preprocessing operation.
     *
     * <p>For example, converting a {@code u8} input tensor to {@code f32} before scaling - OpenVINO
     * requires scale/mean to run on a real (floating-point) type, so a {@code u8} camera frame must
     * be converted first: {@code convert_element_type(f32).scale(255)}.
     *
     * @param type Target element type to convert to.
     * @return Reference to 'this' to allow chaining with other calls in a builder-like manner.
     */
    public PreProcessSteps convert_element_type(ElementType type) {
        ConvertElementType(nativeObj, type.getValue());
        return this;
    }

    /**
     * Add a layout conversion (transpose) preprocessing operation.
     *
     * <p>For example, converting a user's {@code NHWC} tensor to the model's {@code NCHW} layout.
     * Both the user tensor layout and the target layout must be known (set via {@link
     * InputTensorInfo#set_layout} and {@link InputModelInfo#set_layout} respectively).
     *
     * @param dstLayout Target layout to convert to.
     * @return Reference to 'this' to allow chaining with other calls in a builder-like manner.
     */
    public PreProcessSteps convert_layout(Layout dstLayout) {
        ConvertLayout(nativeObj, dstLayout.nativeObj);
        return this;
    }

    /*---------------------------------- native methods -----------------------------------*/
    private static native void Resize(long nativeObj, int alg);

    private static native void Scale(long nativeObj, float value);

    private static native void ScaleValues(long nativeObj, float[] values);

    private static native void Mean(long nativeObj, float value);

    private static native void MeanValues(long nativeObj, float[] values);

    private static native void ConvertElementType(long nativeObj, int type);

    private static native void ConvertLayout(long nativeObj, long dstLayout);

    @Override
    protected native void delete(long nativeObj);
}
