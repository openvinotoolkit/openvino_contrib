// Copyright (C) 2020-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

package org.intel.openvino;

/** Information about user's input tensor. By default, it will be initialized to same data (type/shape/etc) as
 * model's input parameter. User application can override particular parameters (like 'element_type') according to
 * application's data and specify appropriate conversions in pre-processing steps
 */
public class InputTensorInfo extends Wrapper {

    public InputTensorInfo(long addr) {
        super(addr);
    }

    /** Set element type for user's input tensor
     *
     * @param type Element type for user's input tensor.
     *
     * @return Reference to 'this' to allow chaining with other calls in a builder-like manner
     */
    public InputTensorInfo set_element_type(ElementType type) {
        SetElementType(nativeObj, type.getValue());
        return this;
    }

    /** Set layout for user's input tensor
     *
     * @param layout Layout for user's input tensor.
     *
     * @return Reference to 'this' to allow chaining with other calls in a builder-like manner
     */
    public InputTensorInfo set_layout(Layout layout) {
        SetLayout(nativeObj, layout.nativeObj);
        return this;
    }

    /** By default, input image shape is inherited from model input shape. Use this method to specify different
     * width and height of user's input image. In case if input image size is not known, use
     * {@link InputTensorInfo#set_spatial_dynamic_shape} method.
     *
     * @param height Set fixed user's input image height.
     *
     * @param width Set fixed user's input image width.
     *
     * @return Reference to 'this' to allow chaining with other calls in a builder-like manner.
    */
    public InputTensorInfo set_spatial_static_shape(int height, int width) {
        SetSpatialStaticShape(nativeObj, height, width);
        return this;
    }

    /** By default, input image shape is inherited from model input shape. This method specifies that user's
     * input image has dynamic spatial dimensions (width and height). This can be useful for adding resize preprocessing
     * from any input image to model's expected dimensions.
     *
     * @return Reference to 'this' to allow chaining with other calls in a builder-like manner.
     */
    public InputTensorInfo set_spatial_dynamic_shape() {
        SetSpatialDynamicShape(nativeObj);
        return this;
    }

    /*----------------------------------- native methods -----------------------------------*/
    private static native void SetElementType(long addr, int type);

    private static native void SetLayout(long addr, long layout);

    private static native void SetSpatialStaticShape(long addr, int height, int width);

    private static native void SetSpatialDynamicShape(long addr);

    @Override
    protected native void delete(long nativeObj);
}
