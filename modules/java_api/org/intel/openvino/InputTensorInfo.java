// Copyright (C) 2020-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

package org.intel.openvino;

public class InputTensorInfo extends Wrapper {

    public InputTensorInfo(long addr) {
        super(addr);
    }

    public InputTensorInfo set_element_type(ElementType type) {
        SetElementType(nativeObj, type.getValue());
        return this;
    }

    public InputTensorInfo set_layout(Layout layout) {
        SetLayout(nativeObj, layout.nativeObj);
        return this;
    }

    public InputTensorInfo set_spatial_static_shape(int height, int width) {
        SetSpatialStaticShape(nativeObj, height, width);
        return this;
    }

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
