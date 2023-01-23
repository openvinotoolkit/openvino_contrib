// Copyright (C) 2020-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

package org.intel.openvino;

/**
 * Information about model's input tensor. If all information is already included to loaded model,
 * this info may not be needed. However it can be set to specify additional information about model,
 * like 'layout'.
 */
public class InputModelInfo extends Wrapper {

    public InputModelInfo(long addr) {
        super(addr);
    }

    /**
     * Set layout for model's input tensor
     *
     * @param layout Layout for model's input tensor.
     * @return Reference to 'this' to allow chaining with other calls in a builder-like manner
     */
    public InputModelInfo set_layout(Layout layout) {
        SetLayout(nativeObj, layout.nativeObj);
        return this;
    }

    /*----------------------------------- native methods -----------------------------------*/
    private static native void SetLayout(long addr, long layout);

    @Override
    protected native void delete(long nativeObj);
}
