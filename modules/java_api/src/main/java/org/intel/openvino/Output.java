// Copyright (C) 2020-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

package org.intel.openvino;

/** A handle for one of a node's outputs. */
public class Output extends Wrapper {

    public Output(long addr) {
        super(addr);
    }

    /** Returns any tensor names associated with this output */
    public String get_any_name() {
        return GetAnyName(nativeObj);
    }

    /** Returns the shape of the output referred to by this output handle. */
    public int[] get_shape() {
        return GetShape(nativeObj);
    }

    /** Returns the partial shape of the output referred to by this output handle. */
    public PartialShape get_partial_shape() {
        return new PartialShape(GetPartialShape(nativeObj));
    }

    /** Returns the element type of the output referred to by this output handle. */
    public ElementType get_element_type() {
        return ElementType.valueOf(GetElementType(nativeObj));
    }

    /*----------------------------------- native methods -----------------------------------*/
    private static native String GetAnyName(long addr);

    private static native int[] GetShape(long addr);

    private static native long GetPartialShape(long addr);

    private static native int GetElementType(long addr);

    @Override
    protected native void delete(long nativeObj);
}
