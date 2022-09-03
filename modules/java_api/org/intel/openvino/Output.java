// Copyright (C) 2020-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

package org.intel.openvino;

public class Output extends Wrapper {

    public Output(long addr) {
        super(addr);
    }

    public String get_any_name() {
        return GetAnyName(nativeObj);
    }

    public int[] get_shape() {
        return GetShape(nativeObj);
    }

    /*----------------------------------- native methods -----------------------------------*/
    private static native String GetAnyName(long addr);

    private static native int[] GetShape(long addr);

    @Override
    protected native void delete(long nativeObj);
}
