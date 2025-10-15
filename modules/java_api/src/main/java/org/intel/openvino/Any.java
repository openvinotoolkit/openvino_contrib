// Copyright (C) 2020-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

package org.intel.openvino;

import java.util.List;

/** This class represents an object to work with different types. */
public class Any extends Wrapper {

    public Any(long addr) {
        super(addr);
    }

    public int asInt() {
        return asInt(nativeObj);
    }

    public String asString() {
        return asString(nativeObj);
    }

    public List<String> asList() {
        return asList(nativeObj);
    }

    /*----------------------------------- native methods -----------------------------------*/
    private static native int asInt(long addr);

    private static native String asString(long addr);

    private static native List<String> asList(long addr);

    @Override
    protected native void delete(long nativeObj);
}
