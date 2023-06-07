// Copyright (C) 2020-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

package org.intel.openvino;

/** This class represents an object to work with different types. */
public class Any extends Wrapper {

    static {
        try {
            Class.forName("org.intel.openvino.NativeLibrary");
        } catch (ClassNotFoundException e) {
            throw new RuntimeException("Failed to load OpenVINO native libraries");
        }
    }

    public Any(long addr) {
        super(addr);
    }

    public int asInt() {
        return asInt(nativeObj);
    }

    /*----------------------------------- native methods -----------------------------------*/
    private static native int asInt(long addr);

    @Override
    protected native void delete(long nativeObj);
}