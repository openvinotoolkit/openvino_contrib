// Copyright (C) 2020-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

package org.intel.openvino;

public class Wrapper {

    static {
        try {
            Class.forName("org.intel.openvino.NativeLibrary");
        } catch (ClassNotFoundException e) {
            throw new RuntimeException("Failed to load OpenVINO native libraries");
        }
    }

    protected final long nativeObj;

    protected Wrapper(long addr) {
        nativeObj = addr;
    }

    protected long getNativeObjAddr() {
        return nativeObj;
    }

    @Override
    protected void finalize() throws Throwable {
        delete(nativeObj);
        super.finalize();
    }

    /*----------------------------------- native methods -----------------------------------*/
    protected native void delete(long nativeObj);
}
