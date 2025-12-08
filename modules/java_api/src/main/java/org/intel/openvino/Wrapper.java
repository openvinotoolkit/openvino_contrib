// Copyright (C) 2020-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

package org.intel.openvino;

import java.util.concurrent.atomic.AtomicBoolean;

public class Wrapper implements AutoCloseable {
    static {
        try {
            Class.forName("org.intel.openvino.NativeLibrary");
        } catch (ClassNotFoundException e) {
            throw new RuntimeException("Failed to load OpenVINO native libraries");
        }
    }

    protected final long nativeObj;
    private final AtomicBoolean closed = new AtomicBoolean(false);

    protected Wrapper(long addr) {
        nativeObj = addr;
    }

    protected long getNativeObjAddr() {
        return nativeObj;
    }

    @Override
    public void close() {
        if (!closed.compareAndSet(false, true)) {
            delete(nativeObj);
        }
    }

    /*----------------------------------- native methods -----------------------------------*/
    protected native void delete(long nativeObj);
}
