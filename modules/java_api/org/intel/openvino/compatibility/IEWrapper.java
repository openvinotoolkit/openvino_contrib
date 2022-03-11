// Copyright (C) 2020-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

package org.intel.openvino.compatibility;

public class IEWrapper {
    protected final long nativeObj;

    protected IEWrapper(long addr) {
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
