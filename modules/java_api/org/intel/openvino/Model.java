// Copyright (C) 2020-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

package org.intel.openvino;

import java.util.ArrayList;

public class Model extends Wrapper {

    protected Model(long addr) {
        super(addr);
    }

    public String get_name() {
        return getName(nativeObj);
    }

    public Dimension get_batch() {
        return new Dimension(getBatch(nativeObj));
    }

    public ArrayList<Output> outputs() {
        return getOutputs(nativeObj);
    }

    /*----------------------------------- native methods -----------------------------------*/
    private static native String getName(long addr);

    private static native long getBatch(long addr);

    private static native ArrayList<Output> getOutputs(long addr);

    @Override
    protected native void delete(long nativeObj);
}
