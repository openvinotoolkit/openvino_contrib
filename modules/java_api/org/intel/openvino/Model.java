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

    public Output output() {
        return new Output(getOutput(nativeObj));
    }

    public void reshape(int[] shape) {
        Reshape(nativeObj, shape);
    }

    public Output input() {
        return new Output(getInput(nativeObj));
    }

    /*----------------------------------- native methods -----------------------------------*/
    private static native String getName(long addr);

    private static native long getBatch(long addr);

    private static native ArrayList<Output> getOutputs(long addr);

    private static native long getOutput(long addr);

    private static native void Reshape(long addr, int[] shape);

    private static native long getInput(long addr);

    @Override
    protected native void delete(long nativeObj);
}
