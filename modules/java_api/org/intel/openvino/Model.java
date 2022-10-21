// Copyright (C) 2020-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

package org.intel.openvino;

import java.util.ArrayList;

/** A user-defined model */
public class Model extends Wrapper {

    protected Model(long addr) {
        super(addr);
    }

    /**
     * Get the unique name of the model.
     *
     * @return A const reference to the model's unique name.
     */
    public String get_name() {
        return getName(nativeObj);
    }

    /**
     * Helper method to get associated batch size for a Model
     *
     * <p>Checks layout of each parameter in a Model and extracts value for N (B) dimension. All
     * values are then merged and returned
     *
     * <p>Throws ::ov::AssertFailure with details in case of error.
     *
     * @return Dimension representing current batch size. Can represent a number or be a dynamic
     */
    public Dimension get_batch() {
        return new Dimension(getBatch(nativeObj));
    }

    /**
     * Get model outputs.
     *
     * @return A list of model outputs.
     */
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
