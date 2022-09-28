// Copyright (C) 2020-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

package org.intel.openvino;

/**
 * Main class for adding pre- and post- processing steps to existing {@link Model}
 *
 * <p>This is a helper class for writing easy pre- and post- processing operations on {@link Model}
 * object assuming that any preprocess operation takes one input and produces one output.
 *
 * <p>For advanced preprocessing scenarios, like combining several functions with multiple
 * inputs/outputs into one, client's code can use transformation passes over {@link Model}
 */
public class PrePostProcessor extends Wrapper {

    protected PrePostProcessor(long addr) {
        super(addr);
    }

    public PrePostProcessor(Model model) {
        super(GetPrePostProcessor(model.nativeObj));
    }

    /**
     * Gets input pre-processing data structure. Should be used only if model/function has only one
     * input Using returned structure application's code is able to set user's tensor data (e.g
     * layout), preprocess steps, target model's data
     *
     * @return Reference to model's input information structure
     */
    public InputInfo input() {
        return new InputInfo(Input(nativeObj));
    }

    /**
     * Gets output post-processing data structure. Should be used only if model/function has only
     * one output Using returned structure application's code is able to set model's output data,
     * post-process steps, user's tensor data (e.g layout)
     *
     * @return Reference to model's output information structure
     */
    public OutputInfo output() {
        return new OutputInfo(Output(nativeObj));
    }

    /**
     * Adds pre/post-processing operations to function passed in constructor
     *
     * @return Function with added pre/post-processing operations
     */
    public Model build() {
        return new Model(Build(nativeObj));
    }

    /*----------------------------------- native methods -----------------------------------*/

    private static native long GetPrePostProcessor(long model);

    private static native long Input(long preprocess);

    private static native long Output(long preprocess);

    private static native long Build(long preprocess);

    @Override
    protected native void delete(long nativeObj);
}
