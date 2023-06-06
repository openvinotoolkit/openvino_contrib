// Copyright (C) 2020-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

package org.intel.openvino;

/**
 * Class holding preprocessing information for one input
 *
 * <p>From preprocessing pipeline perspective, each input can be represented as:
 *
 * <ul>
 *   <li>User's input parameter info ({@link InputInfo#tensor})
 *   <li>Preprocessing steps applied to user's input ({@link InputInfo#preprocess})
 *   <li>Model's input info, which is a final input's info after preprocessing ({@link
 *       InputInfo#model})
 *       <ul>
 */
public class InputInfo extends Wrapper {

    static {
        try {
            Class.forName("org.intel.openvino.NativeLibrary");
        } catch (ClassNotFoundException e) {
            throw new RuntimeException("Failed to load OpenVINO native libraries");
        }
    }

    public InputInfo(long addr) {
        super(addr);
    }

    /**
     * Get current input preprocess information with ability to add more preprocessing steps
     *
     * @return Reference to current preprocess steps structure
     */
    public PreProcessSteps preprocess() {
        return new PreProcessSteps(preprocess(nativeObj));
    }

    /**
     * Get current input tensor information with ability to change specific data
     *
     * @return Reference to current input tensor structure
     */
    public InputTensorInfo tensor() {
        return new InputTensorInfo(tensor(nativeObj));
    }

    /**
     * Get current input model information with ability to change original model's input data
     *
     * @return Reference to current model's input information structure
     */
    public InputModelInfo model() {
        return new InputModelInfo(model(nativeObj));
    }

    /*----------------------------------- native methods -----------------------------------*/
    private static native long preprocess(long addr);

    private static native long model(long addr);

    private static native long tensor(long addr);

    @Override
    protected native void delete(long nativeObj);
}
