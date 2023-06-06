// Copyright (C) 2020-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

package org.intel.openvino;

/**
 * Class holding postprocessing information for one output
 *
 * <p>From postprocessing pipeline perspective, each output can be represented as:
 *
 * <ul>
 *   <li>Model's output info, (OutputInfo::model)
 *   <li>Postprocessing steps applied to user's input (OutputInfo::postprocess)
 *   <li>User's desired output parameter information, which is a final one after preprocessing
 *       ({@link OutputInfo#tensor})
 * </ul>
 */
public class OutputInfo extends Wrapper {

    static {
        try {
            Class.forName("org.intel.openvino.NativeLibrary");
        } catch (ClassNotFoundException e) {
            throw new RuntimeException("Failed to load OpenVINO native libraries");
        }
    }

    public OutputInfo(long addr) {
        super(addr);
    }

    /**
     * Get current output tensor information with ability to change specific data
     *
     * @return Reference to current output tensor structure
     */
    public InputTensorInfo tensor() {
        return new InputTensorInfo(getTensor(nativeObj));
    }

    /*----------------------------------- native methods -----------------------------------*/

    private static native long getTensor(long addr);

    @Override
    protected native void delete(long nativeObj);
}
