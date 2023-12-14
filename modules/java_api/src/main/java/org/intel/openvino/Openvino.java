// Copyright (C) 2020-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

package org.intel.openvino;

/** OpenVINO Runtime utility methods. */
public class Openvino extends Wrapper {

    protected Openvino(long addr) {
        super(addr);
    }

    /**
     * Serialize given model into IR.
     *
     * <p>The generated .xml and .bin files will be saved into provided paths.This method serializes
     * model "as-is" that means no weights compression is applied.
     *
     * @param model Model which will be converted to IR representation.
     * @param xmlPath Path where .xml file will be saved.
     * @param binPath Path where .bin file will be saved.
     */
    public static void serialize(Model model, final String xmlPath, final String binPath) {
        serialize(model.nativeObj, xmlPath, binPath);
    }

    /*----------------------------------- native methods -----------------------------------*/

    private static native void serialize(
            long modelAddr, final String xmlPath, final String binPath);
}
