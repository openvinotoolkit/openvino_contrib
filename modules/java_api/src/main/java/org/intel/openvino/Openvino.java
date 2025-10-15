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

    /**
     * Save model into IR files (xml and bin).
     *
     * <p>This method saves a model to IR applying all necessary transformations that are usually
     * applied in model conversion flow provided by mo tool. Particularly, floating point weights
     * are compressed to FP16, debug information in model nodes are cleaned up, etc.
     *
     * @param model Model which will be converted to IR representation.
     * @param outputModel Path to output model file.
     * @param compressToFp16 Whether to compress floating point weights to FP16.
     */
    public static void save_model(
            Model model, final String outputModel, final boolean compressToFp16) {
        SaveModel(model.nativeObj, outputModel, compressToFp16);
    }

    /*----------------------------------- native methods -----------------------------------*/

    private static native void serialize(
            long modelAddr, final String xmlPath, final String binPath);

    private static native void SaveModel(
            long modelAddr, final String outputModel, final boolean compressToFp16);
}
