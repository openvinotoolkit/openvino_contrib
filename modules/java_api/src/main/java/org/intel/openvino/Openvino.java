// Copyright (C) 2020-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

package org.intel.openvino;

/** OpenVINO Runtime utility methods. */
public class Openvino extends Wrapper {

    protected Openvino(long addr) {
        super(addr);
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
     * @param compressToFP16 Whether to compress floating point weights to FP16.
     */
    public static void save_model(Model model, final String outputModel, boolean compressToFP16) {
        SaveModel(model.nativeObj, outputModel, compressToFP16);
    }

    /**
     * Serialize given model into IR.
     *
     * <p>The generated .xml and .bin files will be saved into provided paths.This method serializes
     * model "as-is" that means no weights compression is applied. It is recommended to use {@link
     * Openvino#save_model} function in all cases when it is not related to debugging.
     *
     * @param model Model which will be converted to IR representation.
     * @param xmlPath Path where .xml file will be saved.
     * @param binPath Path where .bin file will be saved.
     */
    public static void serialize(Model model, final String xmlPath, final String binPath) {
        serialize(model.nativeObj, xmlPath, binPath);
    }

    /*----------------------------------- native methods -----------------------------------*/
    private static native void SaveModel(
            long addr, final String outputModel, boolean compressToFP16);

    private static native void serialize(
            long modelAddr, final String xmlPath, final String binPath);
}
