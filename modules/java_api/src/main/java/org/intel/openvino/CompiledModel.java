// Copyright (C) 2020-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

package org.intel.openvino;

import java.util.List;

/**
 * This class represents a compiled model.
 *
 * <p>A model is compiled by a specific device by applying multiple optimization transformations,
 * then mapping to compute kernels.
 */
public class CompiledModel extends Wrapper {

    protected CompiledModel(long addr) {
        super(addr);
    }

    /**
     * Creates an inference request object used to infer the compiled model. The created request has
     * allocated input and output tensors (which can be changed later).
     *
     * @return {@link InferRequest} object
     */
    public InferRequest create_infer_request() {
        return new InferRequest(CreateInferRequest(nativeObj));
    }

    /**
     * Gets all inputs of a compiled model. They contain information about input tensors such as
     * tensor shape, names, and element type.
     *
     * @return List of model inputs.
     */
    public List<Output> inputs() {
        return GetInputs(nativeObj);
    }

    /**
     * Gets all outputs of a compiled model. They contain information about output tensors such as
     * tensor shape, name, and element type.
     *
     * @return List of model outputs.
     */
    public List<Output> outputs() {
        return GetOutputs(nativeObj);
    }

    /*----------------------------------- native methods -----------------------------------*/
    private static native long CreateInferRequest(long addr);

    private static native List<Output> GetInputs(long addr);

    private static native List<Output> GetOutputs(long addr);

    @Override
    protected native void delete(long nativeObj);
}
