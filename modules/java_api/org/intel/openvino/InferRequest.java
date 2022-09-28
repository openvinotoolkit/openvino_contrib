// Copyright (C) 2020-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

package org.intel.openvino;

/** This is a class of infer request that can be run in asynchronous or synchronous manners. */
public class InferRequest extends Wrapper {
    protected InferRequest(long addr) {
        super(addr);
    }

    /**
     * Infers specified input(s) in synchronous mode.
     *
     * <p>It blocks all methods of {@link InferRequest} while request is ongoing (running or waiting
     * in a queue). Calling any method leads to throwning the ov::Busy exception.
     */
    public void infer() {
        Infer(nativeObj);
    }

    /**
     * Sets an input tensor to infer models with single input.
     *
     * <p>If model has several inputs, an exception is thrown.
     *
     * @param tensor Reference to the input tensor.
     */
    public void set_input_tensor(Tensor tensor) {
        SetInputTensor(nativeObj, tensor.nativeObj);
    }

    /**
     * Gets an output tensor for inference.
     *
     * @return Output tensor for the model. If model has several outputs, an exception is thrown.
     */
    public Tensor get_output_tensor() {
        return new Tensor(GetOutputTensor(nativeObj));
    }

    /**
     * Gets an input/output tensor for inference by tensor name.
     *
     * @param tensorName Name of a tensor to get.
     * @return The tensor with name "tensorName". If the tensor is not found, an exception is
     *     thrown.
     */
    public Tensor get_tensor(String tensorName) {
        return new Tensor(GetTensor(nativeObj, tensorName));
    }

    /*----------------------------------- native methods -----------------------------------*/
    private static native void Infer(long addr);

    private static native void SetInputTensor(long addr, long tensorAddr);

    private static native long GetOutputTensor(long addr);

    private static native long GetTensor(long addr, String tensorName);

    @Override
    protected native void delete(long nativeObj);
}
