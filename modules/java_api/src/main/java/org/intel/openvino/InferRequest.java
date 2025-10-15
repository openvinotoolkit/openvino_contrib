// Copyright (C) 2020-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

package org.intel.openvino;

/** This is a class of infer request that can be run in asynchronous or synchronous manners. */
public class InferRequest extends Wrapper {

    private boolean isReleased = false;

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
     * Starts inference of specified input(s) in asynchronous mode.
     *
     * <p>It returns immediately. Inference starts also immediately. Calling any method while the
     * request in a running state leads to throwning the ov::Busy exception.
     */
    public void start_async() {
        StartAsync(nativeObj);
    }

    /** Waits for the result to become available. Blocks until the result becomes available. */
    public void wait_async() {
        Wait(nativeObj);
    }

    /**
     * Sets an output tensor to infer models with single output.
     *
     * <p>If model has several outputs, an exception is thrown.
     *
     * @param tensor Reference to the output tensor.
     */
    public void set_output_tensor(Tensor tensor) {
        SetOutputTensor(nativeObj, tensor.nativeObj);
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

    /**
     * Sets an input/output tensor to infer on.
     *
     * @param tensorName Name of the tensor.
     * @param tensor The tensor to set.
     */
    public void set_tensor(String tensorName, Tensor tensor) {
        SetTensor(nativeObj, tensorName, tensor.nativeObj);
    }

    /**
     * Delete the native object to release resources.
     *
     * <p>This method is protected from double deallocation
     */
    public void release() {
        delete(nativeObj);
        isReleased = true;
    }

    /*----------------------------------- native methods -----------------------------------*/
    private static native void Infer(long addr);

    private static native void StartAsync(long addr);

    private static native void Wait(long addr);

    private static native void SetInputTensor(long addr, long tensorAddr);

    private static native void SetOutputTensor(long addr, long tensorAddr);

    private static native long GetOutputTensor(long addr);

    private static native long GetTensor(long addr, String tensorName);

    private static native void SetTensor(long addr, String tensorName, long tensor);

    @Override
    protected native void delete(long nativeObj);
}
