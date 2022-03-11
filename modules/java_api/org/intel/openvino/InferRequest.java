// Copyright (C) 2020-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

package org.intel.openvino;

public class InferRequest extends Wrapper {
    protected InferRequest(long addr) {
        super(addr);
    }

    public void infer() {
        Infer(nativeObj);
    }

    public void set_input_tensor(Tensor input) {
        SetInputTensor(nativeObj, input.nativeObj);
    }

    public Tensor get_output_tensor() {
        return new Tensor(GetOutputTensor(nativeObj));
    }

    /*----------------------------------- native methods -----------------------------------*/
    private static native void Infer(long addr);

    private static native void SetInputTensor(long addr, long tensorAddr);

    private static native long GetOutputTensor(long addr);

    @Override
    protected native void delete(long nativeObj);
}
