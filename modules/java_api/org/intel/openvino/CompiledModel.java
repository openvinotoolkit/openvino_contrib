// Copyright (C) 2020-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

package org.intel.openvino;

public class CompiledModel extends Wrapper {

    protected CompiledModel(long addr) {
        super(addr);
    }

    public InferRequest create_infer_request() {
        return new InferRequest(CreateInferRequest(nativeObj));
    }

    /*----------------------------------- native methods -----------------------------------*/
    private static native long CreateInferRequest(long addr);

    @Override
    protected native void delete(long nativeObj);
}
