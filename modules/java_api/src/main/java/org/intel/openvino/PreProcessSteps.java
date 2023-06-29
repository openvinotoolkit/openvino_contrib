// Copyright (C) 2020-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

package org.intel.openvino;

/**
 * Preprocessing steps. Each step typically intends adding of some operation to input parameter User
 * application can specify sequence of preprocessing steps in a builder-like manner.
 */
public class PreProcessSteps extends Wrapper {

    public PreProcessSteps(long addr) {
        super(addr);
    }

    /**
     * Add resize operation to model's dimensions.
     *
     * @param alg Resize algorithm.
     * @return Reference to 'this' to allow chaining with other calls in a builder-like manner.
     */
    public PreProcessSteps resize(ResizeAlgorithm alg) {
        Resize(nativeObj, alg.getValue());
        return this;
    }

    /*---------------------------------- native methods -----------------------------------*/
    private static native void Resize(long nativeObj, int alg);

    @Override
    protected native void delete(long nativeObj);
}
