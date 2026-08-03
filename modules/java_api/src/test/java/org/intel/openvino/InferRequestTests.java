// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

package org.intel.openvino;

import static org.junit.Assert.*;

import org.junit.Before;
import org.junit.Test;

import java.util.Arrays;

public class InferRequestTests extends OVTest {
    Core core;
    Model net;
    InferRequest inferRequest;

    @Before
    public void setUp() {
        core = new Core();
        net = core.read_model(modelXml);
        CompiledModel compiledModel = core.compile_model(net, device);
        inferRequest = compiledModel.create_infer_request();

        float[] inputData = new float[3 * 32 * 32];
        Arrays.fill(inputData, 1);
        Tensor input = new Tensor(new int[] {1, 3, 32, 32}, inputData);
        inferRequest.set_input_tensor(input);
        inferRequest.infer();
    }

    @Test
    public void testGetOutputTensorByIndex() {
        // The test model has a single output, so index 0 must return the same data as the
        // no-argument getter. This exercises the indexed accessor added for multi-output YOLO
        // models (segmentation, pose).
        Tensor byDefault = inferRequest.get_output_tensor();
        Tensor byIndex = inferRequest.get_output_tensor(0);

        assertArrayEquals(byDefault.get_shape(), byIndex.get_shape());
        assertArrayEquals(byDefault.data(), byIndex.data(), 0.0f);
    }

    @Test
    public void testGetOutputTensorByBadIndex() {
        try {
            inferRequest.get_output_tensor(5);
            fail("Expected an exception for an out-of-range output index");
        } catch (Exception e) {
            // OpenVINO throws when the output index is out of range; the exact message is
            // runtime-defined, so we only assert that an exception was raised.
            assertNotNull(e.getMessage());
        }
    }
}
