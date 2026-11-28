// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

package org.intel.openvino;

import static org.junit.Assert.*;

import org.junit.Before;
import org.junit.Test;

import java.nio.file.Paths;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.Map;

public class ModelMultiInputTests extends OVTest {
    Core core;
    Model net;

    @Before
    public void setUp() throws Exception {
        core = new Core();
        String fixturePath =
                Paths.get(getClass().getResource("two_input_model.xml").toURI()).toString();
        net = core.read_model(fixturePath);
    }

    @Test
    public void testReshapeByInputName() {
        Map<String, int[]> partialShapes = new HashMap<>();
        partialShapes.put("input_a", new int[] {2, 4});
        net.reshape(partialShapes);

        ArrayList<Output> outputs = net.outputs();
        assertEquals(2, outputs.size());
        assertArrayEquals(new int[] {2, 4}, outputs.get(0).get_shape());
        assertArrayEquals(new int[] {1, 5}, outputs.get(1).get_shape());
    }
}
