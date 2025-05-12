package org.intel.openvino;

import static org.junit.Assert.*;

import org.junit.Before;
import org.junit.Test;

import java.util.ArrayList;
import java.util.Arrays;

public class ModelTests extends OVTest {
    Core core;
    Model net;

    @Before
    public void setUp() {
        core = new Core();
        net = core.read_model(modelXml);
    }

    @Test
    public void testOutputsName() {
        ArrayList<Output> outputs = net.outputs();
        assertEquals(1, outputs.size());
        assertEquals("Output name", "fc_out", outputs.get(0).get_any_name());
    }

    @Test
    public void testOutputName() {
        Output output = net.output();
        assertEquals("Output name", "fc_out", output.get_any_name());
    }

    @Test
    public void testOutputType() {
        Output output = net.output();
        assertEquals("Output element type", ElementType.f16, output.get_element_type());
    }

    @Test
    public void testGetShape() {
        ArrayList<Output> outputs = net.outputs();
        int[] ref = new int[] {1, 10};
        assertArrayEquals("Shape", ref, outputs.get(0).get_shape());
    }

    @Test
    public void testGetPartialShape() {
        ArrayList<Output> outputs = net.outputs();
        int[] ref = new int[] {1, 10};

        PartialShape partialShape = outputs.get(0).get_partial_shape();
        for (int i = 0; i < ref.length; i++) {
            Dimension dim = partialShape.get_dimension(i);
            assertEquals(ref[i], dim.get_length());
        }
        assertArrayEquals("MaxShape", ref, partialShape.get_max_shape());
        assertArrayEquals("MinShape", ref, partialShape.get_min_shape());
    }

    @Test
    public void testReshape() {
        int[] inpDims = net.input().get_shape();
        assertEquals(32, inpDims[2]);
        assertEquals(32, inpDims[3]);

        inpDims[2] = 33;
        inpDims[3] = 33;

        net.reshape(inpDims); // Expect no exception

        int[] outDims = net.output().get_shape();
        assertEquals(10, outDims[1]);
    }

    @Test
    public void testStartWaitAsync() {
        CompiledModel compiledModel = core.compile_model(net, device);
        InferRequest req = compiledModel.create_infer_request();

        // Fill input and output data
        float[] inputData = new float[3 * 32 * 32];
        float[] outputData = new float[10];

        Arrays.fill(inputData, 1);
        Arrays.fill(outputData, 1);

        Tensor input = new Tensor(new int[] {1, 3, 32, 32}, inputData);
        Tensor output = new Tensor(new int[] {1, 10}, outputData);

        req.set_input_tensor(input);
        req.set_output_tensor(output);

        // Run inference
        req.start_async();
        req.wait_async();

        // Check that output data is not zero
        outputData = req.get_output_tensor().data();
        for (int i = 0; i < outputData.length; ++i) {
            assertNotEquals(outputData[i], 0.0f);
        }
    }
}
