package org.intel.openvino;

import static org.junit.Assert.assertArrayEquals;
import static org.junit.Assert.assertEquals;

import org.junit.Before;
import org.junit.Test;

import java.util.List;

public class CompiledModelTests extends OVTest {

    private CompiledModel model;

    @Before
    public void init() {
        Core core = new Core();
        model = core.compile_model(modelXml, device);
    }

    @Test
    public void testInputs() {
        List<Output> inputs = model.inputs();

        assertEquals("data", inputs.get(0).get_any_name());
        assertEquals(ElementType.f32, inputs.get(0).get_element_type());

        int[] shape = new int[] {1, 3, 32, 32};
        assertArrayEquals("Shape", shape, inputs.get(0).get_shape());
    }

    @Test
    public void testOutputs() {
        List<Output> outputs = model.outputs();

        assertEquals("fc_out", outputs.get(0).get_any_name());
        assertEquals(ElementType.f32, outputs.get(0).get_element_type());

        int[] shape = new int[] {1, 10};
        assertArrayEquals("Shape", shape, outputs.get(0).get_shape());
    }
}
