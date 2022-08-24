package tests;

import static org.junit.Assert.*;

import org.intel.openvino.*;
import org.junit.Before;
import org.junit.Test;

import java.util.ArrayList;

public class ModelTests extends OVTest {
    Core core;
    Model net;
    ArrayList<Output> outputs;

    @Before
    public void setUp() {
        core = new Core();
        net = core.read_model(modelXml);
        outputs = net.outputs();
    }

    @Test
    public void testOutputName() {
        assertEquals(1, outputs.size());
        assertEquals("Output name", "fc_out", outputs.get(0).get_any_name());
    }

    @Test
    public void testGetShape() {
        int[] ref = new int[] {1, 10};
        assertArrayEquals("Shape", ref, outputs.get(0).get_shape());
    }
}
