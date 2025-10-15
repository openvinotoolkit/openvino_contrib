package org.intel.openvino;

import static org.junit.Assert.*;

import org.junit.Test;

import java.util.HashMap;
import java.util.List;
import java.util.Map;

public class CoreTests extends OVTest {
    Core core = new Core();

    @Test
    public void testReadModel() {
        Model net = core.read_model(modelXml, modelBin);
        assertTrue(!net.get_name().isEmpty());
    }

    @Test
    public void testCompileModelFromFileDeviceAuto() {
        CompiledModel model = core.compile_model(modelXml);
        assertTrue(model instanceof CompiledModel);
    }

    @Test
    public void testCompileModelFromFile() {
        CompiledModel model = core.compile_model(modelXml, device);
        assertTrue(model instanceof CompiledModel);
    }

    @Test
    public void testCompileModelWithProps() {
        Map<String, String> properties = new HashMap<>();
        properties.put("INFERENCE_NUM_THREADS", "1");
        CompiledModel model = core.compile_model(modelXml, device, properties);
        assertTrue(model instanceof CompiledModel);
    }

    @Test
    public void testReadNetworkXmlOnly() {
        Model net = core.read_model(modelXml);
        PrePostProcessor p = new PrePostProcessor(net);
        p.input().tensor().set_layout(new Layout("NCHW"));
        p.build();
        assertEquals("Batch size", 1, net.get_batch().get_length());
    }

    @Test
    public void testReadNetworkOnnx() {
        Model net = core.read_model(modelOnnx);
        PrePostProcessor p = new PrePostProcessor(net);
        p.input().tensor().set_layout(new Layout("NCHW"));
        p.build();
        assertEquals("Batch size", 1, net.get_batch().get_length());
    }

    @Test
    public void testReadModelIncorrectBinPath() {
        String exceptionMessage = "";
        try {
            Model net = core.read_model(modelXml, "model.bin");
        } catch (Exception e) {
            exceptionMessage = e.getMessage();
        }
        assertFalse(exceptionMessage.isEmpty());
    }

    @Test
    public void testLoadNetwork() {
        Model net = core.read_model(modelXml);
        CompiledModel compiledModel = core.compile_model(net, device);

        assertTrue(compiledModel instanceof CompiledModel);
    }

    @Test
    public void testProperty() {
        int nireq1 = core.get_property("CPU", "OPTIMAL_NUMBER_OF_INFER_REQUESTS").asInt();
        assertEquals("Initial number of requests", 1, nireq1);

        Map<String, String> config =
                new HashMap<String, String>() {
                    {
                        put("NUM_STREAMS", "4");
                    }
                };
        core.set_property("CPU", config);
        int nireq2 = core.get_property("CPU", "OPTIMAL_NUMBER_OF_INFER_REQUESTS").asInt();

        assertEquals("Final number of requests", 4, nireq2);

        config.put("NUM_STREAMS", "1");
        core.set_property("CPU", config); // Restore
    }

    @Test
    public void testAvailableDevices() {
        List<String> availableDevices = core.get_available_devices();
        assertNotNull(availableDevices);
    }
}
