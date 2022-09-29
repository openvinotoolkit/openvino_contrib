package tests;

import static org.junit.Assert.*;

import org.intel.openvino.*;
import org.junit.Test;

public class CoreTests extends OVTest {
    Core core = new Core();

    @Test
    public void testReadModel() {
        Model net = core.read_model(modelXml, modelBin);
        assertTrue(!net.get_name().isEmpty());
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
}
