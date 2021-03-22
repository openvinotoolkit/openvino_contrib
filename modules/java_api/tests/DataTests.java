import static org.junit.Assert.*;

import org.intel.openvino.*;
import org.junit.Test;

import java.util.ArrayList;
import java.util.Map;

public class DataTests extends IETest {
    IECore core = new IECore();

    @Test
    public void testLayout() {
        CNNNetwork net = core.ReadNetwork(modelXml);
        Map<String, Data> outputsInfo = net.getOutputsInfo();
        String outName = new ArrayList<String>(outputsInfo.keySet()).get(0);
        Data outputInfo = outputsInfo.get(outName);
        assertEquals("Output layout", outputInfo.getLayout(), Layout.NC);
        assertNotEquals("Output layout", outputInfo.getLayout(), Layout.ANY);

        outputInfo.setLayout(Layout.ANY);
        assertEquals("Output layout", outputInfo.getLayout(), Layout.ANY);
        assertNotEquals("Output layout", outputInfo.getLayout(), Layout.NC);
    }

    @Test
    public void testGetDims() {
        CNNNetwork net = core.ReadNetwork(modelXml);
        Map<String, Data> outputsInfo = net.getOutputsInfo();
        String outName = new ArrayList<String>(outputsInfo.keySet()).get(0);
        Data outputInfo = outputsInfo.get(outName);

        int[] ref = new int[] {1, 10};
        assertArrayEquals("Data", outputInfo.getDims(), ref);
    }
}
