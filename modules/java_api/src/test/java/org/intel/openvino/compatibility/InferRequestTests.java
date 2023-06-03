package org.intel.openvino.compatibility;

import static org.junit.Assert.*;

import org.intel.openvino.compatibility.*;
import org.junit.Before;
import org.junit.Ignore;
import org.junit.Test;

import java.util.ArrayList;
import java.util.Map;

public class InferRequestTests extends IETest {
    IECore core;
    CNNNetwork net;
    ExecutableNetwork executableNetwork;
    InferRequest inferRequest;
    boolean completionCallback;

    @Before
    public void setUp() {
        core = new IECore();
        net = core.ReadNetwork(modelXml);
        executableNetwork = core.LoadNetwork(net, device);
        inferRequest = executableNetwork.CreateInferRequest();
        completionCallback = false;
    }

    @Ignore
    @Test
    public void testGetPerformanceCounts() {
        inferRequest.Infer();

        Map<String, InferenceEngineProfileInfo> res = inferRequest.GetPerformanceCounts();

        assertFalse(res.isEmpty());
        assertEquals("Map size", 22, res.size());
        ArrayList<String> resKeySet = new ArrayList<String>(res.keySet());

        for (int i = 0; i < res.size(); i++) {
            String key = resKeySet.get(i);
            InferenceEngineProfileInfo resVal = res.get(key);

            assertTrue(
                    resVal.status == InferenceEngineProfileInfo.LayerStatus.EXECUTED
                            || resVal.status == InferenceEngineProfileInfo.LayerStatus.NOT_RUN);
        }
    }

    @Test
    public void testStartAsync() {
        inferRequest.StartAsync();
        StatusCode statusCode = inferRequest.Wait(WaitMode.RESULT_READY);

        assertEquals("StartAsync", StatusCode.OK, statusCode);
    }

    @Test
    public void testSetCompletionCallback() {
        inferRequest.SetCompletionCallback(
                new Runnable() {

                    @Override
                    public void run() {
                        completionCallback = true;
                    }
                });

        for (int i = 0; i < 5; i++) {
            inferRequest.Wait(WaitMode.RESULT_READY);
            inferRequest.StartAsync();
        }

        inferRequest.Wait(WaitMode.RESULT_READY);
        inferRequest.StartAsync();
        StatusCode statusCode = inferRequest.Wait(WaitMode.RESULT_READY);

        assertEquals("SetCompletionCallback", true, completionCallback);
    }
}
