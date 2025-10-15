package org.intel.openvino;

import static org.junit.Assert.*;

import org.junit.Before;
import org.junit.Ignore;
import org.junit.Test;

public class PrePostProcessorTests extends OVTest {
    Core core;
    Model net;
    Tensor input;
    int[] dimsArr = {1, 3, 2, 2};

    @Before
    public void setUp() {
        core = new Core();
        net = core.read_model(modelXml);
        float[] data = {0.0f, 1.1f, 2.2f, 3.3f, 4.4f, 5.5f, 6.6f, 7.7f, 8.8f, 9.9f, 1.1f, 2.2f};
        input = new Tensor(dimsArr, data);
    }

    @Ignore // unstable test
    @Test
    public void testWrongLayout() {
        String exceptionMessage = "";
        Layout tensor_layout = new Layout("NCHW");
        PrePostProcessor p = new PrePostProcessor(net);

        p.input()
                .tensor()
                .set_element_type(ElementType.f32)
                .set_layout(tensor_layout)
                .set_spatial_static_shape(dimsArr[2], dimsArr[3]);

        p.input().preprocess().resize(ResizeAlgorithm.RESIZE_LINEAR);
        p.input().model().set_layout(new Layout("NHWC"));

        p.build();

        CompiledModel compiledModel = core.compile_model(net, "CPU");
        InferRequest inferRequest = compiledModel.create_infer_request();

        try {
            inferRequest.set_input_tensor(input);
        } catch (Exception e) {
            exceptionMessage = e.getMessage();
        }
        assertTrue(
                exceptionMessage.contains(
                        "SetInputTensor: Can't set input blob with name: data, because model"
                                + " input"));
    }

    @Test
    public void testWrongElementType() {
        String exceptionMessage = "";
        Layout tensor_layout = new Layout("NCHW");
        PrePostProcessor p = new PrePostProcessor(net);

        p.input()
                .tensor()
                .set_element_type(ElementType.u8)
                .set_layout(tensor_layout)
                .set_spatial_static_shape(dimsArr[2], dimsArr[3]);

        p.input().preprocess().resize(ResizeAlgorithm.RESIZE_LINEAR);
        p.build();

        CompiledModel compiledModel = core.compile_model(net, "CPU");
        InferRequest inferRequest = compiledModel.create_infer_request();

        try {
            inferRequest.set_input_tensor(input);
        } catch (Exception e) {
            exceptionMessage = e.getMessage();
        }
        assertTrue(
                exceptionMessage.contains(
                        "ParameterMismatch: Failed to set tensor for input with precision: f32,"
                                + " since the model input tensor precision is: u8"));
    }
}
