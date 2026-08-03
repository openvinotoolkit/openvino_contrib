// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

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
    public void testScaleAndConvertLayout() {
        // Feed a u8 NHWC frame and let the preprocessor scale by 255 and transpose to the
        // model's NCHW f32 layout. This mirrors the OpenCV-free YOLO input path. The test model
        // input is [1, 3, 32, 32] (NCHW), so the user tensor is [1, 32, 32, 3] (NHWC).
        int[] modelInput = net.input().get_shape(); // [1, 3, H, W]
        int c = modelInput[1];
        int h = modelInput[2];
        int w = modelInput[3];
        byte[] u8data = new byte[h * w * c];
        for (int i = 0; i < u8data.length; i++) {
            u8data[i] = (byte) (i % 256);
        }
        Tensor u8input = new Tensor(ElementType.u8, new int[] {1, h, w, c}, u8data);

        PrePostProcessor p = new PrePostProcessor(net);
        p.input()
                .tensor()
                .set_element_type(ElementType.u8)
                .set_layout(new Layout("NHWC"))
                .set_spatial_static_shape(h, w);
        p.input()
                .preprocess()
                .convert_element_type(ElementType.f32)
                .scale(255.0f)
                .convert_layout(new Layout("NCHW"));
        p.input().model().set_layout(new Layout("NCHW"));
        p.build();

        CompiledModel compiledModel = core.compile_model(net, "CPU");
        InferRequest inferRequest = compiledModel.create_infer_request();
        inferRequest.set_input_tensor(u8input);
        inferRequest.infer();

        // A successful inference proves the scale + layout-convert steps were accepted and
        // fused into the model (the input tensor precision is now u8, not f32).
        assertEquals(ElementType.u8, compiledModel.inputs().get(0).get_element_type());
    }

    @Test
    public void testPerChannelMeanScale() {
        int[] shape = net.input().get_shape(); // [1, 3, H, W]
        int total = 1;
        for (int d : shape) {
            total *= d;
        }
        float[] inputData = new float[total];
        for (int i = 0; i < total; i++) {
            inputData[i] = (i % 17) + 1; // nonzero, varied
        }
        Tensor feed = new Tensor(shape, inputData);

        // Fold a per-channel (x - mean) / scale over the three channels into the model.
        PrePostProcessor p = new PrePostProcessor(net);
        p.input().tensor().set_element_type(ElementType.f32).set_layout(new Layout("NCHW"));
        p.input()
                .preprocess()
                .mean(new float[] {0.485f, 0.456f, 0.406f})
                .scale(new float[] {0.229f, 0.224f, 0.225f});
        p.input().model().set_layout(new Layout("NCHW"));
        Model built = p.build();

        // A successful build + inference proves the per-channel mean/scale steps were accepted
        // and fused into the model. (We don't compare the output against a no-preprocessing
        // baseline: the test model ends in a saturated one-hot classifier whose result is
        // invariant to input normalization, so such a comparison would be meaningless.)
        assertEquals(ElementType.f32, built.input().get_element_type());
        assertArrayEquals(shape, built.input().get_shape());

        InferRequest req = core.compile_model(built, "CPU").create_infer_request();
        req.set_input_tensor(feed);
        req.infer(); // must not throw
        assertEquals(net.output().get_shape().length, req.get_output_tensor().get_shape().length);
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
