package org.intel.openvino;

import static org.junit.Assert.*;

import org.junit.Test;

public class WrapperCloseTests extends OVTest {

    @Test
    public void testTensorClose() {
        Tensor tensor = new Tensor(new int[] {1, 1}, new float[] {1.0f});
        assertIsClosedAfterClose(tensor);
    }

    @Test
    public void testTensorTryWithResources() {
        try (Tensor tensor = new Tensor(new int[] {1, 1}, new float[] {1.0f})) {
            assertFalse(tensor.isClosed());
        }
    }

    @Test
    public void testCoreClose() {
        Core core = new Core();
        assertIsClosedAfterClose(core);
    }

    @Test
    public void testCoreTryWithResources() {
        try (Core core = new Core()) {
            assertFalse(core.isClosed());
        }
    }

    @Test
    public void testModelClose() {
        Core core = new Core();

        Model model = core.read_model(modelXml);
        assertIsClosedAfterClose(model);

        core.close();
    }

    @Test
    public void testModelTryWithResources() {
        try (Core core = new Core();
                Model model = core.read_model(modelXml)) {
            assertFalse(model.isClosed());
        }
    }

    @Test
    public void testCompiledModelClose() {
        Core core = new Core();
        Model model = core.read_model(modelXml);

        CompiledModel compiledModel = core.compile_model(model, device);
        assertIsClosedAfterClose(compiledModel);

        model.close();
        core.close();
    }

    @Test
    public void testCompiledModelTryWithResources() {
        try (Core core = new Core();
                Model model = core.read_model(modelXml);
                CompiledModel compiledModel = core.compile_model(model, device)) {
            assertFalse(compiledModel.isClosed());
        }
    }

    @Test
    public void testInferRequestClose() {
        Core core = new Core();
        Model model = core.read_model(modelXml);
        CompiledModel compiledModel = core.compile_model(model, device);

        InferRequest inferRequest = compiledModel.create_infer_request();
        assertIsClosedAfterClose(inferRequest);

        compiledModel.close();
        model.close();
        core.close();
    }

    @Test
    public void testInferRequestTryWithResources() {
        try (Core core = new Core();
                Model model = core.read_model(modelXml);
                CompiledModel compiledModel = core.compile_model(model, device);
                InferRequest inferRequest = compiledModel.create_infer_request()) {
            assertFalse(inferRequest.isClosed());
        }
    }

    @Test
    public void testPrePostProcessorClose() {
        Core core = new Core();
        Model model = core.read_model(modelXml);

        PrePostProcessor ppp = new PrePostProcessor(model);
        assertIsClosedAfterClose(ppp);

        model.close();
        core.close();
    }

    @Test
    public void testPrePostProcessorTryWithResources() {
        try (Core core = new Core();
                Model model = core.read_model(modelXml);
                PrePostProcessor ppp = new PrePostProcessor(model)) {
            assertFalse(ppp.isClosed());
        }
    }

    @Test
    public void testLayoutClose() {
        Layout layout = new Layout("NCHW");
        assertIsClosedAfterClose(layout);
    }

    @Test
    public void testLayoutTryWithResources() {
        try (Layout layout = new Layout("NCHW")) {
            assertFalse(layout.isClosed());
        }
    }

    @Test
    public void testPartialShapeClose() {
        Core core = new Core();
        Model model = core.read_model(modelXml);
        Output output = model.output();

        PartialShape shape = output.get_partial_shape();
        assertIsClosedAfterClose(shape);

        output.close();
        model.close();
        core.close();
    }

    @Test
    public void testPartialShapeTryWithResources() {
        try (Core core = new Core();
                Model model = core.read_model(modelXml);
                Output output = model.output();
                PartialShape shape = output.get_partial_shape()) {
            assertFalse(shape.isClosed());
        }
    }

    @Test
    public void testDimensionClose() {
        Core core = new Core();
        Model model = core.read_model(modelXml);
        Output output = model.output();
        PartialShape shape = output.get_partial_shape();

        Dimension dim = shape.get_dimension(0);
        assertIsClosedAfterClose(dim);

        shape.close();
        output.close();
        model.close();
        core.close();
    }

    @Test
    public void testDimensionTryWithResources() {
        try (Core core = new Core();
                Model model = core.read_model(modelXml);
                Output output = model.output();
                PartialShape shape = output.get_partial_shape();
                Dimension dim = shape.get_dimension(0)) {
            assertFalse(dim.isClosed());
        }
    }

    @Test
    public void testOutputClose() {
        Core core = new Core();
        Model model = core.read_model(modelXml);

        Output output = model.output();
        assertIsClosedAfterClose(output);

        model.close();
        core.close();
    }

    @Test
    public void testOutputTryWithResources() {
        try (Core core = new Core();
                Model model = core.read_model(modelXml);
                Output output = model.output()) {
            assertFalse(output.isClosed());
        }
    }

    @Test
    public void testAnyClose() {
        Core core = new Core();

        Any any = core.get_property("CPU", "OPTIMAL_NUMBER_OF_INFER_REQUESTS");
        assertIsClosedAfterClose(any);

        core.close();
    }

    @Test
    public void testAnyTryWithResources() {
        try (Core core = new Core();
                Any any = core.get_property("CPU", "OPTIMAL_NUMBER_OF_INFER_REQUESTS")) {
            assertFalse(any.isClosed());
        }
    }

    @Test
    public void testInputInfoClose() {
        Core core = new Core();
        Model model = core.read_model(modelXml);
        PrePostProcessor ppp = new PrePostProcessor(model);

        InputInfo inputInfo = ppp.input();
        assertIsClosedAfterClose(inputInfo);

        ppp.close();
        model.close();
        core.close();
    }

    @Test
    public void testInputInfoTryWithResources() {
        try (Core core = new Core();
                Model model = core.read_model(modelXml);
                PrePostProcessor ppp = new PrePostProcessor(model);
                InputInfo inputInfo = ppp.input()) {
            assertFalse(inputInfo.isClosed());
        }
    }

    @Test
    public void testInputModelInfoClose() {
        Core core = new Core();
        Model model = core.read_model(modelXml);
        PrePostProcessor ppp = new PrePostProcessor(model);
        InputInfo inputInfo = ppp.input();

        InputModelInfo modelInfo = inputInfo.model();
        assertIsClosedAfterClose(modelInfo);

        inputInfo.close();
        ppp.close();
        model.close();
        core.close();
    }

    @Test
    public void testInputModelInfoTryWithResources() {
        try (Core core = new Core();
                Model model = core.read_model(modelXml);
                PrePostProcessor ppp = new PrePostProcessor(model);
                InputInfo inputInfo = ppp.input();
                InputModelInfo modelInfo = inputInfo.model()) {
            assertFalse(modelInfo.isClosed());
        }
    }

    @Test
    public void testInputTensorInfoClose() {
        Core core = new Core();
        Model model = core.read_model(modelXml);
        PrePostProcessor ppp = new PrePostProcessor(model);
        InputInfo inputInfo = ppp.input();

        InputTensorInfo tensorInfo = inputInfo.tensor();
        assertIsClosedAfterClose(tensorInfo);

        inputInfo.close();
        ppp.close();
        model.close();
        core.close();
    }

    @Test
    public void testInputTensorInfoTryWithResources() {
        try (Core core = new Core();
                Model model = core.read_model(modelXml);
                PrePostProcessor ppp = new PrePostProcessor(model);
                InputInfo inputInfo = ppp.input();
                InputTensorInfo tensorInfo = inputInfo.tensor()) {
            assertFalse(tensorInfo.isClosed());
        }
    }

    @Test
    public void testOutputInfoClose() {
        Core core = new Core();
        Model model = core.read_model(modelXml);
        PrePostProcessor ppp = new PrePostProcessor(model);

        OutputInfo outputInfo = ppp.output();
        assertIsClosedAfterClose(outputInfo);

        ppp.close();
        model.close();
        core.close();
    }

    @Test
    public void testOutputInfoTryWithResources() {
        try (Core core = new Core();
                Model model = core.read_model(modelXml);
                PrePostProcessor ppp = new PrePostProcessor(model);
                OutputInfo outputInfo = ppp.output()) {
            assertFalse(outputInfo.isClosed());
        }
    }

    @Test
    public void testPreProcessStepsClose() {
        Core core = new Core();
        Model model = core.read_model(modelXml);
        PrePostProcessor ppp = new PrePostProcessor(model);
        InputInfo inputInfo = ppp.input();

        PreProcessSteps steps = inputInfo.preprocess();
        assertIsClosedAfterClose(steps);

        inputInfo.close();
        ppp.close();
        model.close();
        core.close();
    }

    @Test
    public void testPreProcessStepsTryWithResources() {
        try (Core core = new Core();
                Model model = core.read_model(modelXml);
                PrePostProcessor ppp = new PrePostProcessor(model);
                InputInfo inputInfo = ppp.input();
                PreProcessSteps steps = inputInfo.preprocess()) {
            assertFalse(steps.isClosed());
        }
    }

    @Test
    public void testTryWithResourcesActuallyCloses() {
        Tensor tensor = new Tensor(new int[] {1, 1}, new float[] {1.0f});
        try (tensor) {
            assertFalse("Should not be closed inside try block", tensor.isClosed());
        }
        assertTrue("Should be closed after try block", tensor.isClosed());
    }

    protected void assertIsClosedAfterClose(Wrapper wrapper) {
        assertFalse("Wrapper should not be closed initially", wrapper.isClosed());

        wrapper.close();
        assertTrue("Wrapper should be closed after close()", wrapper.isClosed());

        wrapper.close();
        assertTrue("Wrapper should still be closed after second close()", wrapper.isClosed());

        wrapper.close();
        assertTrue("Wrapper Should still be closed after third close()", wrapper.isClosed());
    }
}
