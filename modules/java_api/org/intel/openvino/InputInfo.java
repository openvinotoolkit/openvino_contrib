package org.intel.openvino;

public class InputInfo extends Wrapper {

    public InputInfo(long addr) {
        super(addr);
    }

    public PreProcessSteps preprocess() {
        return new PreProcessSteps(preprocess(nativeObj));
    }

    public InputTensorInfo tensor() {
        return new InputTensorInfo(tensor(nativeObj));
    }

    public InputModelInfo model() {
        return new InputModelInfo(model(nativeObj));
    }

    /*----------------------------------- native methods -----------------------------------*/
    private static native long preprocess(long addr);

    private static native long model(long addr);

    private static native long tensor(long addr);

    @Override
    protected native void delete(long nativeObj);
}
