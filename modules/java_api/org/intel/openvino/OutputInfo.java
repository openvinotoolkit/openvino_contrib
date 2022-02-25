package org.intel.openvino;

public class OutputInfo extends Wrapper {

    public OutputInfo(long addr) {
        super(addr);
    }

    public InputTensorInfo tensor() {
        return new InputTensorInfo(getTensor(nativeObj));
    }

    /*----------------------------------- native methods -----------------------------------*/

    private static native long getTensor(long addr);

    @Override
    protected native void delete(long nativeObj);
}
