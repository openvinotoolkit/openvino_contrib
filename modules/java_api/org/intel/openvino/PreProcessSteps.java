package org.intel.openvino;

public class PreProcessSteps extends Wrapper {

    public PreProcessSteps(long addr) {
        super(addr);
    }

    public PreProcessSteps resize(ResizeAlgorithm alg) {
        Resize(nativeObj, alg.getValue());
        return this;
    }

    /*---------------------------------- native methods -----------------------------------*/
    private static native void Resize(long nativeObj, int alg);

    @Override
    protected native void delete(long nativeObj);
}
