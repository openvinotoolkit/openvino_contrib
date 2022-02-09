package org.intel.openvino.compatibility;

public class PreProcessInfo extends IEWrapper {

    public PreProcessInfo(long addr) {
        super(addr);
    }

    public void setResizeAlgorithm(ResizeAlgorithm resizeAlgorithm) {
        SetResizeAlgorithm(nativeObj, resizeAlgorithm.getValue());
    }

    /*----------------------------------- native methods -----------------------------------*/
    private static native void SetResizeAlgorithm(long addr, int resizeAlgorithm);

    @Override
    protected native void delete(long nativeObj);
}
