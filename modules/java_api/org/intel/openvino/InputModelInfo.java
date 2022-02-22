package org.intel.openvino;

public class InputModelInfo extends Wrapper {

    public InputModelInfo(long addr) {
        super(addr);
    }

    public InputModelInfo set_layout(Layout layout) {
        SetLayout(nativeObj, layout.nativeObj);
        return this;
    }

    /*----------------------------------- native methods -----------------------------------*/
    private static native void SetLayout(long addr, long layout);

    @Override
    protected native void delete(long nativeObj);
}
