package org.intel.openvino;

public class Data extends IEWrapper {

    protected Data(long addr) {
        super(addr);
    }

    /*----------------------------------- native methods -----------------------------------*/
    @Override
    protected native void delete(long nativeObj);
}
