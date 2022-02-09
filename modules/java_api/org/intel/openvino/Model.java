package org.intel.openvino;

public class Model extends Wrapper {

    protected Model(long addr) {
        super(addr);
    }

    public String get_name() {
        return getName(nativeObj);
    }

    public Dimension get_batch() {
        return new Dimension(getBatch(nativeObj));
    }

    /*----------------------------------- native methods -----------------------------------*/
    private static native String getName(long addr);

    private static native long getBatch(long addr);

    @Override
    protected native void delete(long nativeObj);
}
