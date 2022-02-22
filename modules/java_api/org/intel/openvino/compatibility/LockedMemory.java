package org.intel.openvino.compatibility;

public class LockedMemory extends IEWrapper {

    protected LockedMemory(long addr) {
        super(addr);
    }

    public void get(float[] res) {
        asFloat(nativeObj, res);
    }

    public void get(byte[] res) {
        asByte(nativeObj, res);
    }

    public void get(long[] res) {
        asLong(nativeObj, res);
    }

    public void get(int[] res) {
        asInt(nativeObj, res);
    }

    /*----------------------------------- native methods -----------------------------------*/
    private static native void asByte(long addr, byte[] res);

    private static native void asFloat(long addr, float[] res);

    private static native void asLong(long addr, long[] res);

    private static native void asInt(long addr, int[] res);

    @Override
    protected native void delete(long nativeObj);
}
