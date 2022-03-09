package org.intel.openvino;

public class Layout extends Wrapper {

    public Layout(String str) {
        super(GetLayout(str));
    }

    public static int height_idx(Layout layout) {
        return HeightIdx(layout.nativeObj);
    }

    public static int width_idx(Layout layout) {
        return WidthIdx(layout.nativeObj);
    }

    /*----------------------------------- native methods -----------------------------------*/
    private static native long GetLayout(String str);

    private static native int HeightIdx(long layout);

    private static native int WidthIdx(long layout);

    @Override
    protected native void delete(long nativeObj);
}
