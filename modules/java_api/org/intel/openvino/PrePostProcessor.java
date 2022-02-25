package org.intel.openvino;

public class PrePostProcessor extends Wrapper {

    protected PrePostProcessor(long addr) {
        super(addr);
    }

    public PrePostProcessor(Model model) {
        super(GetPrePostProcessor(model.nativeObj));
    }

    public InputInfo input() {
        return new InputInfo(Input(nativeObj));
    }

    public OutputInfo output() {
        return new OutputInfo(Output(nativeObj));
    }

    public void build() {
        Build(nativeObj);
    }

    /*----------------------------------- native methods -----------------------------------*/

    private static native long GetPrePostProcessor(long model);

    private static native long Input(long preprocess);

    private static native long Output(long preprocess);

    private static native void Build(long preprocess);

    @Override
    protected native void delete(long nativeObj);
}
