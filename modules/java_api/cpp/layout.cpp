#include <jni.h> // JNI header provided by JDK
#include "openvino/openvino.hpp"

#include "openvino_java.hpp"
#include "jni_common.hpp"

using namespace ov;

JNIEXPORT jlong JNICALL Java_org_intel_openvino_Layout_GetLayout(JNIEnv *env, jobject, jstring str)
{
    JNI_METHOD("GetLayout",
        std::string c_str = jstringToString(env, str);
        const Layout *layout = new Layout(c_str);

        return (jlong)layout;
    )
    return 0;
}

JNIEXPORT jint JNICALL Java_org_intel_openvino_Layout_HeightIdx(JNIEnv *env, jobject, jlong addr)
{
    JNI_METHOD("HeightIdx",
        const Layout *layout = (Layout *)(addr);
        auto height = layout::height_idx(*layout);

        return (jint)(height);
    )
    return 0;
}

JNIEXPORT jint JNICALL Java_org_intel_openvino_Layout_WidthIdx(JNIEnv *env, jobject, jlong addr)
{
    JNI_METHOD("WidthIdx",
        const Layout *layout = (Layout *)(addr);
        auto width = layout::width_idx(*layout);

        return (jint)(width);
    )
    return 0;
}

JNIEXPORT void JNICALL Java_org_intel_openvino_Layout_delete(JNIEnv *, jobject, jlong addr)
{
    Layout *layout = (Layout *)(addr);
    delete layout;
}
