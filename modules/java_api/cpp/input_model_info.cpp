#include <jni.h> // JNI header provided by JDK
#include "openvino/openvino.hpp"

#include "openvino_java.hpp"
#include "jni_common.hpp"

using namespace ov;

JNIEXPORT void JNICALL Java_org_intel_openvino_InputModelInfo_SetLayout(JNIEnv *env, jobject obj, jlong addr, jlong l_addr)
{
    JNI_METHOD("SetLayout",
        preprocess::InputModelInfo *info = (preprocess::InputModelInfo *)addr;
        const Layout *layout = (Layout *)(l_addr);
        info->set_layout(*layout);
    )
}

/*  We don't use delete operator for native object because we don't own this object:
    no new operator has been used to allocate memory for it */
JNIEXPORT void JNICALL Java_org_intel_openvino_InputModelInfo_delete(JNIEnv *, jobject, jlong) {}
