#include <jni.h> // JNI header provided by JDK
#include "openvino/openvino.hpp"

#include "openvino_java.hpp"
#include "jni_common.hpp"

using namespace ov;

JNIEXPORT void JNICALL Java_org_intel_openvino_PreProcessSteps_Resize(JNIEnv *env, jobject, jlong addr, jint algorithm)
{
    JNI_METHOD("Resize",
        preprocess::PreProcessSteps *pps = (preprocess::PreProcessSteps *)addr;
        auto resize_algorithm = preprocess::ResizeAlgorithm(algorithm);

        pps->resize(resize_algorithm);
    )
}

/*  We don't use delete operator for native object because we don't own this object:
    no new operator has been used to allocate memory for it */
JNIEXPORT void JNICALL Java_org_intel_openvino_PreProcessSteps_delete(JNIEnv *, jobject, jlong) {}
