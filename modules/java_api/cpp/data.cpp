#include <inference_engine.hpp>

#include "openvino_java.hpp"
#include "jni_common.hpp"

/*  We don't use delete operator for native object because we don't own this object:
    no new operator has been used to allocate memory for it */
JNIEXPORT void JNICALL Java_org_intel_openvino_Data_delete(JNIEnv *env, jobject obj, jlong addr) {}
