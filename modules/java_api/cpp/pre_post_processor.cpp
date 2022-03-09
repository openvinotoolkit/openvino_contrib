#include <jni.h> // JNI header provided by JDK
#include "openvino/openvino.hpp"

#include "openvino_java.hpp"
#include "jni_common.hpp"

using namespace ov;

JNIEXPORT jlong JNICALL Java_org_intel_openvino_PrePostProcessor_GetPrePostProcessor(JNIEnv *env, jobject, jlong modelAddr)
{
    JNI_METHOD("GetPrePostProcessor",
        std::shared_ptr<Model> *model = reinterpret_cast<std::shared_ptr<Model> *>(modelAddr);
        preprocess::PrePostProcessor *processor = new preprocess::PrePostProcessor(*model);

        return (jlong)processor;
    )
    return 0;
}

JNIEXPORT jlong JNICALL Java_org_intel_openvino_PrePostProcessor_Input(JNIEnv *env, jobject, jlong addr)
{
    JNI_METHOD("Input",
        preprocess::PrePostProcessor *processor = (preprocess::PrePostProcessor *)addr;
        return (jlong)(&processor->input());
    )
    return 0;
}

JNIEXPORT jlong JNICALL Java_org_intel_openvino_PrePostProcessor_Output(JNIEnv *env, jobject, jlong addr)
{
    JNI_METHOD("Output",
        preprocess::PrePostProcessor *processor = (preprocess::PrePostProcessor *)addr;
        return (jlong)(&processor->output());
    )
    return 0;
}

JNIEXPORT void JNICALL Java_org_intel_openvino_PrePostProcessor_Build(JNIEnv *env, jobject, jlong addr)
{
    JNI_METHOD("Build",
        preprocess::PrePostProcessor *processor = (preprocess::PrePostProcessor *)addr;
        processor->build();
    )
}

JNIEXPORT void JNICALL Java_org_intel_openvino_PrePostProcessor_delete(JNIEnv *, jobject, jlong addr)
{
    preprocess::PrePostProcessor *processor = (preprocess::PrePostProcessor *)addr;
    delete processor;
}
