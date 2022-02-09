#include <jni.h> // JNI header provided by JDK
#include "openvino/openvino.hpp"

#include "openvino_java.hpp"
#include "jni_common.hpp"

using namespace ov;

JNIEXPORT void JNICALL Java_org_intel_openvino_InferRequest_Infer(JNIEnv *env, jobject obj, jlong addr)
{
    JNI_METHOD("Infer",
        InferRequest *infer_request = (InferRequest *)addr;
        infer_request->infer();)
}

JNIEXPORT void JNICALL Java_org_intel_openvino_InferRequest_SetInputTensor(JNIEnv *env, jobject, jlong addr, jlong tensorAddr)
{
    JNI_METHOD("SetInputTensor",
        InferRequest *infer_request = (InferRequest *)addr;
        Tensor *input_tensor = (Tensor *)tensorAddr;
        infer_request->set_input_tensor(*input_tensor);)
}

JNIEXPORT jlong JNICALL Java_org_intel_openvino_InferRequest_GetOutputTensor(JNIEnv *env, jobject obj, jlong addr)
{
    JNI_METHOD("GetOutputTensor",
        InferRequest *infer_request = (InferRequest *)addr;
        Tensor *output_tensor = new Tensor();

        *output_tensor = infer_request->get_output_tensor();

        return (jlong)output_tensor;
    )
    return 0;
}

JNIEXPORT void JNICALL Java_org_intel_openvino_InferRequest_delete(JNIEnv *, jobject, jlong addr)
{
    InferRequest *req = (InferRequest *)addr;
    delete req;
}
