// Copyright (C) 2020-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

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

JNIEXPORT void JNICALL Java_org_intel_openvino_InferRequest_StartAsync(JNIEnv *env, jobject obj, jlong addr)
{
    JNI_METHOD("StartAsync",
        InferRequest *infer_request = (InferRequest *)addr;
        infer_request->start_async();
    )
}

JNIEXPORT void JNICALL Java_org_intel_openvino_InferRequest_Wait(JNIEnv *env, jobject obj, jlong addr)
{
    JNI_METHOD("Wait",
        InferRequest *infer_request = (InferRequest *)addr;
        infer_request->wait();
    )
}

JNIEXPORT void JNICALL Java_org_intel_openvino_InferRequest_SetInputTensor(JNIEnv *env, jobject, jlong addr, jlong tensorAddr)
{
    JNI_METHOD("SetInputTensor",
        InferRequest *infer_request = (InferRequest *)addr;
        Tensor *input_tensor = (Tensor *)tensorAddr;
        infer_request->set_input_tensor(*input_tensor);)
}

JNIEXPORT void JNICALL Java_org_intel_openvino_InferRequest_SetOutputTensor(JNIEnv *env, jobject, jlong addr, jlong tensorAddr)
{
    JNI_METHOD("SetOutputTensor",
        InferRequest *infer_request = (InferRequest *)addr;
        Tensor *tensor = (Tensor *)tensorAddr;
        infer_request->set_output_tensor(*tensor);)
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

JNIEXPORT jlong JNICALL Java_org_intel_openvino_InferRequest_GetTensor(JNIEnv *env, jobject obj, jlong addr, jstring tensorName)
{
    JNI_METHOD("GetTensor",
        InferRequest *infer_request = (InferRequest *)addr;
        Tensor *output_tensor = new Tensor();

        std::string c_tensorName = jstringToString(env, tensorName);
        *output_tensor = infer_request->get_tensor(c_tensorName);

        return (jlong)output_tensor;
    )
    return 0;
}

JNIEXPORT void JNICALL Java_org_intel_openvino_InferRequest_SetTensor(JNIEnv *env, jobject obj, jlong addr, jstring tensorName, jlong tensorAddr)
{
    JNI_METHOD("SetTensor",
        InferRequest *infer_request = (InferRequest *)addr;

        std::string c_tensorName = jstringToString(env, tensorName);
        const Tensor *tensor = (Tensor *)tensorAddr;
        infer_request->set_tensor(c_tensorName, *tensor);
    )
}

JNIEXPORT void JNICALL Java_org_intel_openvino_InferRequest_delete(JNIEnv *env, jobject obj, jlong addr)
{
    jclass cls = env->GetObjectClass(obj);
    jfieldID field = env->GetFieldID(cls, "isReleased", "Z");
    jboolean isReleased = env->GetBooleanField(obj, field);
    if (!isReleased) {
      InferRequest *req = (InferRequest *)addr;
      delete req;
    }
}
