// Copyright (C) 2020-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include <jni.h> // JNI header provided by JDK
#include "openvino/openvino.hpp"

#include "openvino_java.hpp"
#include "jni_common.hpp"

using namespace ov;

JNIEXPORT jlong JNICALL Java_org_intel_openvino_CompiledModel_CreateInferRequest(JNIEnv *env, jobject obj, jlong addr)
{
    JNI_METHOD("CreateInferRequest",
        CompiledModel *compiled_model = (CompiledModel *)addr;

        InferRequest *infer_request = new InferRequest();
        *infer_request = compiled_model->create_infer_request();

        return (jlong)infer_request;
    )
    return 0;
}

JNIEXPORT void JNICALL Java_org_intel_openvino_CompiledModel_delete(JNIEnv *, jobject, jlong addr)
{
    CompiledModel *compiled_model = (CompiledModel *)addr;
    delete compiled_model;
}
