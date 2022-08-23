// Copyright (C) 2020-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include <jni.h> // JNI header provided by JDK
#include "openvino/openvino.hpp"

#include "openvino_java.hpp"
#include "jni_common.hpp"

using namespace ov;

JNIEXPORT jlong JNICALL Java_org_intel_openvino_InputInfo_preprocess(JNIEnv *env, jobject obj, jlong addr)
{
    JNI_METHOD("preprocess",
        preprocess::InputInfo *info = (preprocess::InputInfo *)addr;
        return (jlong)(&info->preprocess());
    )
    return 0;
}

JNIEXPORT jlong JNICALL Java_org_intel_openvino_InputInfo_tensor(JNIEnv *env, jobject obj, jlong addr)
{
    JNI_METHOD("tensor",
        preprocess::InputInfo *info = (preprocess::InputInfo *)addr;
        return (jlong)(&info->tensor());
    )
    return 0;
}

JNIEXPORT jlong JNICALL Java_org_intel_openvino_InputInfo_model(JNIEnv *env, jobject obj, jlong addr)
{
    JNI_METHOD("model",
        preprocess::InputInfo *info = (preprocess::InputInfo *)addr;
        return (jlong)(&info->model());
    )
    return 0;
}

/*  We don't use delete operator for native object because we don't own this object:
    no new operator has been used to allocate memory for it */
JNIEXPORT void JNICALL Java_org_intel_openvino_InputInfo_delete(JNIEnv *, jobject, jlong) {}
