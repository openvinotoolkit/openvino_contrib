// Copyright (C) 2020-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include <jni.h> // JNI header provided by JDK
#include "openvino/openvino.hpp"

#include "openvino_java.hpp"
#include "jni_common.hpp"

using namespace ov;

JNIEXPORT jlong JNICALL Java_org_intel_openvino_OutputInfo_getTensor(JNIEnv *env, jobject obj, jlong addr)
{
    JNI_METHOD("getTensor",
        preprocess::OutputInfo *info = (preprocess::OutputInfo *)addr;
        return (jlong)(&info->tensor());
    )
    return 0;
}

/*  We don't use delete operator for native object because we don't own this object:
    no new operator has been used to allocate memory for it */
JNIEXPORT void JNICALL Java_org_intel_openvino_OutputInfo_delete(JNIEnv *, jobject, jlong) {}
