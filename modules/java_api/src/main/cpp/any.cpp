// Copyright (C) 2020-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include <jni.h> // JNI header provided by JDK
#include "openvino/openvino.hpp"

#include "openvino_java.hpp"
#include "jni_common.hpp"

using namespace ov;

JNIEXPORT jint JNICALL Java_org_intel_openvino_Any_asInt(JNIEnv *env, jobject obj, jlong addr) {
    JNI_METHOD("asInt",
        Any *obj = (Any *)addr;
        return obj->as<uint32_t>();
    )
    return 0;
}

JNIEXPORT void JNICALL Java_org_intel_openvino_Any_delete(JNIEnv *, jobject, jlong addr)
{
    Any *obj = (Any *)addr;
    delete obj;
}
