// Copyright (C) 2020-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include <jni.h> // JNI header provided by JDK
#include "openvino/openvino.hpp"

#include "openvino_java.hpp"
#include "jni_common.hpp"

using namespace ov;

JNIEXPORT jint JNICALL Java_org_intel_openvino_Dimension_getLength(JNIEnv *env, jobject, jlong addr)
{
    JNI_METHOD("getLength",
        Dimension *dim = (Dimension *)addr;
        return (jint)dim->get_length();
    )
    return 0;
}
