// Copyright (C) 2020-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include <jni.h> // JNI header provided by JDK
#include "openvino/openvino.hpp"

#include "openvino_java.hpp"
#include "jni_common.hpp"

using namespace ov;

JNIEXPORT jlong JNICALL Java_org_intel_openvino_PartialShape_GetDimension(JNIEnv *env, jobject obj, jlong addr, jint index) {
    JNI_METHOD("GetDimension",
        PartialShape* partial_shape = (PartialShape *)addr;
        return (jlong) &(*partial_shape)[index];
    )
    return 0;
}

JNIEXPORT jintArray JNICALL Java_org_intel_openvino_PartialShape_GetMaxShape(JNIEnv *env, jobject obj, jlong addr) {
    JNI_METHOD("GetMaxShape",
        PartialShape* partial_shape = (PartialShape *)addr;
        Shape max_shape = partial_shape->get_max_shape();

        jintArray result = env->NewIntArray(max_shape.size());
        if (!result) {
            throw std::runtime_error("Out of memory!");
        } jint *arr = env->GetIntArrayElements(result, nullptr);

        for (int i = 0; i < max_shape.size(); ++i)
            arr[i] = max_shape[i];

        env->ReleaseIntArrayElements(result, arr, 0);
        return result;
    )
    return 0;
}

JNIEXPORT jintArray JNICALL Java_org_intel_openvino_PartialShape_GetMinShape(JNIEnv *env, jobject obj, jlong addr) {
    JNI_METHOD("GetMinShape",
        PartialShape* partial_shape = (PartialShape *)addr;
        Shape min_shape = partial_shape->get_min_shape();

        jintArray result = env->NewIntArray(min_shape.size());
        if (!result) {
            throw std::runtime_error("Out of memory!");
        } jint *arr = env->GetIntArrayElements(result, nullptr);

        for (int i = 0; i < min_shape.size(); ++i)
            arr[i] = min_shape[i];

        env->ReleaseIntArrayElements(result, arr, 0);
        return result;
    )
    return 0;
}
