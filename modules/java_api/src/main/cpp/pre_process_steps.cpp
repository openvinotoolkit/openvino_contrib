// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include <jni.h> // JNI header provided by JDK
#include "openvino/openvino.hpp"

#include "openvino_java.hpp"
#include "jni_common.hpp"

using namespace ov;

JNIEXPORT void JNICALL Java_org_intel_openvino_PreProcessSteps_Resize(JNIEnv *env, jobject, jlong addr, jint algorithm)
{
    JNI_METHOD("Resize",
        preprocess::PreProcessSteps *pps = (preprocess::PreProcessSteps *)addr;
        auto resize_algorithm = preprocess::ResizeAlgorithm(algorithm);

        pps->resize(resize_algorithm);
    )
}

JNIEXPORT void JNICALL Java_org_intel_openvino_PreProcessSteps_Scale(JNIEnv *env, jobject, jlong addr, jfloat value)
{
    JNI_METHOD("Scale",
        preprocess::PreProcessSteps *pps = (preprocess::PreProcessSteps *)addr;
        pps->scale(static_cast<float>(value));
    )
}

JNIEXPORT void JNICALL Java_org_intel_openvino_PreProcessSteps_ScaleValues(JNIEnv *env, jobject, jlong addr, jfloatArray values)
{
    JNI_METHOD("ScaleValues",
        preprocess::PreProcessSteps *pps = (preprocess::PreProcessSteps *)addr;

        const jsize length = env->GetArrayLength(values);
        jfloat *data = env->GetFloatArrayElements(values, nullptr);
        std::vector<float> scale_values(data, data + length);
        env->ReleaseFloatArrayElements(values, data, JNI_ABORT);

        pps->scale(scale_values);
    )
}

JNIEXPORT void JNICALL Java_org_intel_openvino_PreProcessSteps_Mean(JNIEnv *env, jobject, jlong addr, jfloat value)
{
    JNI_METHOD("Mean",
        preprocess::PreProcessSteps *pps = (preprocess::PreProcessSteps *)addr;
        pps->mean(static_cast<float>(value));
    )
}

JNIEXPORT void JNICALL Java_org_intel_openvino_PreProcessSteps_MeanValues(JNIEnv *env, jobject, jlong addr, jfloatArray values)
{
    JNI_METHOD("MeanValues",
        preprocess::PreProcessSteps *pps = (preprocess::PreProcessSteps *)addr;

        const jsize length = env->GetArrayLength(values);
        jfloat *data = env->GetFloatArrayElements(values, nullptr);
        std::vector<float> mean_values(data, data + length);
        env->ReleaseFloatArrayElements(values, data, JNI_ABORT);

        pps->mean(mean_values);
    )
}

JNIEXPORT void JNICALL Java_org_intel_openvino_PreProcessSteps_ConvertElementType(JNIEnv *env, jobject, jlong addr, jint type)
{
    JNI_METHOD("ConvertElementType",
        preprocess::PreProcessSteps *pps = (preprocess::PreProcessSteps *)addr;
        auto t_type = get_ov_type(type);

        pps->convert_element_type(t_type);
    )
}

JNIEXPORT void JNICALL Java_org_intel_openvino_PreProcessSteps_ConvertLayout(JNIEnv *env, jobject, jlong addr, jlong dstLayoutAddr)
{
    JNI_METHOD("ConvertLayout",
        preprocess::PreProcessSteps *pps = (preprocess::PreProcessSteps *)addr;
        const Layout *dst_layout = (Layout *)dstLayoutAddr;

        pps->convert_layout(*dst_layout);
    )
}

/*  We don't use delete operator for native object because we don't own this object:
    no new operator has been used to allocate memory for it */
JNIEXPORT void JNICALL Java_org_intel_openvino_PreProcessSteps_delete(JNIEnv *, jobject, jlong) {}
