// Copyright (C) 2020-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include <jni.h> // JNI header provided by JDK
#include "openvino/openvino.hpp"

#include "openvino_java.hpp"
#include "jni_common.hpp"

using namespace ov;

JNIEXPORT void JNICALL Java_org_intel_openvino_InputTensorInfo_SetElementType(JNIEnv *env, jobject obj, jlong addr, jint type)
{
    JNI_METHOD("SetElementType",
        preprocess::InputTensorInfo *info = (preprocess::InputTensorInfo *)addr;
        auto t_type = get_ov_type(type);

        info->set_element_type(t_type);
    )
}

JNIEXPORT void JNICALL Java_org_intel_openvino_InputTensorInfo_SetLayout(JNIEnv *env, jobject obj, jlong addr, jlong l_addr)
{
    JNI_METHOD("SetLayout",
        preprocess::InputTensorInfo *info = (preprocess::InputTensorInfo *)addr;
        const Layout *layout = (Layout *)(l_addr);

        info->set_layout(*layout);
    )
}

JNIEXPORT void JNICALL Java_org_intel_openvino_InputTensorInfo_SetSpatialStaticShape(JNIEnv *env, jobject obj, jlong addr, jint height, jint width)
{
    JNI_METHOD("SetSpatialStaticShape",
        preprocess::InputTensorInfo *info = (preprocess::InputTensorInfo *)addr;
        size_t c_height = static_cast<size_t>(height);
        size_t c_width = static_cast<size_t>(width);

        info->set_spatial_static_shape(c_height, c_width);
    )
}

JNIEXPORT void JNICALL Java_org_intel_openvino_InputTensorInfo_SetSpatialDynamicShape(JNIEnv *env, jobject obj, jlong addr)
{
    JNI_METHOD("SetSpatialDynamicShape",
        preprocess::InputTensorInfo *info = (preprocess::InputTensorInfo *)addr;

        info->set_spatial_dynamic_shape();
    )
}

/*  We don't use delete operator for native object because we don't own this object:
    no new operator has been used to allocate memory for it */
JNIEXPORT void JNICALL Java_org_intel_openvino_InputTensorInfo_delete(JNIEnv *, jobject, jlong) {}
