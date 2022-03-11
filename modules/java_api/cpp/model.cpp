// Copyright (C) 2020-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include <jni.h> // JNI header provided by JDK
#include "openvino/openvino.hpp"

#include "openvino_java.hpp"
#include "jni_common.hpp"

using namespace ov;

JNIEXPORT jstring JNICALL Java_org_intel_openvino_Model_getName(JNIEnv *env, jobject, jlong addr)
{
    JNI_METHOD("getName",
        std::shared_ptr<Model> *model = reinterpret_cast<std::shared_ptr<Model> *>(addr);
        return env->NewStringUTF((*model)->get_name().c_str());
    )
    return 0;
}

JNIEXPORT jlong JNICALL Java_org_intel_openvino_Model_getBatch(JNIEnv *env, jobject, jlong addr)
{
    JNI_METHOD("getBatch",
        std::shared_ptr<Model> *model = reinterpret_cast<std::shared_ptr<Model> *>(addr);

        Dimension *dim = new Dimension();
        *dim = get_batch((*model));

        return (jlong)dim;
    )
    return 0;
}

JNIEXPORT void JNICALL Java_org_intel_openvino_Model_delete(JNIEnv *, jobject, jlong addr)
{
    std::shared_ptr<Model> *model = reinterpret_cast<std::shared_ptr<Model> *>(addr);
    delete model;
}
