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

JNIEXPORT jobject JNICALL Java_org_intel_openvino_Model_getOutputs(JNIEnv * env, jobject, jlong modelAddr) {
    JNI_METHOD("getOutputs",
        std::shared_ptr<Model> *model = reinterpret_cast<std::shared_ptr<Model> *>(modelAddr);
        const std::vector<ov::Output<ov::Node>>& outputs_vec = (*model)->outputs();

        jclass arrayClass = env->FindClass("java/util/ArrayList");
        jmethodID arrayInit = env->GetMethodID(arrayClass, "<init>", "()V");
        jobject arrayObj = env->NewObject(arrayClass, arrayInit);
        jmethodID arrayAdd = env->GetMethodID(arrayClass, "add", "(Ljava/lang/Object;)Z");

        jclass outputClass = env->FindClass("org/intel/openvino/Output");
        jmethodID outputConstructor = env->GetMethodID(outputClass,"<init>","(J)V");

        Output<Node> *output;
        for (const auto &item : outputs_vec) {
            output = new Output<Node>();
            *output = item;

            jobject outputObj = env->NewObject(outputClass, outputConstructor, (jlong)(output));
            env->CallObjectMethod(arrayObj, arrayAdd, outputObj);
        }

        return arrayObj;
    )
    return 0;
}

JNIEXPORT void JNICALL Java_org_intel_openvino_Model_delete(JNIEnv *, jobject, jlong addr)
{
    std::shared_ptr<Model> *model = reinterpret_cast<std::shared_ptr<Model> *>(addr);
    delete model;
}
