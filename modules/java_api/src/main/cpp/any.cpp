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

JNIEXPORT jobject JNICALL Java_org_intel_openvino_Any_asList(JNIEnv *env, jobject obj, jlong addr) {
    JNI_METHOD("asList",
        Any *obj = (Any *)addr;

        if (obj->is<std::vector<ov::PropertyName>>()) {
            jclass arrayClass = env->FindClass("java/util/ArrayList");
            jmethodID arrayInit = env->GetMethodID(arrayClass, "<init>", "()V");
            jobject arrayObj = env->NewObject(arrayClass, arrayInit);
            jmethodID arrayAdd = env->GetMethodID(arrayClass, "add", "(Ljava/lang/Object;)Z");

            for (const auto& it : obj->as<std::vector<ov::PropertyName>>()) {
                std::string property_name = it;
                jstring string = env->NewStringUTF(property_name.c_str());
                env->CallObjectMethod(arrayObj, arrayAdd, string);
            }

            return arrayObj;
        }
        return vectorToJavaList(env, obj->as<std::vector<std::string>>());
    )
    return 0;
}

JNIEXPORT jstring JNICALL Java_org_intel_openvino_Any_asString(JNIEnv *env, jobject obj, jlong addr) {
    JNI_METHOD("asString",
        Any *obj = (Any *)addr;
        std::string n_string = obj->as<std::string>();
        return env->NewStringUTF(n_string.c_str());
    )
    return 0;
}

JNIEXPORT void JNICALL Java_org_intel_openvino_Any_delete(JNIEnv *, jobject, jlong addr)
{
    Any *obj = (Any *)addr;
    delete obj;
}
