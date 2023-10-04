// Copyright (C) 2020-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include <inference_engine.hpp>

#include "openvino_java.hpp"
#include "jni_common.hpp"

using namespace InferenceEngine;

JNIEXPORT void JNICALL Java_org_intel_openvino_compatibility_LockedMemory_asFloat(JNIEnv *env, jobject obj, jlong addr, jfloatArray res)
{
    static const char method_name[] = "asFloat";
    try
    {
        LockedMemory<const void> *lmem = (LockedMemory<const void> *) addr;
        const float *buffer = lmem->as<const float *>();

        const jsize size = env->GetArrayLength(res);
        env->SetFloatArrayRegion(res, 0, size, buffer);
    }
    catch (const std::exception &e)
    {
        throwJavaException(env, &e, method_name);
    }
    catch (...)
    {
        throwJavaException(env, 0, method_name);
    }
}

JNIEXPORT void JNICALL Java_org_intel_openvino_compatibility_LockedMemory_asByte(JNIEnv *env, jobject obj, jlong addr, jbyteArray res)
{
    static const char method_name[] = "asByte";
    try
    {
        LockedMemory<const void> *lmem = (LockedMemory<const void> *) addr;
        const uint8_t *buffer = lmem->as<const uint8_t *>();

        const jsize size = env->GetArrayLength(res);
        env->SetByteArrayRegion(res, 0, size, reinterpret_cast<const jbyte*>(buffer));
    }
    catch (const std::exception &e)
    {
        throwJavaException(env, &e, method_name);
    }
    catch (...)
    {
        throwJavaException(env, 0, method_name);
    }
}

JNIEXPORT void JNICALL Java_org_intel_openvino_compatibility_LockedMemory_asLong(JNIEnv *env, jobject obj, jlong addr, jlongArray res)
{
    static const char method_name[] = "asLong";
    try
    {
        LockedMemory<const void> *lmem = (LockedMemory<const void> *) addr;
        const int64_t *buffer = lmem->as<const int64_t *>();

        const jsize size = env->GetArrayLength(res);
        env->SetLongArrayRegion(res, 0, size, reinterpret_cast<const jlong*>(buffer));
    }
    catch (const std::exception &e)
    {
        throwJavaException(env, &e, method_name);
    }
    catch (...)
    {
        throwJavaException(env, 0, method_name);
    }
}

JNIEXPORT void JNICALL Java_org_intel_openvino_compatibility_LockedMemory_asInt(JNIEnv *env, jobject obj, jlong addr, jintArray res)
{
    static const char method_name[] = "asInt";
    try
    {
        LockedMemory<const void> *lmem = (LockedMemory<const void> *) addr;
        const int32_t *buffer = lmem->as<const int32_t *>();

        const jsize size = env->GetArrayLength(res);
        env->SetIntArrayRegion(res, 0, size, reinterpret_cast<const jint*>(buffer));
    }
    catch (const std::exception &e)
    {
        throwJavaException(env, &e, method_name);
    }
    catch (...)
    {
        throwJavaException(env, 0, method_name);
    }
}

JNIEXPORT void JNICALL Java_org_intel_openvino_compatibility_LockedMemory_delete(JNIEnv *, jobject, jlong addr)
{
    LockedMemory<const void> *lmem = (LockedMemory<const void> *) addr;
    delete lmem;
}
