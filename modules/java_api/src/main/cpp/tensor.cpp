// Copyright (C) 2020-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include <jni.h> // JNI header provided by JDK
#include "openvino/openvino.hpp"

#include "openvino_java.hpp"
#include "jni_common.hpp"

using namespace ov;

JNIEXPORT jlong JNICALL Java_org_intel_openvino_Tensor_TensorCArray(JNIEnv *env, jobject, jint type, jintArray shape, jlong matDataAddr)
{
    JNI_METHOD(
        "TensorCArray",
        auto input_type = get_ov_type(type);
        Shape input_shape = jintArrayToVector(env, shape);
        Tensor *ov_tensor = new Tensor();

        // support other types
        switch (input_type) {
            case element::u8:
            {
                uint8_t *data = (uint8_t *)matDataAddr;
                *ov_tensor = Tensor(input_type, input_shape, data);
                break;
            }
            case element::f32:
            {
                float *data = (float *)matDataAddr;
                *ov_tensor = Tensor(input_type, input_shape, data);
                break;
            }
            default: {
                delete ov_tensor;
                throw std::runtime_error("Unsupported element type!");
            }
        }
        return (jlong)ov_tensor;
    )
    return 0;
}

JNIEXPORT jlong JNICALL Java_org_intel_openvino_Tensor_TensorFloat(JNIEnv *env, jobject, jintArray shape, jfloatArray data)
{
    JNI_METHOD(
        "TensorFloat",
        Shape input_shape = jintArrayToVector(env, shape);
        Tensor *ov_tensor = new Tensor(element::f32, input_shape);

        env->GetFloatArrayRegion(data, 0, ov_tensor->get_size(), (jfloat*)ov_tensor->data());

        return (jlong)ov_tensor;
    );
    return 0;
}

JNIEXPORT jlong JNICALL Java_org_intel_openvino_Tensor_TensorInt(JNIEnv *env, jobject, jintArray shape, jintArray data)
{
    JNI_METHOD(
        "TensorInt",
        Shape input_shape = jintArrayToVector(env, shape);
        Tensor *ov_tensor = new Tensor(element::i32, input_shape);

        env->GetIntArrayRegion(data, 0, ov_tensor->get_size(), (jint*)ov_tensor->data());

        return (jlong)ov_tensor;
    );
    return 0;
}

JNIEXPORT jlong JNICALL Java_org_intel_openvino_Tensor_TensorLong(JNIEnv *env, jobject, jintArray shape, jlongArray data)
{
    JNI_METHOD(
        "TensorLong",
        Shape input_shape = jintArrayToVector(env, shape);
        Tensor *ov_tensor = new Tensor(element::i64, input_shape);

        env->GetLongArrayRegion(data, 0, ov_tensor->get_size(), (jlong*)ov_tensor->data());

        return (jlong)ov_tensor;
    );
    return 0;
}

JNIEXPORT jint JNICALL Java_org_intel_openvino_Tensor_GetSize(JNIEnv *env, jobject, jlong addr)
{
    JNI_METHOD(
        "GetSize",
        Tensor *ov_tensor = (Tensor *)addr;
        return (jint)ov_tensor->get_size();
    )
    return 0;
}

JNIEXPORT jintArray JNICALL Java_org_intel_openvino_Tensor_GetShape(JNIEnv *env, jobject, jlong addr)
{
    JNI_METHOD(
        "GetShape",
        Tensor *ov_tensor = (Tensor *)addr;
        Shape shape = ov_tensor->get_shape();

        jintArray result = env->NewIntArray(shape.size());
        if (!result) {
            throw std::runtime_error("Out of memory!");
        } jint *arr = env->GetIntArrayElements(result, nullptr);

        for (int i = 0; i < shape.size(); ++i)
            arr[i] = shape[i];

        env->ReleaseIntArrayElements(result, arr, 0);
        return result;
    )
    return 0;
}

JNIEXPORT jint JNICALL Java_org_intel_openvino_Tensor_GetElementType(JNIEnv *env, jobject, jlong addr)
{
    JNI_METHOD(
        "GetElementType",
        Tensor *ov_tensor = (Tensor *)addr;

        element::Type_t t_type = ov_tensor->get_element_type();
        jint type = static_cast<jint>(t_type);
        return type;
    )
    return 0;
}

JNIEXPORT jfloatArray JNICALL Java_org_intel_openvino_Tensor_asFloat(JNIEnv *env, jobject, jlong addr)
{
    JNI_METHOD(
        "asFloat",
        Tensor *ov_tensor = (Tensor *)addr;

        size_t size = ov_tensor->get_size();
        const float *data = ov_tensor->data<const float>();

        jfloatArray result = env->NewFloatArray(size);
        if (!result) {
            throw std::runtime_error("Out of memory!");
        } jfloat *arr = env->GetFloatArrayElements(result, nullptr);

        for (size_t i = 0; i < size; ++i)
            arr[i] = data[i];

        env->ReleaseFloatArrayElements(result, arr, 0);
        return result;
    )
    return 0;
}

JNIEXPORT jintArray JNICALL Java_org_intel_openvino_Tensor_asInt(JNIEnv *env, jobject, jlong addr)
{
    JNI_METHOD(
        "asInt",
        Tensor *ov_tensor = (Tensor *)addr;

        size_t size = ov_tensor->get_size();
        const int *data = ov_tensor->data<const int>();

        jintArray result = env->NewIntArray(size);
        if (!result) {
            throw std::runtime_error("Out of memory!");
        } jint *arr = env->GetIntArrayElements(result, nullptr);

        for (size_t i = 0; i < size; ++i)
            arr[i] = data[i];

        env->ReleaseIntArrayElements(result, arr, 0);
        return result;
    )
    return 0;
}

JNIEXPORT void JNICALL Java_org_intel_openvino_Tensor_delete(JNIEnv *, jobject, jlong addr)
{
    Tensor *tensor = (Tensor *)addr;
    delete tensor;
}
