// Copyright (C) 2020-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include <jni.h> // JNI header provided by JDK
#include "openvino/openvino.hpp"

#include "openvino_java.hpp"
#include "jni_common.hpp"

using namespace ov;

JNIEXPORT jlong JNICALL Java_org_intel_openvino_CompiledModel_CreateInferRequest(JNIEnv *env, jobject obj, jlong addr)
{
    JNI_METHOD("CreateInferRequest",
        CompiledModel *compiled_model = (CompiledModel *)addr;

        InferRequest *infer_request = new InferRequest();
        *infer_request = compiled_model->create_infer_request();

        return (jlong)infer_request;
    )
    return 0;
}

JNIEXPORT jobject JNICALL Java_org_intel_openvino_CompiledModel_GetInputs(JNIEnv * env, jobject obj, jlong modelAddr) {
    JNI_METHOD("GetInputs",
        CompiledModel *compiled_model = (CompiledModel *) modelAddr;
        const std::vector<ov::Output<const ov::Node>>& inputs_vec = compiled_model->inputs();

        jclass arrayClass = env->FindClass("java/util/ArrayList");
        jmethodID arrayInit = env->GetMethodID(arrayClass, "<init>", "()V");
        jobject arrayObj = env->NewObject(arrayClass, arrayInit);
        jmethodID arrayAdd = env->GetMethodID(arrayClass, "add", "(Ljava/lang/Object;)Z");

        jclass outputClass = env->FindClass("org/intel/openvino/Output");
        jmethodID outputConstructor = env->GetMethodID(outputClass,"<init>","(J)V");

        Output<const Node> *input;
        for (const auto &item : inputs_vec) {
            input = new Output<const Node>;
            *input = item;

            jobject inputObj = env->NewObject(outputClass, outputConstructor, (jlong)(input));
            env->CallObjectMethod(arrayObj, arrayAdd, inputObj);
        }

        return arrayObj;
    )
    return 0;
}

JNIEXPORT jobject JNICALL Java_org_intel_openvino_CompiledModel_GetOutputs(JNIEnv * env, jobject obj, jlong modelAddr) {
    JNI_METHOD("GetOutputs",
        CompiledModel *compiled_model = (CompiledModel *) modelAddr;
        const std::vector<ov::Output<const ov::Node>>& outputs_vec = compiled_model->outputs();

        jclass arrayClass = env->FindClass("java/util/ArrayList");
        jmethodID arrayInit = env->GetMethodID(arrayClass, "<init>", "()V");
        jobject arrayObj = env->NewObject(arrayClass, arrayInit);
        jmethodID arrayAdd = env->GetMethodID(arrayClass, "add", "(Ljava/lang/Object;)Z");

        jclass outputClass = env->FindClass("org/intel/openvino/Output");
        jmethodID outputConstructor = env->GetMethodID(outputClass,"<init>","(J)V");

        Output<const Node> *output;
        for (const auto &item : outputs_vec) {
            output = new Output<const Node>;
            *output = item;

            jobject outputObj = env->NewObject(outputClass, outputConstructor, (jlong)(output));
            env->CallObjectMethod(arrayObj, arrayAdd, outputObj);
        }

        return arrayObj;
    )
    return 0;
}

JNIEXPORT void JNICALL Java_org_intel_openvino_CompiledModel_delete(JNIEnv *, jobject, jlong addr)
{
    CompiledModel *compiled_model = (CompiledModel *)addr;
    delete compiled_model;
}
