// Copyright (C) 2020-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include <jni.h> // JNI header provided by JDK
#include "openvino/openvino.hpp"

#include "openvino_java.hpp"
#include "jni_common.hpp"

using namespace ov;

JNIEXPORT jlong JNICALL Java_org_intel_openvino_Core_GetCore(JNIEnv *env, jobject obj)
{

    JNI_METHOD("GetCore",
        Core *core = new Core();
        return (jlong)core;
    )
    return 0;
}

JNIEXPORT jlong JNICALL Java_org_intel_openvino_Core_ReadModel(JNIEnv *env, jobject obj, jlong coreAddr, jstring xml)
{
    JNI_METHOD("ReadModel",
        std::string n_xml = jstringToString(env, xml);
        Core *core = (Core *)coreAddr;

        std::shared_ptr<Model> *model = new std::shared_ptr<Model>;
        *model = core->read_model(n_xml);

        return reinterpret_cast<jlong>(model);
    )
    return 0;
}

JNIEXPORT jlong JNICALL Java_org_intel_openvino_Core_ReadModel1(JNIEnv *env, jobject obj, jlong coreAddr, jstring xml, jstring bin)
{

    JNI_METHOD("ReadModel1",
        std::string n_xml = jstringToString(env, xml);
        std::string n_bin = jstringToString(env, bin);

        Core *core = (Core *)coreAddr;
        std::shared_ptr<Model> *model = new std::shared_ptr<Model>;
        *model = core->read_model(n_xml, n_bin);

        return reinterpret_cast<jlong>(model);
    )
    return 0;
}

JNIEXPORT jlong JNICALL Java_org_intel_openvino_Core_CompileModel(JNIEnv *env, jobject obj, jlong coreAddr, jlong netAddr, jstring device)
{
    JNI_METHOD("CompileModel",
        std::string n_device = jstringToString(env, device);

        Core *core = (Core *)coreAddr;
        std::shared_ptr<Model> *model = reinterpret_cast<std::shared_ptr<Model> *>(netAddr);

        CompiledModel *compiled_model = new CompiledModel();
        *compiled_model = core->compile_model(*model, n_device);

        return (jlong)compiled_model;
    )
    return 0;
}

JNIEXPORT void JNICALL Java_org_intel_openvino_Core_delete(JNIEnv *, jobject, jlong addr)
{
    Core *core = (Core *)addr;
    delete core;
}
