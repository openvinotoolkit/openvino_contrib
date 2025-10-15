// Copyright (C) 2020-2023 Intel Corporation
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

JNIEXPORT jlong JNICALL Java_org_intel_openvino_Core_GetCore1(JNIEnv *env, jobject obj, jstring xmlConfigFile)
{

    JNI_METHOD("GetCore1",
        std::string n_xml = jstringToString(env, xmlConfigFile);
        Core *core = new Core(n_xml);
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

JNIEXPORT jlong JNICALL Java_org_intel_openvino_Core_CompileModel1(JNIEnv *env, jobject obj, jlong coreAddr, jstring path)
{
    JNI_METHOD("CompileModel1",
        std::string n_path = jstringToString(env, path);

        Core *core = (Core *)coreAddr;
        CompiledModel *compiled_model = new CompiledModel();
        *compiled_model = core->compile_model(n_path);

        return (jlong)compiled_model;
    )
    return 0;
}

JNIEXPORT jlong JNICALL Java_org_intel_openvino_Core_CompileModel2(JNIEnv *env, jobject obj, jlong coreAddr, jstring path, jstring device)
{
    JNI_METHOD("CompileModel2",
        std::string n_device = jstringToString(env, device);
        std::string n_path = jstringToString(env, path);

        Core *core = (Core *)coreAddr;
        CompiledModel *compiled_model = new CompiledModel();
        *compiled_model = core->compile_model(n_path, n_device);

        return (jlong)compiled_model;
    )
    return 0;
}

JNIEXPORT jlong JNICALL Java_org_intel_openvino_Core_CompileModel3(JNIEnv *env, jobject obj, jlong coreAddr, jstring path, jstring device, jobject props)
{
    JNI_METHOD("CompileModel3",
        std::string n_device = jstringToString(env, device);
        std::string n_path = jstringToString(env, path);
        AnyMap map;
        for (const auto& it : javaMapToMap(env, props)) {
            map[it.first] = it.second;
        }

        Core *core = (Core *)coreAddr;
        // AnyMap will be copied inside compile_model, so we don't have to track the lifetime of this object
        CompiledModel *compiled_model = new CompiledModel();
        *compiled_model = core->compile_model(n_path, n_device, map);

        return (jlong)compiled_model;
    )
    return 0;
}

JNIEXPORT jlong JNICALL Java_org_intel_openvino_Core_CompileModel4(JNIEnv *env, jobject obj, jlong coreAddr, jlong modelAddr, jstring device, jobject props)
{
    JNI_METHOD("CompileModel4",
        std::string n_device = jstringToString(env, device);
        std::shared_ptr<Model> *model = reinterpret_cast<std::shared_ptr<Model> *>(modelAddr);
        AnyMap map;
        for (const auto& it : javaMapToMap(env, props)) {
            map[it.first] = it.second;
        }

        Core *core = (Core *)coreAddr;
        CompiledModel *compiled_model = new CompiledModel();
        *compiled_model = core->compile_model(*model, n_device, map);

        return (jlong)compiled_model;
    )
    return 0;
}

JNIEXPORT jlong JNICALL Java_org_intel_openvino_Core_GetProperty(JNIEnv *env, jobject obj, jlong coreAddr, jstring device, jstring name)
{
    JNI_METHOD("GetProperty",
        std::string n_device = jstringToString(env, device);
        std::string n_name = jstringToString(env, name);

        Core *core = (Core *)coreAddr;

        Any *property = new Any();
        *property = core->get_property(n_device, n_name);

        return (jlong)property;
    )
    return 0;
}

JNIEXPORT void JNICALL Java_org_intel_openvino_Core_SetProperty(JNIEnv *env, jobject obj, jlong coreAddr, jstring device, jobject prop) {
    JNI_METHOD("SetProperty",
        std::string n_device = jstringToString(env, device);
        Core *core = (Core *)coreAddr;
        AnyMap map;
        for (const auto& it : javaMapToMap(env, prop)) {
            map[it.first] = it.second;
        }
        core->set_property(n_device, map);
    )
}

JNIEXPORT jobject JNICALL Java_org_intel_openvino_Core_GetAvailableDevices(JNIEnv *env, jobject obj, jlong coreAddr) {
    JNI_METHOD("GetAvailableDevices",
        Core *core = (Core *)coreAddr;
        const std::vector<std::string>& devices_vec = core->get_available_devices();

        jclass arrayClass = env->FindClass("java/util/ArrayList");
        jmethodID arrayInit = env->GetMethodID(arrayClass, "<init>", "()V");
        jobject arrayObj = env->NewObject(arrayClass, arrayInit);
        jmethodID arrayAdd = env->GetMethodID(arrayClass, "add", "(Ljava/lang/Object;)Z");

        for (const std::string &item : devices_vec) {
            jstring device = env->NewStringUTF(item.c_str());
            env->CallObjectMethod(arrayObj, arrayAdd, device);
        }

        return arrayObj;
    )
    return 0;
}

JNIEXPORT void JNICALL Java_org_intel_openvino_Core_delete(JNIEnv *, jobject, jlong addr)
{
    Core *core = (Core *)addr;
    delete core;
}
