// Copyright (C) 2020-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include <jni.h> // JNI header provided by JDK
#include "openvino/openvino.hpp"
#include "openvino/core/graph_util.hpp"

#include "openvino_java.hpp"
#include "jni_common.hpp"

using namespace ov;

JNIEXPORT void JNICALL Java_org_intel_openvino_Openvino_serialize(JNIEnv *env, jobject obj, jlong modelAddr, jstring xmlPath, jstring binPath)
{
    JNI_METHOD("serialize",
        std::string xml_path = jstringToString(env, xmlPath);
        std::string bin_path = jstringToString(env, binPath);
        std::shared_ptr<const Model> *model = reinterpret_cast<std::shared_ptr<const Model> *>(modelAddr);

        serialize(*model, xml_path, bin_path);
    )
}

JNIEXPORT void JNICALL Java_org_intel_openvino_Openvino_SaveModel(JNIEnv *env, jobject obj, jlong modelAddr, jstring outputModel, jboolean compressToFp16)
{
    JNI_METHOD("SaveModel",
        std::string n_output_model = jstringToString(env, outputModel);
        std::shared_ptr<const Model> *model = reinterpret_cast<std::shared_ptr<const Model> *>(modelAddr);
        save_model(*model, n_output_model, (bool) compressToFp16);
    )
}
