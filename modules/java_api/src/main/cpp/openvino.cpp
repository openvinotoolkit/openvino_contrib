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
        std::shared_ptr<const Model> model = std::make_shared<const Model>(modelAddr);

        serialize(model, xml_path, bin_path);
    )
}
