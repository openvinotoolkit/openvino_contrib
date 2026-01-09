// Copyright (C) 2020-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include <jni.h>

#ifdef __cplusplus
extern "C"
{
#endif
    //ov
    JNIEXPORT void JNICALL Java_org_intel_openvino_Openvino_serialize(JNIEnv *, jobject, jlong, jstring, jstring);
    JNIEXPORT void JNICALL Java_org_intel_openvino_Openvino_SaveModel(JNIEnv *, jobject, jlong, jstring, jboolean);

    // ov::Core
    JNIEXPORT jlong JNICALL Java_org_intel_openvino_Core_GetCore(JNIEnv *, jobject);
    JNIEXPORT jlong JNICALL Java_org_intel_openvino_Core_GetCore1(JNIEnv *, jobject, jstring);
    JNIEXPORT jlong JNICALL Java_org_intel_openvino_Core_ReadModel(JNIEnv *, jobject, jlong, jstring);
    JNIEXPORT jlong JNICALL Java_org_intel_openvino_Core_ReadModel1(JNIEnv *, jobject, jlong, jstring, jstring);
    JNIEXPORT jlong JNICALL Java_org_intel_openvino_Core_CompileModel(JNIEnv *, jobject, jlong, jlong, jstring);
    JNIEXPORT jlong JNICALL Java_org_intel_openvino_Core_CompileModel1(JNIEnv *, jobject, jlong, jstring);
    JNIEXPORT jlong JNICALL Java_org_intel_openvino_Core_CompileModel2(JNIEnv *, jobject, jlong, jstring, jstring);
    JNIEXPORT jlong JNICALL Java_org_intel_openvino_Core_CompileModel3(JNIEnv *, jobject, jlong, jstring, jstring, jobject);
    JNIEXPORT jlong JNICALL Java_org_intel_openvino_Core_CompileModel4(JNIEnv *, jobject, jlong, jlong, jstring, jobject);
    JNIEXPORT jlong JNICALL Java_org_intel_openvino_Core_GetProperty(JNIEnv *, jobject, jlong, jstring, jstring);
    JNIEXPORT void JNICALL Java_org_intel_openvino_Core_SetProperty(JNIEnv *, jobject, jlong, jstring, jobject);
    JNIEXPORT jobject JNICALL Java_org_intel_openvino_Core_GetAvailableDevices(JNIEnv *, jobject, jlong);
    JNIEXPORT void JNICALL Java_org_intel_openvino_Core_delete(JNIEnv *, jobject, jlong);

    // ov::Any
    JNIEXPORT jint JNICALL Java_org_intel_openvino_Any_asInt(JNIEnv *, jobject, jlong);
    JNIEXPORT jobject JNICALL Java_org_intel_openvino_Any_asList(JNIEnv *, jobject, jlong);
    JNIEXPORT jstring JNICALL Java_org_intel_openvino_Any_asString(JNIEnv *, jobject, jlong);

    // ov::Model
    JNIEXPORT jstring JNICALL Java_org_intel_openvino_Model_getName(JNIEnv *, jobject, jlong);
    JNIEXPORT jlong JNICALL Java_org_intel_openvino_Model_getBatch(JNIEnv *, jobject, jlong);
    JNIEXPORT jobject JNICALL Java_org_intel_openvino_Model_getOutputs(JNIEnv *, jobject, jlong);
    JNIEXPORT jlong JNICALL Java_org_intel_openvino_Model_getOutput(JNIEnv *, jobject, jlong);
    JNIEXPORT void JNICALL Java_org_intel_openvino_Model_Reshape(JNIEnv *, jobject, jlong, jintArray);
    JNIEXPORT jlong JNICALL Java_org_intel_openvino_Model_getInput(JNIEnv *, jobject, jlong);
    JNIEXPORT void JNICALL Java_org_intel_openvino_Model_delete(JNIEnv *, jobject, jlong);

    // ov::CompiledModel
    JNIEXPORT jlong JNICALL Java_org_intel_openvino_CompiledModel_CreateInferRequest(JNIEnv *, jobject, jlong);
    JNIEXPORT void JNICALL Java_org_intel_openvino_CompiledModel_delete(JNIEnv *, jobject, jlong);
    JNIEXPORT jobject JNICALL Java_org_intel_openvino_CompiledModel_GetInputs(JNIEnv *, jobject, jlong);
    JNIEXPORT jobject JNICALL Java_org_intel_openvino_CompiledModel_GetOutputs(JNIEnv *, jobject, jlong);

    // ov::InferRequest
    JNIEXPORT void JNICALL Java_org_intel_openvino_InferRequest_Infer(JNIEnv *, jobject, jlong);
    JNIEXPORT void JNICALL Java_org_intel_openvino_InferRequest_StartAsync(JNIEnv *, jobject, jlong);
    JNIEXPORT void JNICALL Java_org_intel_openvino_InferRequest_Wait(JNIEnv *, jobject, jlong);
    JNIEXPORT void JNICALL Java_org_intel_openvino_InferRequest_SetInputTensor(JNIEnv *, jobject, jlong, jlong);
    JNIEXPORT void JNICALL Java_org_intel_openvino_InferRequest_SetOutputTensor(JNIEnv *, jobject, jlong, jlong);
    JNIEXPORT jlong JNICALL Java_org_intel_openvino_InferRequest_GetOutputTensor(JNIEnv *, jobject, jlong);
    JNIEXPORT jlong JNICALL Java_org_intel_openvino_InferRequest_GetTensor(JNIEnv *, jobject, jlong, jstring);
    JNIEXPORT void JNICALL Java_org_intel_openvino_InferRequest_SetTensor(JNIEnv *, jobject, jlong, jstring, jlong);
    JNIEXPORT void JNICALL Java_org_intel_openvino_InferRequest_delete(JNIEnv *, jobject, jlong);

    // ov::Tensor
    JNIEXPORT jlong JNICALL Java_org_intel_openvino_Tensor_TensorCArray(JNIEnv *, jobject, jint, jintArray, jlong);
    JNIEXPORT jlong JNICALL Java_org_intel_openvino_Tensor_TensorFloat(JNIEnv *, jobject, jintArray, jfloatArray);
    JNIEXPORT jlong JNICALL Java_org_intel_openvino_Tensor_TensorInt(JNIEnv *, jobject, jintArray, jintArray);
    JNIEXPORT jlong JNICALL Java_org_intel_openvino_Tensor_TensorLong(JNIEnv *, jobject, jintArray, jlongArray);
    JNIEXPORT jint JNICALL Java_org_intel_openvino_Tensor_GetSize(JNIEnv *, jobject, jlong);
    JNIEXPORT jintArray JNICALL Java_org_intel_openvino_Tensor_GetShape(JNIEnv *, jobject, jlong);
    JNIEXPORT jint JNICALL Java_org_intel_openvino_Tensor_GetElementType(JNIEnv *, jobject, jlong);
    JNIEXPORT jfloatArray JNICALL Java_org_intel_openvino_Tensor_asFloat(JNIEnv *, jobject, jlong);
    JNIEXPORT jintArray JNICALL Java_org_intel_openvino_Tensor_asInt(JNIEnv *, jobject, jlong);
    JNIEXPORT void JNICALL Java_org_intel_openvino_Tensor_delete(JNIEnv *, jobject, jlong);

    // ov::PrePostProcessor
    JNIEXPORT jlong JNICALL Java_org_intel_openvino_PrePostProcessor_GetPrePostProcessor(JNIEnv *, jobject, jlong);
    JNIEXPORT jlong JNICALL Java_org_intel_openvino_PrePostProcessor_Input(JNIEnv *, jobject, jlong);
    JNIEXPORT jlong JNICALL Java_org_intel_openvino_PrePostProcessor_Output(JNIEnv *, jobject, jlong);
    JNIEXPORT jlong JNICALL Java_org_intel_openvino_PrePostProcessor_Build(JNIEnv *, jobject, jlong);
    JNIEXPORT void JNICALL Java_org_intel_openvino_PrePostProcessor_delete(JNIEnv *, jobject, jlong);

    // ov::preprocess::InputInfo
    JNIEXPORT jlong JNICALL Java_org_intel_openvino_InputInfo_preprocess(JNIEnv *, jobject, jlong);
    JNIEXPORT jlong JNICALL Java_org_intel_openvino_InputInfo_tensor(JNIEnv *, jobject, jlong);
    JNIEXPORT jlong JNICALL Java_org_intel_openvino_InputInfo_model(JNIEnv *, jobject, jlong);
    JNIEXPORT void JNICALL Java_org_intel_openvino_InputInfo_delete(JNIEnv *, jobject, jlong);

    // ov::preprocess::InputTensorInfo
    JNIEXPORT void JNICALL Java_org_intel_openvino_InputTensorInfo_SetElementType(JNIEnv *, jobject, jlong, jint);
    JNIEXPORT void JNICALL Java_org_intel_openvino_InputTensorInfo_SetLayout(JNIEnv *, jobject, jlong, jlong);
    JNIEXPORT void JNICALL Java_org_intel_openvino_InputTensorInfo_SetSpatialStaticShape(JNIEnv *, jobject, jlong, jint, jint);
    JNIEXPORT void JNICALL Java_org_intel_openvino_InputTensorInfo_SetSpatialDynamicShape(JNIEnv *, jobject, jlong);
    JNIEXPORT void JNICALL Java_org_intel_openvino_InputTensorInfo_delete(JNIEnv *, jobject, jlong);

    // ov::Layout
    JNIEXPORT jlong JNICALL Java_org_intel_openvino_Layout_GetLayout(JNIEnv *, jobject, jstring);
    JNIEXPORT jint JNICALL Java_org_intel_openvino_Layout_HeightIdx(JNIEnv *, jobject, jlong);
    JNIEXPORT jint JNICALL Java_org_intel_openvino_Layout_WidthIdx(JNIEnv *, jobject, jlong);
    JNIEXPORT void JNICALL Java_org_intel_openvino_Layout_delete(JNIEnv *, jobject, jlong);

    // ov::preprocess::PreProcessSteps
    JNIEXPORT void JNICALL Java_org_intel_openvino_PreProcessSteps_Resize(JNIEnv *, jobject, jlong, jint);
    JNIEXPORT void JNICALL Java_org_intel_openvino_PreProcessSteps_delete(JNIEnv *, jobject, jlong);

    // ov::preprocess::InputModelInfo
    JNIEXPORT void JNICALL Java_org_intel_openvino_InputModelInfo_SetLayout(JNIEnv *, jobject, jlong, jlong);
    JNIEXPORT void JNICALL Java_org_intel_openvino_InputModelInfo_delete(JNIEnv *, jobject, jlong);

    // ov::preprocess::OutputInfo
    JNIEXPORT jlong JNICALL Java_org_intel_openvino_OutputInfo_getTensor(JNIEnv *, jobject, jlong);
    JNIEXPORT void JNICALL Java_org_intel_openvino_OutputInfo_delete(JNIEnv *, jobject, jlong);

    // ov::Dimension
    JNIEXPORT jint JNICALL Java_org_intel_openvino_Dimension_getLength(JNIEnv *, jobject, jlong);

    // ov::Output<ov::Node>
    JNIEXPORT jstring JNICALL Java_org_intel_openvino_Output_GetAnyName(JNIEnv *, jobject, jlong);
    JNIEXPORT jintArray JNICALL Java_org_intel_openvino_Output_GetShape(JNIEnv *, jobject, jlong);
    JNIEXPORT jlong JNICALL Java_org_intel_openvino_Output_GetPartialShape(JNIEnv *, jobject, jlong);
    JNIEXPORT int JNICALL Java_org_intel_openvino_Output_GetElementType(JNIEnv *, jobject, jlong);
    JNIEXPORT void JNICALL Java_org_intel_openvino_Output_delete(JNIEnv *, jobject, jlong);

    // ov::PartialShape
    JNIEXPORT jlong JNICALL Java_org_intel_openvino_PartialShape_GetDimension(JNIEnv *, jobject, jlong, jint);
    JNIEXPORT jintArray JNICALL Java_org_intel_openvino_PartialShape_GetMaxShape(JNIEnv *, jobject, jlong);
    JNIEXPORT jintArray JNICALL Java_org_intel_openvino_PartialShape_GetMinShape(JNIEnv *, jobject, jlong);

#ifdef __cplusplus
}
#endif
