// Copyright (C) 2020-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include <jni.h>

#ifdef __cplusplus
extern "C"
{
#endif

    /* -------------------------------------- DEPRICATED API ------------------------------------------*/

    //
    // IECore
    //
    JNIEXPORT jlong JNICALL Java_org_intel_openvino_compatibility_IECore_ReadNetwork1(JNIEnv *, jobject, jlong, jstring, jstring);
    JNIEXPORT jlong JNICALL Java_org_intel_openvino_compatibility_IECore_ReadNetwork(JNIEnv *, jobject, jlong, jstring);
    JNIEXPORT jlong JNICALL Java_org_intel_openvino_compatibility_IECore_LoadNetwork(JNIEnv *, jobject, jlong, jlong, jstring);
    JNIEXPORT jlong JNICALL Java_org_intel_openvino_compatibility_IECore_LoadNetwork1(JNIEnv *, jobject, jlong, jlong, jstring, jobject);
    JNIEXPORT void JNICALL Java_org_intel_openvino_compatibility_IECore_RegisterPlugin(JNIEnv *, jobject, jlong, jstring, jstring);
    JNIEXPORT void JNICALL Java_org_intel_openvino_compatibility_IECore_UnregisterPlugin(JNIEnv *, jobject, jlong, jstring);
    JNIEXPORT void JNICALL Java_org_intel_openvino_compatibility_IECore_AddExtension(JNIEnv *, jobject, jlong, jstring);
    JNIEXPORT void JNICALL Java_org_intel_openvino_compatibility_IECore_AddExtension1(JNIEnv *, jobject, jlong, jstring, jstring);
    JNIEXPORT void JNICALL Java_org_intel_openvino_compatibility_IECore_RegisterPlugins(JNIEnv *, jobject, jlong, jstring);
    JNIEXPORT void JNICALL Java_org_intel_openvino_compatibility_IECore_SetConfig(JNIEnv *, jobject, jlong, jobject, jstring);
    JNIEXPORT void JNICALL Java_org_intel_openvino_compatibility_IECore_SetConfig1(JNIEnv *, jobject, jlong, jobject);
    JNIEXPORT jlong JNICALL Java_org_intel_openvino_compatibility_IECore_GetConfig(JNIEnv *, jobject, jlong, jstring, jstring);
    JNIEXPORT jlong JNICALL Java_org_intel_openvino_compatibility_IECore_GetCore(JNIEnv *, jobject);
    JNIEXPORT jlong JNICALL Java_org_intel_openvino_compatibility_IECore_GetCore1(JNIEnv *, jobject, jstring);
    JNIEXPORT void JNICALL Java_org_intel_openvino_compatibility_IECore_delete(JNIEnv *, jobject, jlong);

    //
    // CNNNetwork
    //
    JNIEXPORT jstring JNICALL Java_org_intel_openvino_compatibility_CNNNetwork_getName(JNIEnv *, jobject, jlong);
    JNIEXPORT jint JNICALL Java_org_intel_openvino_compatibility_CNNNetwork_getBatchSize(JNIEnv *, jobject, jlong);
    JNIEXPORT jobject JNICALL Java_org_intel_openvino_compatibility_CNNNetwork_GetInputsInfo(JNIEnv *, jobject, jlong);
    JNIEXPORT jobject JNICALL Java_org_intel_openvino_compatibility_CNNNetwork_GetOutputsInfo(JNIEnv *, jobject, jlong);
    JNIEXPORT void JNICALL Java_org_intel_openvino_compatibility_CNNNetwork_reshape(JNIEnv *, jobject, jlong, jobject);
    JNIEXPORT jobject JNICALL Java_org_intel_openvino_compatibility_CNNNetwork_getInputShapes(JNIEnv *, jobject, jlong);
    JNIEXPORT void JNICALL Java_org_intel_openvino_compatibility_CNNNetwork_addOutput(JNIEnv *, jobject, jlong, jstring, jint);
    JNIEXPORT void JNICALL Java_org_intel_openvino_compatibility_CNNNetwork_addOutput1(JNIEnv *, jobject, jlong, jstring);
    JNIEXPORT void JNICALL Java_org_intel_openvino_compatibility_CNNNetwork_delete(JNIEnv *, jobject, jlong);

    //
    // InferRequest
    //
    JNIEXPORT void JNICALL Java_org_intel_openvino_compatibility_InferRequest_Infer(JNIEnv *, jobject, jlong);
    JNIEXPORT void JNICALL Java_org_intel_openvino_compatibility_InferRequest_StartAsync(JNIEnv *, jobject, jlong);
    JNIEXPORT jint JNICALL Java_org_intel_openvino_compatibility_InferRequest_Wait(JNIEnv *, jobject, jlong, jint);
    JNIEXPORT jint JNICALL Java_org_intel_openvino_compatibility_InferRequest_SetCompletionCallback(JNIEnv *, jobject, jlong, jobject);
    JNIEXPORT long JNICALL Java_org_intel_openvino_compatibility_InferRequest_GetBlob(JNIEnv *, jobject, jlong, jstring);
    JNIEXPORT void JNICALL Java_org_intel_openvino_compatibility_InferRequest_SetBlob(JNIEnv *, jobject, jlong, jstring, jlong);
    JNIEXPORT jobject JNICALL Java_org_intel_openvino_compatibility_InferRequest_GetPerformanceCounts(JNIEnv *, jobject, jlong);
    JNIEXPORT void JNICALL Java_org_intel_openvino_compatibility_InferRequest_delete(JNIEnv *, jobject, jlong);

    //
    // ExecutableNetwork
    //
    JNIEXPORT jlong JNICALL Java_org_intel_openvino_compatibility_ExecutableNetwork_CreateInferRequest(JNIEnv *, jobject, jlong);
    JNIEXPORT jlong JNICALL Java_org_intel_openvino_compatibility_ExecutableNetwork_GetMetric(JNIEnv *, jobject, jlong, jstring);
    JNIEXPORT void JNICALL Java_org_intel_openvino_compatibility_ExecutableNetwork_delete(JNIEnv *, jobject, jlong);

    //
    // Blob
    //
    JNIEXPORT jlong JNICALL Java_org_intel_openvino_compatibility_Blob_GetTensorDesc(JNIEnv *, jobject, jlong);
    JNIEXPORT jlong JNICALL Java_org_intel_openvino_compatibility_Blob_GetBlob(JNIEnv *, jobject, jlong);
    JNIEXPORT jlong JNICALL Java_org_intel_openvino_compatibility_Blob_BlobByte(JNIEnv *, jobject, jlong, jbyteArray);
    JNIEXPORT jlong JNICALL Java_org_intel_openvino_compatibility_Blob_BlobFloat(JNIEnv *, jobject, jlong, jfloatArray);
    JNIEXPORT jlong JNICALL Java_org_intel_openvino_compatibility_Blob_BlobInt(JNIEnv *, jobject, jlong, jintArray);
    JNIEXPORT jlong JNICALL Java_org_intel_openvino_compatibility_Blob_BlobLong(JNIEnv *, jobject, jlong, jlongArray);
    JNIEXPORT jlong JNICALL Java_org_intel_openvino_compatibility_Blob_BlobCArray(JNIEnv *, jobject, jlong, jlong);
    JNIEXPORT jint JNICALL Java_org_intel_openvino_compatibility_Blob_size(JNIEnv *, jobject, jlong);
    JNIEXPORT jlong JNICALL Java_org_intel_openvino_compatibility_Blob_rmap(JNIEnv *, jobject, jlong);
    JNIEXPORT void JNICALL Java_org_intel_openvino_compatibility_Blob_delete(JNIEnv *, jobject, jlong);

    //
    // LockedMemory
    //
    JNIEXPORT void JNICALL Java_org_intel_openvino_compatibility_LockedMemory_asByte(JNIEnv *, jobject, jlong, jbyteArray);
    JNIEXPORT void JNICALL Java_org_intel_openvino_compatibility_LockedMemory_asFloat(JNIEnv *, jobject, jlong, jfloatArray);
    JNIEXPORT void JNICALL Java_org_intel_openvino_compatibility_LockedMemory_asLong(JNIEnv *, jobject, jlong, jlongArray);
    JNIEXPORT void JNICALL Java_org_intel_openvino_compatibility_LockedMemory_asInt(JNIEnv *, jobject, jlong, jintArray);
    JNIEXPORT void JNICALL Java_org_intel_openvino_compatibility_LockedMemory_delete(JNIEnv *, jobject, jlong);

    //
    // InputInfo
    //
    JNIEXPORT jlong JNICALL Java_org_intel_openvino_compatibility_InputInfo_getPreProcess(JNIEnv *, jobject, jlong);
    JNIEXPORT void JNICALL Java_org_intel_openvino_compatibility_InputInfo_SetLayout(JNIEnv *, jobject, jlong, jint);
    JNIEXPORT jint JNICALL Java_org_intel_openvino_compatibility_InputInfo_getLayout(JNIEnv *, jobject, jlong);
    JNIEXPORT void JNICALL Java_org_intel_openvino_compatibility_InputInfo_SetPrecision(JNIEnv *, jobject, jlong, jint);
    JNIEXPORT jint JNICALL Java_org_intel_openvino_compatibility_InputInfo_getPrecision(JNIEnv *, jobject, jlong);
    JNIEXPORT jlong JNICALL Java_org_intel_openvino_compatibility_InputInfo_GetTensorDesc(JNIEnv *, jobject, jlong);
    JNIEXPORT void JNICALL Java_org_intel_openvino_compatibility_InputInfo_delete(JNIEnv *, jobject, jlong);

    //
    // PreProcessInfo
    //
    JNIEXPORT void JNICALL Java_org_intel_openvino_compatibility_PreProcessInfo_SetResizeAlgorithm(JNIEnv *, jobject, jlong, jint);
    JNIEXPORT void JNICALL Java_org_intel_openvino_compatibility_PreProcessInfo_delete(JNIEnv *, jobject, jlong);

    //
    // TensorDesc
    //
    JNIEXPORT jlong JNICALL Java_org_intel_openvino_compatibility_TensorDesc_GetTensorDesc(JNIEnv *, jobject, jint, jintArray, jint);
    JNIEXPORT jintArray JNICALL Java_org_intel_openvino_compatibility_TensorDesc_GetDims(JNIEnv *, jobject, jlong);
    JNIEXPORT jint JNICALL Java_org_intel_openvino_compatibility_TensorDesc_getLayout(JNIEnv *, jobject, jlong);
    JNIEXPORT jint JNICALL Java_org_intel_openvino_compatibility_TensorDesc_getPrecision(JNIEnv *, jobject, jlong);
    JNIEXPORT void JNICALL Java_org_intel_openvino_compatibility_TensorDesc_delete(JNIEnv *, jobject, jlong);

    //
    // Parameter
    //
    JNIEXPORT jstring JNICALL Java_org_intel_openvino_compatibility_Parameter_asString(JNIEnv *, jobject, jlong);
    JNIEXPORT jint JNICALL Java_org_intel_openvino_compatibility_Parameter_asInt(JNIEnv *, jobject, jlong);
    JNIEXPORT void JNICALL Java_org_intel_openvino_compatibility_Parameter_delete(JNIEnv *, jobject, jlong);

    //
    // Data
    //
    JNIEXPORT void JNICALL Java_org_intel_openvino_compatibility_Data_setLayout(JNIEnv *, jobject, jlong, jint);
    JNIEXPORT jint JNICALL Java_org_intel_openvino_compatibility_Data_getLayout(JNIEnv *, jobject, jlong);
    JNIEXPORT jintArray JNICALL Java_org_intel_openvino_compatibility_Data_GetDims(JNIEnv *, jobject, jlong);
    JNIEXPORT void JNICALL Java_org_intel_openvino_compatibility_Data_delete(JNIEnv *, jobject, jlong);

    /* -------------------------------------- API 2.0 ------------------------------------------*/

    //ov
    JNIEXPORT void JNICALL Java_org_intel_openvino_Openvino_serialize(JNIEnv *, jobject, jlong, jstring, jstring);

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
    JNIEXPORT void JNICALL Java_org_intel_openvino_Dimension_delete(JNIEnv *, jobject, jlong);

    // ov::Output<ov::Node>
    JNIEXPORT jstring JNICALL Java_org_intel_openvino_Output_GetAnyName(JNIEnv *, jobject, jlong);
    JNIEXPORT jintArray JNICALL Java_org_intel_openvino_Output_GetShape(JNIEnv *, jobject, jlong);
    JNIEXPORT int JNICALL Java_org_intel_openvino_Output_GetElementType(JNIEnv *, jobject, jlong);
    JNIEXPORT void JNICALL Java_org_intel_openvino_Output_delete(JNIEnv *, jobject, jlong);

#ifdef __cplusplus
}
#endif
