// Copyright (C) 2020-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "functional_test_utils/skip_tests_config.hpp"

#include <cuda/runtime.hpp>
#include <string>
#include <vector>

std::vector<std::string> disabledTestPatterns() {
    std::vector<std::string> retVector{
        // CVS-55937
        R"(.*SplitLayerTest.*numSplits=30.*)",
        // CVS-51758
        R"(.*InferRequestPreprocessConversionTest.*oLT=(NHWC|NCHW).*)",
        R"(.*InferRequestPreprocessDynamicallyInSetBlobTest.*oPRC=0.*oLT=1.*)",
        // Not Implemented
        R"(.*Behavior.*ExecutableNetworkBaseTest.*(CheckExecGraphInfoBeforeExecution|CheckExecGraphInfoAfterExecution|CheckExecGraphInfoSerialization).*)",
        R"(.*smoke_BehaviorTests.*OVExecGraphImportExportTest.ieImportExportedFunction.*)",
        R"(.*CachingSupportCase*.*ReadConcatSplitAssign.*)",
        // 101751, 101746, 101747, 101748, 101755
        R"(.*(d|D)ynamic*.*)",
        R"(.*smoke_AutoBatch_BehaviorTests*.*)",
        R"(.*smoke_Auto_BehaviorTests*.*)",
        R"(.*smoke_Multi_BehaviorTests*.*)",
        R"(.*HETERO(W|w)ithMULTI*.*)",
        // Plugin version was changed to ov::Version
        R"(.*VersionTest.*pluginCurrentVersionIsCorrect.*)",
        // New plugin API doesn't support changes of pre-processing
        R"(.*InferRequestPreprocessTest.*SetPreProcessToInputInfo.*)",
        R"(.*InferRequestPreprocessTest.*SetPreProcessToInferRequest.*)",
        // New plugin work with tensors, so it means that blob in old API can have different pointers
        R"(.*InferRequestIOBBlobTest.*secondCallGetInputDoNotReAllocateData.*)",
        R"(.*InferRequestIOBBlobTest.*secondCallGetOutputDoNotReAllocateData.*)",
        R"(.*InferRequestIOBBlobTest.*secondCallGetInputAfterInferSync.*)",
        R"(.*InferRequestIOBBlobTest.*secondCallGetOutputAfterInferSync.*)",
        // Old API cannot deallocate tensor
        R"(.*InferRequestIOBBlobTest.*canProcessDeallocatedOutputBlobAfterGetAndSetBlob.*)",
        // 119703
        R"(.*smoke_GroupConvolutionBias(Add|AddAdd)_2D_ExplicitPaddingSymmetric2.*FP16*.*)",
        // Issue: 128924
        R"(.*smoke_OVClassNetworkTestP/OVClassModelTestP.ImportModelWithNullContextThrows.*)",
    };

#ifdef _WIN32
    // CVS-63989
    retVector.emplace_back(R"(.*ReferenceSigmoidLayerTest.*u64.*)");
    // CVS-64054
    retVector.emplace_back(R"(.*ReferenceTopKTest.*topk_max_sort_none)");
    retVector.emplace_back(R"(.*ReferenceTopKTest.*topk_min_sort_none)");
#endif

    if (!CUDA::isHalfSupported(CUDA::Device{})) {
        retVector.emplace_back(
            R"(.*OVExecGraphImportExportTest.*importExportedFunctionParameterResultOnly.*targetDevice=NVIDIA_elementType=f16.*)");
        retVector.emplace_back(
            R"(.*OVExecGraphImportExportTest.*importExportedIENetworkParameterResultOnly.*targetDevice=NVIDIA_elementType=f16.*)");
    }

    return retVector;
}
