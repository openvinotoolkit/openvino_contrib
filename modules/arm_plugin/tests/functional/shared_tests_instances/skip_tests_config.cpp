// Copyright (C) 2020-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <vector>
#include <string>

#include "functional_test_utils/skip_tests_config.hpp"

std::vector<std::string> disabledTestPatterns() {
    return {
        ".*(r|R)eusableCPUStreamsExecutor.*",  //  TEST DO not support hetero case when all plugins use executors cache
        ".*ExecGraphTests.*", // Not implemented
        ".*EltwiseLayerTest.*eltwiseOpType=Mod.*", // Failed
        ".*PreprocessTest.*", // Does not cover all needed cases
        ".*CPUconfigItem=CPU_BIND_THREAD_YES.*", // unsupported configuration option
        ".*(GRU|LSTM|RNN)SequenceTest.*mode=CONVERT_TO_TI.*", // Nodes from sequence are not supported by plugin (TensorIterator.0)
#ifdef __ARM_FEATURE_FP16_VECTOR_ARITHMETIC
        ".*ActivationLayerTest.*CompareWithRefs/Tan_.*netPRC=FP16.*", // Failed (a small input change leads to a large output change)
        ".*PermConvPermConcat.*CompareWithRefs.*1.1.7.32.*1.5.*_netPRC=FP16.*", // accuracy differnce error = 0.001
        ".*MatmulSqueezeAddTest.CompareWithRefImpl.*1.512.*netPRC=FP16.*", // Sporadic hangs
#endif
        // need to implement Export / Import
        ".*IEClassImportExportTestP.*",
        ".*Multi_BehaviorTests/InferRequestTests.canRun3SyncRequestsConsistentlyFromThreads.*", // Sporadic hangs,
        // CVS-58963: Not implemented yet
        ".*InferRequestIOBBlobTest.*OutOfFirstOutIsInputForSecondNetwork.*",
        // Unexpected behavior
        ".*(Hetero|Multi).*InferRequestCallbackTests.*ReturnResultNotReadyFromWaitInAsyncModeForTooSmallTimeout.*",
        // Not implemented
        ".*Behavior.*ExecutableNetworkBaseTest.*(canSetConfigToExecNet|canSetConfigToExecNetAndCheckConfigAndCheck).*",
        ".*Behavior.*ExecutableNetworkBaseTest.*CanCreateTwoExeNetworksAndCheckFunction.*",
        ".*Behavior.*ExecutableNetworkBaseTest.*(CheckExecGraphInfoBeforeExecution|CheckExecGraphInfoAfterExecution).*",
        ".*Behavior.*ExecutableNetworkBaseTest.*canExport.*",
        ".*Behavior.*ExecutableNetworkBaseTest.*canSetConfigToExecNetWithIncorrectConfig.*",
        ".*Multi.*BehaviorTests.*ExecutableNetworkBaseTest.*checkGetExecGraphInfoIsNotNullptr.*",
        ".*(Auto|Multi).*Behavior.*ExecutableNetworkBaseTest.*CheckExecGraphInfoSerialization.*",
        ".*ExclusiveAsyncRequest.*", // Unsupported config test
        // Failed according to accuracy
        R"(.*SoftMaxLayerTest.*CompareWithRefs.*f16.*undefined.*undefined.*\(1.3.10.10\).*Axis=2.*)",
        ".*ExecutableNetwork.*CanSetConfig.*", // Wont be supported
        ".*checkGetExecGraphInfoIsNotNullptr.*(AUTO|HETERO|MULTI).*", // Dose not supported in OpenVINO
        ".*OVInferenceChaining.*Dynamic.*", // Dynamic shape is not supported
        ".*ReturnResultNotReadyFromWaitInAsyncModeForTooSmallTimeout.*", // Unsupported topology
        ".*NmsLayerTest.*CompareWithRefs.*numBatches=3.*numBoxes=100.*numClasses=5.*paramsPrec=FP32.*maxBoxPrec=I32.*thrPrec=FP32.*",
        ".*OVClassNetworkTestP.*QueryNetworkMultiThrows.*",
        ".*OVClassNetworkTestP.*LoadNetworkMultiWithoutSettingDevicePrioritiesThrows.*",
        R"(.*OVClassQueryNetworkTest.*DeviceID.*)",
        R"(.*OVClassLoadNetworkTest.*DeviceID.*)",
        R"(.*OVClassLoadNetworkTest.*(MULTIwithHETERO|HETEROwithMULTI|MULTIwithAUTO)NoThrow.*)",
        R"(.*OVClassLoadNetworkTest.*QueryNetwork(MULTIWithHETERO|HETEROWithMULTI)NoThrow_V10.*)",
        // Problem with interface
#ifndef __ARM_FEATURE_FP16_VECTOR_ARITHMETIC
        ".*ConversionLayerTest.*inputPRC=FP16_targetPRC=I8.*",
#endif
        ".*ConversionLayerTest.*inputPRC=FP16_targetPRC=U32.*", // PR 14292 breaks Convert and ConvertLike tests
        ".*ConversionLayerTest.*inputPRC=FP32_targetPRC=U32.*",
        ".*ConversionLayerTest.*inputPRC=FP32_targetPRC=I8.*",
#ifdef __arm__
        // Sporadic hanges on linux-debian_9_arm runner (armv7l) 72140
        ".*canStartSeveralAsyncInsideCompletionCallbackWithSafeDtor.*AUTO.*",
#endif
    };
}
