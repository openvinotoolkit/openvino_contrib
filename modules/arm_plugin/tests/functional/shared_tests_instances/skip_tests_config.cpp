// Copyright (C) 2020-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <vector>
#include <string>

#include "functional_test_utils/skip_tests_config.hpp"

std::vector<std::string> disabledTestPatterns() {
    return {
        ".*reusableCPUStreamsExecutor.*",  //  TEST DO not support hetero case when all plugins use executors cache
        ".*ExecGraphTests.*", // Not implemented
        ".*Eltwise.*eltwiseOpType=Mod.*netPRC=FP16.*", // Failed
        ".*PreprocessTest.*", // Does not cover all needed cases
        ".*GRUCellTest.*decomposition0.*",  // GruCell should be decomposed
        ".*ConstantResultSubgraphTest.*inPrc=(I8|U64|I64|BOOL).*", // Unsupported precisions
        ".*TensorIteratorTest.*unrolling=0.*",  // Skip due to unsupported LSTM, RNN and GRU sequenses
        ".*CPUconfigItem=CPU_BIND_THREAD_YES.*", // unsupported configuration option
        ".*(GRU|LSTM|RNN)SequenceTest.*mode=CONVERT_TO_TI.*", // Nodes from sequence are not supported by plugin (TensorIterator.0)
#ifdef __ARM_FEATURE_FP16_VECTOR_ARITHMETIC
        ".*ActivationLayerTest.*CompareWithRefs/Tan_.*netPRC=FP16.*" // Failed (a small input change leads to a large output change)
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
#ifdef __arm__
        // Sporadic hanges on linux-debian_9_arm runner (armv7l) 72140
        ".*canStartSeveralAsyncInsideCompletionCallbackWithSafeDtor.*AUTO.*",
#endif
    };
}
