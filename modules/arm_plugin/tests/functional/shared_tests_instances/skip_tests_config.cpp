// Copyright (C) 2020-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <vector>
#include <string>

#include "functional_test_utils/skip_tests_config.hpp"

std::vector<std::string> disabledTestPatterns() {
    return {
        ".*reusableCPUStreamsExecutor.*",  //  TEST DO not support hetero case when all plugins use executors cache
        ".*ExecGraphTests.*", // Not implemented
        R"(.*Eltwise.*eltwiseOpType=Mod.*netPRC=FP16.*)", // Failed
        ".*PreprocessTest.*", // Does not cover all needed cases
        ".*GRUCellTest.*decomposition0.*",  // GruCell should be decomposed
        R"(.*ConstantResultSubgraphTest.*inPrc=(I8|U64|I64|BOOL).*)", // Unsupported precisions
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
        R"(.*InferRequestIOBBlobTest.*OutOfFirstOutIsInputForSecondNetwork.*)",
        // Unexpected behavior
        R"(.*(Hetero|Multi).*InferRequestCallbackTests.*ReturnResultNotReadyFromWaitInAsyncModeForTooSmallTimeout.*)",
        R"(.*ExclusiveAsyncRequest.*)" // Unsupported config test
    };
}