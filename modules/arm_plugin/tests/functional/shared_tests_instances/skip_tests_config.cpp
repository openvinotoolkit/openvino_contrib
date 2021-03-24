// Copyright (C) 2020-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <vector>
#include <string>

#include "functional_test_utils/skip_tests_config.hpp"

std::vector<std::string> disabledTestPatterns() {
    return {
        ".*reusableCPUStreamsExecutor.*",  //  TEST DO not support hetero case when all plugins use executers cache
        ".*Multi.*canSetExclusiveAsyncRequests.*",  // Unsupported topology
        ".*Multi.*withoutExclusiveAsyncRequests.*",  // Unsupported topology
        ".*ExecGraphTests.*", // Not implemented
        R"(.*Eltwise.*eltwiseOpType=Mod.*netPRC=FP16.*)", // Failed
        ".*PreprocessTest.*", // Does not cover all needed cases
        ".*GRUCellTest.*decomposition0.*",  // GruCell should be decomposed
        R"(.*ConstantResultSubgraphTest.*inPrc=(I8|U64|I64|BOOL).*)", // Unsupported precisions
#ifdef __ARM_FEATURE_FP16_VECTOR_ARITHMETIC
        ".*ActivationLayerTest.*CompareWithRefs/Tan_.*netPRC=FP16.*" // Failed (a small input change leads to a large output change)
#endif
    };
}