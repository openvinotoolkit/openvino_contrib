# Copyright (C) 2020-2022 Intel Corporation
#
# SPDX-License-Identifier: Apache-2.0
#

set(TARGET_NAME armFuncTests)

addIeTargetTest(
        NAME ${TARGET_NAME}
        ROOT ${CMAKE_CURRENT_SOURCE_DIR}
        DEPENDENCIES
                openvino_arm_cpu_plugin
        LINK_LIBRARIES
                arm_compute::arm_compute
                arm_compute::arm_compute_core
                arm_compute::half
                openvino_arm_cpu_transformations
                openvino_arm_cpu_opset
                openvino::ngraph_reference
                openvino::funcSharedTests
                openvino::lptNgraphFunctions
        INCLUDES
        "${IE_MAIN_ARM_PLUGIN_SOURCE_DIR}/src/transformations"
        "${IE_MAIN_ARM_PLUGIN_SOURCE_DIR}/src/opset"
        ADD_CPPLINT
        LABELS
                ARM
)
