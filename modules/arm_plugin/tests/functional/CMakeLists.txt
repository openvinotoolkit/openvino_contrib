# Copyright (C) 2020-2021 Intel Corporation
#
# SPDX-License-Identifier: Apache-2.0
#

set(TARGET_NAME armFuncTests)

addIeTargetTest(
        NAME ${TARGET_NAME}
        ROOT ${CMAKE_CURRENT_SOURCE_DIR}
        DEPENDENCIES
                armPlugin IE::lptNgraphFunctions
        LINK_LIBRARIES
                IE::funcSharedTests IE::lptNgraphFunctions
        INCLUDES
            "${IE_MAIN_ARM_PLUGIN_SOURCE_DIR}/include"
        ADD_CPPLINT
        LABELS
                ARM
)
