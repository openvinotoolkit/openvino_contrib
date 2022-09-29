# Copyright (C) 2020-2022 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

get_filename_component(sample_name "${CMAKE_CURRENT_SOURCE_DIR}" NAME)

ie_add_java_sample(NAME ${sample_name} OPENCV_DEPENDENCIES)
