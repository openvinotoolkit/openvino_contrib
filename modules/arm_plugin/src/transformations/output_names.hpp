// Copyright (C) 2020-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <string>

#include <openvino/core/node.hpp>

std::string create_ie_output_name(const ngraph::Output<const ngraph::Node>& output);

std::string create_ie_output_name(const ngraph::Output<ngraph::Node>& output);
