// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <openvino/frontend/node_context.hpp>

ov::OutputVector translate_sentencepiece_op(const ov::frontend::NodeContext& node);
ov::frontend::NamedOutputVector translate_sentencepiece_tokenizer(const ov::frontend::NodeContext& node);
ov::OutputVector translate_case_fold_utf8(const ov::frontend::NodeContext& node);
ov::OutputVector translate_normalize_utf8(const ov::frontend::NodeContext& node);
ov::OutputVector translate_static_regex_replace(const ov::frontend::NodeContext& node);
ov::OutputVector translate_regex_split_with_offsets(const ov::frontend::NodeContext& node);
ov::OutputVector translate_wordpiece_tokenize_with_offsets(const ov::frontend::NodeContext& node);
ov::OutputVector translate_lookup_table_find_v2(const ov::frontend::NodeContext& node);
ov::OutputVector translate_reshape(const ov::frontend::NodeContext& node);
ov::OutputVector translate_const(const ov::frontend::NodeContext& node);
