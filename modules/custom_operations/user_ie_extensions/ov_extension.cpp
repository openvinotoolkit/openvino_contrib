// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <openvino/core/extension.hpp>
#include <openvino/core/op_extension.hpp>
#include <openvino/frontend/extension.hpp>
#include <openvino/frontend/node_context.hpp>

#ifdef calculate_grid
#    include "calculate_grid.hpp"
#    define CALCULATE_GRID_EXT                                                                          \
            std::make_shared<ov::OpExtension<TemplateExtension::CalculateGrid>>(),                      \
            std::make_shared<ov::frontend::OpExtension<TemplateExtension::CalculateGrid>>(),
#else
#    define CALCULATE_GRID_EXT
#endif

#ifdef complex_mul
#    include "complex_mul.hpp"
#    define COMPLEX_MUL_EXT                                                                            \
            std::make_shared<ov::OpExtension<TemplateExtension::ComplexMultiplication>>(),             \
            std::make_shared<ov::frontend::OpExtension<TemplateExtension::ComplexMultiplication>>(),
#else
#    define COMPLEX_MUL_EXT
#endif

#ifdef fft
#    include "fft.hpp"
#    define FFT_EXT                                                                                    \
            std::make_shared<ov::OpExtension<TemplateExtension::FFT>>(),                               \
            std::make_shared<ov::frontend::OpExtension<TemplateExtension::FFT>>(),
#else
#    define FFT_EXT
#endif

#ifdef sparse_conv_transpose
#    include "sparse_conv_transpose.hpp"
#    define S_CONV_TRANSPOSE_EXT                                                                      \
            std::make_shared<ov::OpExtension<TemplateExtension::SparseConvTranspose>>(),              \
            std::make_shared<ov::frontend::OpExtension<TemplateExtension::SparseConvTranspose>>(),
#else
#    define S_CONV_TRANSPOSE_EXT
#endif

#ifdef sparse_conv
#    include "sparse_conv.hpp"
#    define S_CONV_EXT                                                                                \
            std::make_shared<ov::OpExtension<TemplateExtension::SparseConv>>(),                       \
            std::make_shared<ov::frontend::OpExtension<TemplateExtension::SparseConv>>(),
#else
#    define S_CONV_EXT
#endif

#ifdef sentence_piece
#    include "sentence_piece/sentence_piece.hpp"
#    define SENTENSE_PIECE_EXT                                                                                              \
            std::make_shared<ov::OpExtension<TemplateExtension::SentencepieceTokenizer>>(),                                 \
            std::make_shared<ov::frontend::ConversionExtension>("SentencepieceOp", translate_sentencepiece_op),             \
            std::make_shared<ov::frontend::ConversionExtension>("RaggedTensorToSparse", translate_sentencepiece_tokenizer),
#else
#    define SENTENSE_PIECE_EXT
#endif

OPENVINO_CREATE_EXTENSIONS(std::vector<ov::Extension::Ptr>(
    {
        CALCULATE_GRID_EXT
        FFT_EXT
        S_CONV_TRANSPOSE_EXT
        S_CONV_EXT
        COMPLEX_MUL_EXT
        SENTENSE_PIECE_EXT
    }));
