// Copyright (C) 2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "cuda_eltwise.hpp"

#include <fmt/format.h>
#include <ie_blob.h>

#include <algorithm>
#include <common_test_utils/common_utils.hpp>
#include <error.hpp>
#include <functional_test_utils/blob_utils.hpp>
#include <functional_test_utils/precision_utils.hpp>
#include <ie_precision.hpp>
#include <map>
#include <memory>
#include <ngraph/node.hpp>
#include <ngraph/op/divide.hpp>
#include <ngraph/op/parameter.hpp>
#include <ngraph/type/bfloat16.hpp>
#include <ngraph/type/float16.hpp>
#include <ngraph_functions/builders.hpp>
#include <ngraph_functions/utils/ngraph_helpers.hpp>
#include <sstream>
#include <vector>

namespace LayerTestsDefinitions {

namespace {

template <InferenceEngine::Precision::ePrecision PRC>
void replace(InferenceEngine::Blob::Ptr& blob, float old_value, float new_value) {
    using DataType = typename InferenceEngine::PrecisionTrait<PRC>::value_type;
    DataType* raw_ptr = blob->buffer().as<DataType*>();
    for (size_t i = 0; i < blob->size(); ++i) {
        if constexpr (PRC == InferenceEngine::Precision::FP16) {
            const auto value = ngraph::float16::from_bits(raw_ptr[i]);
            if (value == static_cast<ngraph::float16>(old_value)) {
                raw_ptr[i] = static_cast<DataType>(ngraph::float16(new_value).to_bits());
            }
        } else if constexpr (PRC == InferenceEngine::Precision::BF16) {
            const auto value = ngraph::bfloat16::from_bits(raw_ptr[i]);
            if (value == static_cast<ngraph::bfloat16>(old_value)) {
                raw_ptr[i] = static_cast<DataType>(ngraph::bfloat16(new_value).to_bits());
            }
        } else {
            if (raw_ptr[i] == old_value) {
                raw_ptr[i] = new_value;
            }
        }
    }
}

void replace(InferenceEngine::Blob::Ptr& blob, InferenceEngine::Precision precision, float old_value, float new_value) {
    switch (precision) {
        case InferenceEngine::Precision::FP32:
            replace<InferenceEngine::Precision::FP32>(blob, old_value, new_value);
            break;
        case InferenceEngine::Precision::FP16:
            replace<InferenceEngine::Precision::FP16>(blob, old_value, new_value);
            break;
        case InferenceEngine::Precision::BF16:
            replace<InferenceEngine::Precision::BF16>(blob, old_value, new_value);
            break;
        case InferenceEngine::Precision::FP64:
            replace<InferenceEngine::Precision::FP64>(blob, old_value, new_value);
            break;
        case InferenceEngine::Precision::I16:
            replace<InferenceEngine::Precision::I16>(blob, old_value, new_value);
            break;
        case InferenceEngine::Precision::U8:
            replace<InferenceEngine::Precision::U8>(blob, old_value, new_value);
            break;
        case InferenceEngine::Precision::I8:
            replace<InferenceEngine::Precision::I8>(blob, old_value, new_value);
            break;
        case InferenceEngine::Precision::U16:
            replace<InferenceEngine::Precision::U16>(blob, old_value, new_value);
            break;
        case InferenceEngine::Precision::I32:
            replace<InferenceEngine::Precision::I32>(blob, old_value, new_value);
            break;
        case InferenceEngine::Precision::U32:
            replace<InferenceEngine::Precision::U32>(blob, old_value, new_value);
            break;
        case InferenceEngine::Precision::I64:
            replace<InferenceEngine::Precision::I64>(blob, old_value, new_value);
            break;
        case InferenceEngine::Precision::U64:
            replace<InferenceEngine::Precision::U64>(blob, old_value, new_value);
            break;
        default:
            CUDAPlugin::throwIEException(fmt::format("replace(): Unsupported type: {}", precision.name()));
    }
}

}  // namespace

std::string CudaEltwiseLayerTest::getTestCaseName(testing::TestParamInfo<CudaEltwiseTestParams> obj) {
    EltwiseTestParams ew_params;
    OperationMode mode;
    std::tie(ew_params, mode) = obj.param;

    std::ostringstream result;
    result << EltwiseLayerTest::getTestCaseName({ew_params, obj.index}) << "_";
    result << "OperationMode=" << (mode == OperationMode::PYTHON_DIVIDE ? "PYTHON_DIVIDE" : "NORMAL");
    return result.str();
}

InferenceEngine::Blob::Ptr CudaEltwiseLayerTest::GenerateInput(const InferenceEngine::InputInfo& info) const {
    const auto ew_params = std::get<0>(this->GetParam());
    const auto op_type = std::get<1>(ew_params);
    const auto precision = info.getPrecision();
    const auto is_float = precision.is_float();
    switch (op_type) {
        case ngraph::helpers::EltwiseTypes::POWER:
            return is_float ? FuncTestUtils::createAndFillBlob(info.getTensorDesc(), 2, 2, 128)
                            : FuncTestUtils::createAndFillBlob(info.getTensorDesc(), 4, 2);
        case ngraph::helpers::EltwiseTypes::DIVIDE:
        case ngraph::helpers::EltwiseTypes::MOD: {
            auto blob = FuncTestUtils::createAndFillBlob(info.getTensorDesc(), range, start_from, resolution, seed);
            if (!is_float && info.name() == secondary_input_name) {
                replace(blob, precision, 0, 1);
            }
            return blob;
        }
        case ngraph::helpers::EltwiseTypes::ERF:
            return FuncTestUtils::createAndFillBlob(info.getTensorDesc(), 6, -3);
        default:
            return FuncTestUtils::createAndFillBlob(info.getTensorDesc(), range, start_from, resolution, seed);
    }
}

void CudaEltwiseLayerTest::SetUp() {
    this->to_check_nans = true;

    std::vector<std::vector<size_t>> input_shapes;
    InferenceEngine::Precision net_precision;
    ngraph::helpers::InputLayerType secondary_input_type;
    CommonTestUtils::OpType op_type;
    ngraph::helpers::EltwiseTypes eltwise_type;
    std::map<std::string, std::string> additional_config;
    const EltwiseTestParams ew_params = std::get<0>(this->GetParam());
    const OperationMode mode = std::get<1>(this->GetParam());
    std::tie(input_shapes,
             eltwise_type,
             secondary_input_type,
             op_type,
             net_precision,
             inPrc,
             outPrc,
             inLayout,
             targetDevice,
             additional_config) = ew_params;
    const auto ng_precision = FuncTestUtils::PrecisionUtils::convertIE2nGraphPrc(net_precision);

    std::vector<size_t> input_shape1, input_shape2;
    if (input_shapes.size() == 1) {
        input_shape1 = input_shape2 = input_shapes.front();
    } else if (input_shapes.size() == 2) {
        input_shape1 = input_shapes.front();
        input_shape2 = input_shapes.back();
    } else {
        CUDAPlugin::throwIEException("Incorrect number of input shapes");
    }

    configuration.insert(additional_config.begin(), additional_config.end());
    auto input = ngraph::builder::makeParams(ng_precision, {input_shape1});

    std::vector<size_t> shape_input_secondary;
    switch (op_type) {
        case CommonTestUtils::OpType::SCALAR:
            shape_input_secondary = std::vector<size_t>({1});
            break;
        case CommonTestUtils::OpType::VECTOR:
            shape_input_secondary = input_shape2;
            break;
        default:
            FAIL() << "Unsupported Secondary operation type";
    }

    constexpr bool is_random = true;
    std::shared_ptr<ngraph::Node> secondary_input;
    auto data = NGraphFunctions::Utils::generateVector<ngraph::element::Type_t::i32>(
        ngraph::shape_size(shape_input_secondary), up_to, start_from, seed);
    if (secondary_input_type == ngraph::helpers::InputLayerType::CONSTANT) {
        switch (eltwise_type) {
            case ngraph::helpers::EltwiseTypes::DIVIDE:
            case ngraph::helpers::EltwiseTypes::MOD: {
                if (ng_precision.is_integral()) {
                    std::replace(data.begin(), data.end(), 0, 1);
                }
                secondary_input = ngraph::builder::makeConstant(ng_precision, shape_input_secondary, data);
                break;
            }
            case ngraph::helpers::EltwiseTypes::POWER:
                // to avoid floating point overflow on some platforms, let's fill the constant with small numbers.
                secondary_input =
                    ngraph::builder::makeConstant<float>(ng_precision, shape_input_secondary, {}, is_random, 3);
                break;
            default:
                secondary_input = ngraph::builder::makeConstant(ng_precision, shape_input_secondary, data);
        }
    } else {
        secondary_input = ngraph::builder::makeInputLayer(ng_precision, secondary_input_type, shape_input_secondary);
        input.push_back(std::dynamic_pointer_cast<ngraph::op::v0::Parameter>(secondary_input));
    }
    secondary_input_name = secondary_input->get_friendly_name();

    const bool is_python_divide = mode == OperationMode::PYTHON_DIVIDE;
    auto eltwise = eltwise_type == ngraph::helpers::EltwiseTypes::DIVIDE
                       ? std::make_shared<ngraph::op::v1::Divide>(input[0], secondary_input, is_python_divide)
                       : ngraph::builder::makeEltwise(input[0], secondary_input, eltwise_type);
    function = std::make_shared<ngraph::Function>(eltwise, input, "Eltwise");
}

TEST_P(CudaEltwiseLayerTest, CompareWithRefs) {
    SKIP_IF_CURRENT_TEST_IS_DISABLED()
    Run();
}

}  // namespace LayerTestsDefinitions
