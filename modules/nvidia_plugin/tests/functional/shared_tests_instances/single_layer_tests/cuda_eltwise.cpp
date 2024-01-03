// Copyright (C) 2022-2023 Intel Corporation
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
#include <ov_models/utils/ov_helpers.hpp>
#include <sstream>
#include <vector>

#include "common_test_utils/node_builders/eltwise.hpp"

namespace LayerTestsDefinitions {

namespace {
using ov::test::utils::EltwiseTypes;
using ov::test::utils::InputLayerType;

template <InferenceEngine::Precision::ePrecision PRC>
void replace(InferenceEngine::Blob::Ptr& blob, float old_value, float new_value, bool is_integer) {
    using DataType = typename InferenceEngine::PrecisionTrait<PRC>::value_type;
    DataType* raw_ptr = blob->buffer().as<DataType*>();
    for (size_t i = 0; i < blob->size(); ++i) {
        if constexpr (PRC == InferenceEngine::Precision::FP16) {
            const auto value = ngraph::float16::from_bits(raw_ptr[i]);
            const bool should_replace =
                (is_integer && static_cast<int>(static_cast<float>(value)) == static_cast<int>(old_value)) ||
                value == static_cast<ngraph::float16>(old_value);
            if (should_replace) {
                raw_ptr[i] = static_cast<DataType>(ngraph::float16(new_value).to_bits());
            }
        } else if constexpr (PRC == InferenceEngine::Precision::BF16) {
            const auto value = ngraph::bfloat16::from_bits(raw_ptr[i]);
            const bool should_replace =
                (is_integer && static_cast<int>(static_cast<float>(value)) == static_cast<int>(old_value)) ||
                value == static_cast<ngraph::bfloat16>(old_value);
            if (should_replace) {
                raw_ptr[i] = static_cast<DataType>(ngraph::bfloat16(new_value).to_bits());
            }
        } else {
            const bool should_replace =
                (is_integer && static_cast<int>(raw_ptr[i]) == static_cast<int>(old_value)) || raw_ptr[i] == old_value;
            if (should_replace) {
                raw_ptr[i] = new_value;
            }
        }
    }
}

void replace(InferenceEngine::Blob::Ptr& blob,
             InferenceEngine::Precision precision,
             float old_value,
             float new_value,
             bool is_integer) {
    switch (precision) {
        case InferenceEngine::Precision::FP32:
            replace<InferenceEngine::Precision::FP32>(blob, old_value, new_value, is_integer);
            break;
        case InferenceEngine::Precision::FP16:
            replace<InferenceEngine::Precision::FP16>(blob, old_value, new_value, is_integer);
            break;
        case InferenceEngine::Precision::BF16:
            replace<InferenceEngine::Precision::BF16>(blob, old_value, new_value, is_integer);
            break;
        case InferenceEngine::Precision::FP64:
            replace<InferenceEngine::Precision::FP64>(blob, old_value, new_value, is_integer);
            break;
        case InferenceEngine::Precision::I16:
            replace<InferenceEngine::Precision::I16>(blob, old_value, new_value, is_integer);
            break;
        case InferenceEngine::Precision::U8:
            replace<InferenceEngine::Precision::U8>(blob, old_value, new_value, is_integer);
            break;
        case InferenceEngine::Precision::I8:
            replace<InferenceEngine::Precision::I8>(blob, old_value, new_value, is_integer);
            break;
        case InferenceEngine::Precision::U16:
            replace<InferenceEngine::Precision::U16>(blob, old_value, new_value, is_integer);
            break;
        case InferenceEngine::Precision::I32:
            replace<InferenceEngine::Precision::I32>(blob, old_value, new_value, is_integer);
            break;
        case InferenceEngine::Precision::U32:
            replace<InferenceEngine::Precision::U32>(blob, old_value, new_value, is_integer);
            break;
        case InferenceEngine::Precision::I64:
            replace<InferenceEngine::Precision::I64>(blob, old_value, new_value, is_integer);
            break;
        case InferenceEngine::Precision::U64:
            replace<InferenceEngine::Precision::U64>(blob, old_value, new_value, is_integer);
            break;
        default:
            ov::nvidia_gpu::throw_ov_exception(fmt::format("replace(): Unsupported type: {}", precision.name()));
    }
}

InferenceEngine::Precision convertTestElementTypeToNGraphPrecision(const ov::test::ElementType& value) {
    switch (value) {
        case ov::element::Type_t::undefined:
            return InferenceEngine::Precision::UNSPECIFIED;
        case ov::element::Type_t::boolean:
            return InferenceEngine::Precision::BOOL;
        case ov::element::Type_t::bf16:
            return InferenceEngine::Precision::BF16;
        case ov::element::Type_t::f16:
            return InferenceEngine::Precision::FP16;
        case ov::element::Type_t::f32:
            return InferenceEngine::Precision::FP32;
        case ov::element::Type_t::f64:
            return InferenceEngine::Precision::FP64;
        case ov::element::Type_t::i4:
            return InferenceEngine::Precision::I4;
        case ov::element::Type_t::i8:
            return InferenceEngine::Precision::I8;
        case ov::element::Type_t::i16:
            return InferenceEngine::Precision::I16;
        case ov::element::Type_t::i32:
            return InferenceEngine::Precision::I32;
        case ov::element::Type_t::i64:
            return InferenceEngine::Precision::I64;
        case ov::element::Type_t::u4:
            return InferenceEngine::Precision::U4;
        case ov::element::Type_t::u8:
            return InferenceEngine::Precision::U8;
        case ov::element::Type_t::u16:
            return InferenceEngine::Precision::U16;
        case ov::element::Type_t::u32:
            return InferenceEngine::Precision::U32;
        case ov::element::Type_t::u64:
            return InferenceEngine::Precision::U64;
        default:
            throw std::invalid_argument(
                fmt::format("Cannot convert ElementType = {} to InferenceEngine::Precision", value));
    }
};

}  // namespace

std::string CudaEltwiseLayerTest::getTestCaseName(testing::TestParamInfo<CudaEltwiseTestParams> obj) {
    ov::test::subgraph::EltwiseTestParams ew_params;
    OperationMode mode;
    std::tie(ew_params, mode) = obj.param;

    std::ostringstream result;
    result << ov::test::subgraph::EltwiseLayerTest::getTestCaseName({ew_params, obj.index}) << "_";
    result << "OperationMode=" << (mode == OperationMode::PYTHON_DIVIDE ? "PYTHON_DIVIDE" : "NORMAL");
    return result.str();
}

InferenceEngine::Blob::Ptr CudaEltwiseLayerTest::GenerateInput(const InferenceEngine::InputInfo& info) const {
    const auto ew_params = std::get<0>(this->GetParam());
    const auto op_type = std::get<1>(ew_params);
    const auto precision = info.getPrecision();
    const auto is_float = precision.is_float();
    switch (op_type) {
        case EltwiseTypes::POWER:
            return is_float ? FuncTestUtils::createAndFillBlob(info.getTensorDesc(), 2, 2, 128)
                            : FuncTestUtils::createAndFillBlob(info.getTensorDesc(), 4, 2);
        case EltwiseTypes::DIVIDE:
        case EltwiseTypes::MOD: {
            auto blob = FuncTestUtils::createAndFillBlob(info.getTensorDesc(), range, start_from, resolution, seed);
            if (!is_float && info.name() == secondary_input_name) {
                replace(blob, precision, 0, 1, true);
            }
            return blob;
        }
        case EltwiseTypes::ERF:
            return FuncTestUtils::createAndFillBlob(info.getTensorDesc(), 6, -3);
        default:
            return FuncTestUtils::createAndFillBlob(info.getTensorDesc(), range, start_from, resolution, seed);
    }
}

void CudaEltwiseLayerTest::SetUp() {
    PluginCache::get().reset();

    this->to_check_nans = true;

    std::vector<ov::test::InputShape> shapes;
    ov::test::ElementType netType;
    ov::test::ElementType in_prc;
    ov::test::ElementType out_prc;
    InputLayerType secondaryInputType;
    ov::test::utils::OpType opType;
    EltwiseTypes eltwiseType;
    ov::AnyMap additionalConfig;
    const ov::test::subgraph::EltwiseTestParams ew_params = std::get<0>(this->GetParam());
    const OperationMode mode = std::get<1>(this->GetParam());

    std::tie(shapes, eltwiseType, secondaryInputType, opType, netType, in_prc, out_prc, targetDevice, configuration) =
        ew_params;

    inPrc = convertTestElementTypeToNGraphPrecision(in_prc);
    outPrc = convertTestElementTypeToNGraphPrecision(out_prc);

    init_input_shapes(shapes);

    ov::ParameterVector parameters{std::make_shared<ov::op::v0::Parameter>(netType, inputDynamicShapes.front())};

    ov::PartialShape shape_input_secondary;
    switch (opType) {
        case ov::test::utils::OpType::SCALAR: {
            shape_input_secondary = {1};
            break;
        }
        case ov::test::utils::OpType::VECTOR:
            shape_input_secondary = inputDynamicShapes.back();
            break;
        default:
            FAIL() << "Unsupported Secondary operation type";
    }
    // To propagate shape_input_secondary just in static case because all shapes are defined in dynamic scenarion
    if (secondaryInputType == InputLayerType::PARAMETER) {
        transformInputShapesAccordingEltwise(shape_input_secondary);
    }

    std::shared_ptr<ngraph::Node> secondaryInput;
    if (secondaryInputType == InputLayerType::PARAMETER) {
        auto input = std::make_shared<ov::op::v0::Parameter>(netType, shape_input_secondary);
        secondaryInput = input;
        parameters.push_back(input);
    } else {
        constexpr bool is_random = true;
        ov::Shape shape = inputDynamicShapes.back().get_max_shape();
        auto data = NGraphFunctions::Utils::generateVector<ngraph::element::Type_t::f32>(
            ngraph::shape_size(shape), up_to, start_from, seed);
        switch (eltwiseType) {
            case EltwiseTypes::DIVIDE:
            case EltwiseTypes::MOD: {
                if (ov::element::Type{netType}.is_integral()) {
                    std::replace_if(
                        data.begin(),
                        data.end(),
                        [](const auto& value) { return static_cast<int>(static_cast<float>(value)) == 0; },
                        1);
                }
                secondaryInput = std::make_shared<ov::op::v0::Constant>(netType, shape, data);
                break;
            }
            case EltwiseTypes::POWER: {
                ov::Tensor random_tensor(netType, shape);
                ov::test::utils::fill_tensor_random(random_tensor, 3, -3);
                secondaryInput = std::make_shared<ov::op::v0::Constant>(random_tensor);
                break;
            }
            default: {
                secondaryInput = std::make_shared<ov::op::v0::Constant>(netType, shape, data);
            }
        }
    }

    parameters[0]->set_friendly_name("param0");
    secondaryInput->set_friendly_name("param1");

    secondary_input_name = secondaryInput->get_friendly_name();

    const bool is_python_divide = mode == OperationMode::PYTHON_DIVIDE;
    auto eltwise = eltwiseType == EltwiseTypes::DIVIDE
                       ? std::make_shared<ngraph::op::v1::Divide>(parameters[0], secondaryInput, is_python_divide)
                       : ov::test::utils::make_eltwise(parameters[0], secondaryInput, eltwiseType);
    function = std::make_shared<ngraph::Function>(eltwise, parameters, "Eltwise");
}

TEST_P(CudaEltwiseLayerTest, CompareWithRefs) {
    SKIP_IF_CURRENT_TEST_IS_DISABLED()
    Run();
}

}  // namespace LayerTestsDefinitions
