// Copyright (C) 2020-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <ngraph/pass/graph_rewrite.hpp>
#include <ngraph/variant.hpp>

namespace ArmPlugin {
namespace pass {

struct ConvertQuantize: public ngraph::pass::MatcherPass {
    ConvertQuantize();
};

struct FakeQuantizeFusionBase: public ngraph::pass::MatcherPass {
    template <class Node>
    void registerMatcher(const std::string& name, bool withActivation = false);
};

struct ConvFakeQuantizeFusion: public FakeQuantizeFusionBase {
    ConvFakeQuantizeFusion();
};

struct GroupConvFakeQuantizeFusion: public FakeQuantizeFusionBase {
    GroupConvFakeQuantizeFusion();
};

struct ConvActivationFakeQuantizeFusion: public FakeQuantizeFusionBase {
    ConvActivationFakeQuantizeFusion();
};

struct GroupConvActivationFakeQuantizeFusion: public FakeQuantizeFusionBase {
    GroupConvActivationFakeQuantizeFusion();
};

struct MatMulFakeQuantizeFusion: public FakeQuantizeFusionBase {
    MatMulFakeQuantizeFusion();
};

struct AvgPoolFakeQuantizeFusion: public FakeQuantizeFusionBase {
    AvgPoolFakeQuantizeFusion();
};

struct ReduceMeanFakeQuantizeFusion: public FakeQuantizeFusionBase {
    ReduceMeanFakeQuantizeFusion();
};

struct QuantizeFusion: public ngraph::pass::GraphRewrite {
    QuantizeFusion() {
        add_matcher<AvgPoolFakeQuantizeFusion>();
        add_matcher<ReduceMeanFakeQuantizeFusion>();
        add_matcher<ConvFakeQuantizeFusion>();
        add_matcher<GroupConvFakeQuantizeFusion>();
        add_matcher<ConvActivationFakeQuantizeFusion>();
        add_matcher<GroupConvActivationFakeQuantizeFusion>();
        add_matcher<MatMulFakeQuantizeFusion>();
    }
};

struct PropogateQuantizationInfo: public ngraph::pass::FunctionPass {
    bool run_on_function(std::shared_ptr<ngraph::Function> f) override;
};

struct DeqMulAddToArmDequantizeConvert: public ngraph::pass::MatcherPass {
    DeqMulAddToArmDequantizeConvert();
};

struct DeqMulToArmDequantizeConvert: public ngraph::pass::MatcherPass {
    DeqMulToArmDequantizeConvert();
};

struct AddArmDequantizeOnInputsBase: public ngraph::pass::MatcherPass {
    template <class Node>
    void registerMatcher(const std::string& name);
};

struct AddArmDequantizeOnInputsConv: public AddArmDequantizeOnInputsBase {
    AddArmDequantizeOnInputsConv();
};

struct AddArmDequantizeOnInputsGroupConv: public AddArmDequantizeOnInputsBase {
    AddArmDequantizeOnInputsGroupConv();
};

struct AddArmDequantizeOnInputsAdd: public AddArmDequantizeOnInputsBase {
    AddArmDequantizeOnInputsAdd();
};

struct AddArmDequantizeOnInputsSubtract: public AddArmDequantizeOnInputsBase {
    AddArmDequantizeOnInputsSubtract();
};

struct AddArmDequantizeOnInputs: public ngraph::pass::GraphRewrite {
    AddArmDequantizeOnInputs() {
        add_matcher<AddArmDequantizeOnInputsConv>();
        add_matcher<AddArmDequantizeOnInputsGroupConv>();
        add_matcher<AddArmDequantizeOnInputsAdd>();
        add_matcher<AddArmDequantizeOnInputsSubtract>();
    }
};

struct ConvertBiasToI32: public ngraph::pass::MatcherPass {
    ConvertBiasToI32();
};


struct DetectMaybeQuantized: public ngraph::pass::FunctionPass {
    bool run_on_function(std::shared_ptr<ngraph::Function> f) override;
};

}  // namespace pass
}  // namespace ArmPlugin

namespace ngraph {
template <>
struct NGRAPH_API VariantWrapper<ngraph::element::Type> : public VariantImpl<ngraph::element::Type> {
    NGRAPH_RTTI_DECLARATION;
    VariantWrapper(const ngraph::element::Type& value) : VariantImpl<ngraph::element::Type>{value} {}
    ~VariantWrapper() override;
};
}  // namespace ngraph
