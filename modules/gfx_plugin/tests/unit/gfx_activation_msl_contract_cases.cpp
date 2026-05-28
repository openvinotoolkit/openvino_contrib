// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "unit/gfx_activation_contract_cases.hpp"

#include <utility>

#include "openvino/op/abs.hpp"
#include "openvino/op/acos.hpp"
#include "openvino/op/acosh.hpp"
#include "openvino/op/asin.hpp"
#include "openvino/op/asinh.hpp"
#include "openvino/op/atan.hpp"
#include "openvino/op/atanh.hpp"
#include "openvino/op/ceiling.hpp"
#include "openvino/op/clamp.hpp"
#include "openvino/op/constant.hpp"
#include "openvino/op/cos.hpp"
#include "openvino/op/cosh.hpp"
#include "openvino/op/elu.hpp"
#include "openvino/op/erf.hpp"
#include "openvino/op/exp.hpp"
#include "openvino/op/floor.hpp"
#include "openvino/op/gelu.hpp"
#include "openvino/op/hsigmoid.hpp"
#include "openvino/op/hswish.hpp"
#include "openvino/op/log.hpp"
#include "openvino/op/mish.hpp"
#include "openvino/op/negative.hpp"
#include "openvino/op/parameter.hpp"
#include "openvino/op/relu.hpp"
#include "openvino/op/round.hpp"
#include "openvino/op/sigmoid.hpp"
#include "openvino/op/sign.hpp"
#include "openvino/op/sin.hpp"
#include "openvino/op/sinh.hpp"
#include "openvino/op/softplus.hpp"
#include "openvino/op/softsign.hpp"
#include "openvino/op/sqrt.hpp"
#include "openvino/op/swish.hpp"
#include "openvino/op/tan.hpp"
#include "openvino/op/tanh.hpp"

namespace ov {
namespace gfx_plugin {
namespace {

std::shared_ptr<ov::op::v0::Parameter> param(const ov::element::Type &type,
                                             ov::Shape shape) {
  return std::make_shared<ov::op::v0::Parameter>(type, std::move(shape));
}

} // namespace

std::vector<ActivationMslArtifactCase> activation_msl_artifact_cases() {
  return {
      {"Relu",
       [] {
         return std::make_shared<ov::op::v0::Relu>(
             param(ov::element::f32, ov::Shape{2, 3, 4}));
       },
       "max(x, 0.0f)"},
      {"Sigmoid",
       [] {
         return std::make_shared<ov::op::v0::Sigmoid>(
             param(ov::element::f32, ov::Shape{2, 3, 4}));
       },
       "1.0f / (1.0f + precise::exp(-x))"},
      {"Tanh",
       [] {
         return std::make_shared<ov::op::v0::Tanh>(
             param(ov::element::f32, ov::Shape{2, 3, 4}));
       },
       "tanh(clamp(x, -20.0f, 20.0f))"},
      {"Elu",
       [] {
         return std::make_shared<ov::op::v0::Elu>(
             param(ov::element::f32, ov::Shape{2, 3, 4}), 0.5);
       },
       "(exp(x) - 1.0f) * 0.500000"},
      {"Clamp",
       [] {
         return std::make_shared<ov::op::v0::Clamp>(
             param(ov::element::f32, ov::Shape{2, 3, 4}), -0.25, 0.75);
       },
       "clamp(x, -0.250000f, 0.750000f)"},
      {"GeluErf",
       [] {
         return std::make_shared<ov::op::v7::Gelu>(
             param(ov::element::f32, ov::Shape{2, 3, 4}),
             ov::op::GeluApproximationMode::ERF);
       },
       "erf(x * 0.70710678118f)"},
      {"GeluTanh",
       [] {
         return std::make_shared<ov::op::v7::Gelu>(
             param(ov::element::f32, ov::Shape{2, 3, 4}),
             ov::op::GeluApproximationMode::TANH);
       },
       "0.79788456f"},
      {"HSwish",
       [] {
         return std::make_shared<ov::op::v4::HSwish>(
             param(ov::element::f32, ov::Shape{2, 3, 4}));
       },
       "x * clamp(x + 3.0f, 0.0f, 6.0f) / 6.0f"},
      {"HSigmoid",
       [] {
         return std::make_shared<ov::op::v5::HSigmoid>(
             param(ov::element::f32, ov::Shape{2, 3, 4}));
       },
       "clamp(x + 3.0f, 0.0f, 6.0f) / 6.0f"},
      {"SoftPlus",
       [] {
         return std::make_shared<ov::op::v4::SoftPlus>(
             param(ov::element::f32, ov::Shape{2, 3, 4}));
       },
       "log(1.0f + exp(x))"},
      {"SwishDefaultBeta",
       [] {
         return std::make_shared<ov::op::v4::Swish>(
             param(ov::element::f32, ov::Shape{2, 3, 4}));
       },
       "1.000000f * x"},
      {"SwishStaticBeta",
       [] {
         const auto input = param(ov::element::f32, ov::Shape{2, 3, 4});
         const auto beta = ov::op::v0::Constant::create(ov::element::f32,
                                                        ov::Shape{}, {0.5f});
         return std::make_shared<ov::op::v4::Swish>(input, beta);
       },
       "0.500000f * x"},
      {"Mish",
       [] {
         return std::make_shared<ov::op::v4::Mish>(
             param(ov::element::f32, ov::Shape{2, 3, 4}));
       },
       "x * tanh(clamp(log(1.0f + exp(x)), -20.0f, 20.0f))"},
      {"SoftSign",
       [] {
         return std::make_shared<ov::op::v9::SoftSign>(
             param(ov::element::f32, ov::Shape{2, 3, 4}));
       },
       "x / (1.0f + fabs(x))"},
      {"Abs",
       [] {
         return std::make_shared<ov::op::v0::Abs>(
             param(ov::element::f32, ov::Shape{2, 3, 4}));
       },
       "fabs(x)"},
      {"Sign",
       [] {
         return std::make_shared<ov::op::v0::Sign>(
             param(ov::element::f32, ov::Shape{2, 3, 4}));
       },
       "x > 0.0f ? 1.0f"},
      {"Negative",
       [] {
         return std::make_shared<ov::op::v0::Negative>(
             param(ov::element::f32, ov::Shape{2, 3, 4}));
       },
       "-x"},
      {"Exp",
       [] {
         return std::make_shared<ov::op::v0::Exp>(
             param(ov::element::f32, ov::Shape{2, 3, 4}));
       },
       "exp(x)"},
      {"Log",
       [] {
         return std::make_shared<ov::op::v0::Log>(
             param(ov::element::f32, ov::Shape{2, 3, 4}));
       },
       "log(x)"},
      {"Sqrt",
       [] {
         return std::make_shared<ov::op::v0::Sqrt>(
             param(ov::element::f32, ov::Shape{2, 3, 4}));
       },
       "sqrt(x)"},
      {"Floor",
       [] {
         return std::make_shared<ov::op::v0::Floor>(
             param(ov::element::f32, ov::Shape{2, 3, 4}));
       },
       "floor(x)"},
      {"Ceiling",
       [] {
         return std::make_shared<ov::op::v0::Ceiling>(
             param(ov::element::f32, ov::Shape{2, 3, 4}));
       },
       "ceil(x)"},
      {"Sin",
       [] {
         return std::make_shared<ov::op::v0::Sin>(
             param(ov::element::f32, ov::Shape{2, 3, 4}));
       },
       "sin(x)"},
      {"Cos",
       [] {
         return std::make_shared<ov::op::v0::Cos>(
             param(ov::element::f32, ov::Shape{2, 3, 4}));
       },
       "cos(x)"},
      {"Tan",
       [] {
         return std::make_shared<ov::op::v0::Tan>(
             param(ov::element::f32, ov::Shape{2, 3, 4}));
       },
       "tan(x)"},
      {"Erf",
       [] {
         return std::make_shared<ov::op::v0::Erf>(
             param(ov::element::f32, ov::Shape{2, 3, 4}));
       },
       "erf(x)"},
      {"Asin",
       [] {
         return std::make_shared<ov::op::v0::Asin>(
             param(ov::element::f32, ov::Shape{2, 3, 4}));
       },
       "asin(x)"},
      {"Acos",
       [] {
         return std::make_shared<ov::op::v0::Acos>(
             param(ov::element::f32, ov::Shape{2, 3, 4}));
       },
       "acos(x)"},
      {"Atan",
       [] {
         return std::make_shared<ov::op::v0::Atan>(
             param(ov::element::f32, ov::Shape{2, 3, 4}));
       },
       "atan(x)"},
      {"Asinh",
       [] {
         return std::make_shared<ov::op::v3::Asinh>(
             param(ov::element::f32, ov::Shape{2, 3, 4}));
       },
       "asinh(x)"},
      {"Acosh",
       [] {
         return std::make_shared<ov::op::v3::Acosh>(
             param(ov::element::f32, ov::Shape{2, 3, 4}));
       },
       "acosh(x)"},
      {"Atanh",
       [] {
         return std::make_shared<ov::op::v3::Atanh>(
             param(ov::element::f32, ov::Shape{2, 3, 4}));
       },
       "atanh(x)"},
      {"Sinh",
       [] {
         return std::make_shared<ov::op::v0::Sinh>(
             param(ov::element::f32, ov::Shape{2, 3, 4}));
       },
       "sinh(x)"},
      {"Cosh",
       [] {
         return std::make_shared<ov::op::v0::Cosh>(
             param(ov::element::f32, ov::Shape{2, 3, 4}));
       },
       "cosh(x)"},
      {"RoundEven",
       [] {
         return std::make_shared<ov::op::v5::Round>(
             param(ov::element::f32, ov::Shape{2, 3, 4}),
             ov::op::v5::Round::RoundMode::HALF_TO_EVEN);
       },
       "rint(x)"},
      {"RoundAway",
       [] {
         return std::make_shared<ov::op::v5::Round>(
             param(ov::element::f32, ov::Shape{2, 3, 4}),
             ov::op::v5::Round::RoundMode::HALF_AWAY_FROM_ZERO);
       },
       "round(x)"},
  };
}

} // namespace gfx_plugin
} // namespace ov
