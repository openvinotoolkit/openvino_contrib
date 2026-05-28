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

std::vector<ActivationOpenClArtifactCase> activation_opencl_artifact_cases() {
  return {
      {"F32Relu",
       [] {
         return std::make_shared<ov::op::v0::Relu>(
             param(ov::element::f32, ov::Shape{2, 3, 4}));
       },
       "opencl/generated/activation_f32", "gfx_opencl_generated_activation_f32",
       GfxOpenClBaselineOp::Relu},
      {"F32Sigmoid",
       [] {
         return std::make_shared<ov::op::v0::Sigmoid>(
             param(ov::element::f32, ov::Shape{2, 3, 4}));
       },
       "opencl/generated/activation_f32", "gfx_opencl_generated_activation_f32",
       GfxOpenClBaselineOp::Sigmoid},
      {"F16Tanh",
       [] {
         return std::make_shared<ov::op::v0::Tanh>(
             param(ov::element::f16, ov::Shape{2, 3, 4}));
       },
       "opencl/generated/activation_f16", "gfx_opencl_generated_activation_f16",
       GfxOpenClBaselineOp::Tanh},
      {"F32Elu",
       [] {
         return std::make_shared<ov::op::v0::Elu>(
             param(ov::element::f32, ov::Shape{2, 3, 4}), 0.5);
       },
       "opencl/generated/activation_f32",
       "gfx_opencl_generated_activation_f32",
       GfxOpenClBaselineOp::Elu,
       {0.5f, 0.0f}},
      {"F32Clamp",
       [] {
         return std::make_shared<ov::op::v0::Clamp>(
             param(ov::element::f32, ov::Shape{2, 3, 4}), -0.25, 0.75);
       },
       "opencl/generated/activation_f32",
       "gfx_opencl_generated_activation_f32",
       GfxOpenClBaselineOp::Clamp,
       {-0.25f, 0.75f}},
      {"F32GeluTanh",
       [] {
         return std::make_shared<ov::op::v7::Gelu>(
             param(ov::element::f32, ov::Shape{2, 3, 4}),
             ov::op::GeluApproximationMode::TANH);
       },
       "opencl/generated/activation_f32", "gfx_opencl_generated_activation_f32",
       GfxOpenClBaselineOp::GeluTanh},
      {"F32HSwish",
       [] {
         return std::make_shared<ov::op::v4::HSwish>(
             param(ov::element::f32, ov::Shape{2, 3, 4}));
       },
       "opencl/generated/activation_f32", "gfx_opencl_generated_activation_f32",
       GfxOpenClBaselineOp::HSwish},
      {"F32HSigmoid",
       [] {
         return std::make_shared<ov::op::v5::HSigmoid>(
             param(ov::element::f32, ov::Shape{2, 3, 4}));
       },
       "opencl/generated/activation_f32", "gfx_opencl_generated_activation_f32",
       GfxOpenClBaselineOp::HSigmoid},
      {"F32SoftPlus",
       [] {
         return std::make_shared<ov::op::v4::SoftPlus>(
             param(ov::element::f32, ov::Shape{2, 3, 4}));
       },
       "opencl/generated/activation_f32", "gfx_opencl_generated_activation_f32",
       GfxOpenClBaselineOp::SoftPlus},
      {"F32SwishDefaultBeta",
       [] {
         return std::make_shared<ov::op::v4::Swish>(
             param(ov::element::f32, ov::Shape{2, 3, 4}));
       },
       "opencl/generated/activation_f32",
       "gfx_opencl_generated_activation_f32",
       GfxOpenClBaselineOp::Swish,
       {1.0f, 0.0f}},
      {"F32SwishStaticBeta",
       [] {
         const auto input = param(ov::element::f32, ov::Shape{2, 3, 4});
         const auto beta = ov::op::v0::Constant::create(ov::element::f32,
                                                        ov::Shape{}, {0.5f});
         return std::make_shared<ov::op::v4::Swish>(input, beta);
       },
       "opencl/generated/activation_f32",
       "gfx_opencl_generated_activation_f32",
       GfxOpenClBaselineOp::Swish,
       {0.5f, 0.0f}},
      {"F32SoftSign",
       [] {
         return std::make_shared<ov::op::v9::SoftSign>(
             param(ov::element::f32, ov::Shape{2, 3, 4}));
       },
       "opencl/generated/activation_f32", "gfx_opencl_generated_activation_f32",
       GfxOpenClBaselineOp::SoftSign},
      {"F32Sign",
       [] {
         return std::make_shared<ov::op::v0::Sign>(
             param(ov::element::f32, ov::Shape{2, 3, 4}));
       },
       "opencl/generated/activation_f32", "gfx_opencl_generated_activation_f32",
       GfxOpenClBaselineOp::Sign},
      {"F32Abs",
       [] {
         return std::make_shared<ov::op::v0::Abs>(
             param(ov::element::f32, ov::Shape{2, 3, 4}));
       },
       "opencl/generated/activation_f32", "gfx_opencl_generated_activation_f32",
       GfxOpenClBaselineOp::Abs},
      {"F32Negative",
       [] {
         return std::make_shared<ov::op::v0::Negative>(
             param(ov::element::f32, ov::Shape{2, 3, 4}));
       },
       "opencl/generated/activation_f32", "gfx_opencl_generated_activation_f32",
       GfxOpenClBaselineOp::Negative},
      {"F32Exp",
       [] {
         return std::make_shared<ov::op::v0::Exp>(
             param(ov::element::f32, ov::Shape{2, 3, 4}));
       },
       "opencl/generated/activation_f32", "gfx_opencl_generated_activation_f32",
       GfxOpenClBaselineOp::Exp},
      {"F32Log",
       [] {
         return std::make_shared<ov::op::v0::Log>(
             param(ov::element::f32, ov::Shape{2, 3, 4}));
       },
       "opencl/generated/activation_f32", "gfx_opencl_generated_activation_f32",
       GfxOpenClBaselineOp::Log},
      {"F32Sqrt",
       [] {
         return std::make_shared<ov::op::v0::Sqrt>(
             param(ov::element::f32, ov::Shape{2, 3, 4}));
       },
       "opencl/generated/activation_f32", "gfx_opencl_generated_activation_f32",
       GfxOpenClBaselineOp::Sqrt},
      {"F32Floor",
       [] {
         return std::make_shared<ov::op::v0::Floor>(
             param(ov::element::f32, ov::Shape{2, 3, 4}));
       },
       "opencl/generated/activation_f32", "gfx_opencl_generated_activation_f32",
       GfxOpenClBaselineOp::Floor},
      {"F32Ceiling",
       [] {
         return std::make_shared<ov::op::v0::Ceiling>(
             param(ov::element::f32, ov::Shape{2, 3, 4}));
       },
       "opencl/generated/activation_f32", "gfx_opencl_generated_activation_f32",
       GfxOpenClBaselineOp::Ceiling},
      {"F32Mish",
       [] {
         return std::make_shared<ov::op::v4::Mish>(
             param(ov::element::f32, ov::Shape{2, 3, 4}));
       },
       "opencl/generated/activation_f32", "gfx_opencl_generated_activation_f32",
       GfxOpenClBaselineOp::Mish},
      {"F32Sin",
       [] {
         return std::make_shared<ov::op::v0::Sin>(
             param(ov::element::f32, ov::Shape{2, 3, 4}));
       },
       "opencl/generated/activation_f32", "gfx_opencl_generated_activation_f32",
       GfxOpenClBaselineOp::Sin},
      {"F32Cos",
       [] {
         return std::make_shared<ov::op::v0::Cos>(
             param(ov::element::f32, ov::Shape{2, 3, 4}));
       },
       "opencl/generated/activation_f32", "gfx_opencl_generated_activation_f32",
       GfxOpenClBaselineOp::Cos},
      {"F32Tan",
       [] {
         return std::make_shared<ov::op::v0::Tan>(
             param(ov::element::f32, ov::Shape{2, 3, 4}));
       },
       "opencl/generated/activation_f32", "gfx_opencl_generated_activation_f32",
       GfxOpenClBaselineOp::Tan},
      {"F32Erf",
       [] {
         return std::make_shared<ov::op::v0::Erf>(
             param(ov::element::f32, ov::Shape{2, 3, 4}));
       },
       "opencl/generated/activation_f32", "gfx_opencl_generated_activation_f32",
       GfxOpenClBaselineOp::Erf},
      {"F32Asin",
       [] {
         return std::make_shared<ov::op::v0::Asin>(
             param(ov::element::f32, ov::Shape{2, 3, 4}));
       },
       "opencl/generated/activation_f32", "gfx_opencl_generated_activation_f32",
       GfxOpenClBaselineOp::Asin},
      {"F32Acos",
       [] {
         return std::make_shared<ov::op::v0::Acos>(
             param(ov::element::f32, ov::Shape{2, 3, 4}));
       },
       "opencl/generated/activation_f32", "gfx_opencl_generated_activation_f32",
       GfxOpenClBaselineOp::Acos},
      {"F32Atan",
       [] {
         return std::make_shared<ov::op::v0::Atan>(
             param(ov::element::f32, ov::Shape{2, 3, 4}));
       },
       "opencl/generated/activation_f32", "gfx_opencl_generated_activation_f32",
       GfxOpenClBaselineOp::Atan},
      {"F32Asinh",
       [] {
         return std::make_shared<ov::op::v3::Asinh>(
             param(ov::element::f32, ov::Shape{2, 3, 4}));
       },
       "opencl/generated/activation_f32", "gfx_opencl_generated_activation_f32",
       GfxOpenClBaselineOp::Asinh},
      {"F32Acosh",
       [] {
         return std::make_shared<ov::op::v3::Acosh>(
             param(ov::element::f32, ov::Shape{2, 3, 4}));
       },
       "opencl/generated/activation_f32", "gfx_opencl_generated_activation_f32",
       GfxOpenClBaselineOp::Acosh},
      {"F32Atanh",
       [] {
         return std::make_shared<ov::op::v3::Atanh>(
             param(ov::element::f32, ov::Shape{2, 3, 4}));
       },
       "opencl/generated/activation_f32", "gfx_opencl_generated_activation_f32",
       GfxOpenClBaselineOp::Atanh},
      {"F32Sinh",
       [] {
         return std::make_shared<ov::op::v0::Sinh>(
             param(ov::element::f32, ov::Shape{2, 3, 4}));
       },
       "opencl/generated/activation_f32", "gfx_opencl_generated_activation_f32",
       GfxOpenClBaselineOp::Sinh},
      {"F32Cosh",
       [] {
         return std::make_shared<ov::op::v0::Cosh>(
             param(ov::element::f32, ov::Shape{2, 3, 4}));
       },
       "opencl/generated/activation_f32", "gfx_opencl_generated_activation_f32",
       GfxOpenClBaselineOp::Cosh},
      {"F32RoundEven",
       [] {
         return std::make_shared<ov::op::v5::Round>(
             param(ov::element::f32, ov::Shape{2, 3, 4}),
             ov::op::v5::Round::RoundMode::HALF_TO_EVEN);
       },
       "opencl/generated/activation_f32", "gfx_opencl_generated_activation_f32",
       GfxOpenClBaselineOp::RoundEven},
      {"F32RoundAway",
       [] {
         return std::make_shared<ov::op::v5::Round>(
             param(ov::element::f32, ov::Shape{2, 3, 4}),
             ov::op::v5::Round::RoundMode::HALF_AWAY_FROM_ZERO);
       },
       "opencl/generated/activation_f32", "gfx_opencl_generated_activation_f32",
       GfxOpenClBaselineOp::RoundAway},
  };
}

} // namespace gfx_plugin
} // namespace ov
