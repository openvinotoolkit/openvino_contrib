// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//
#pragma once

#include <string>
#include <string_view>

#include "compiler/operation_support.hpp"

namespace ov {
namespace gfx_plugin {
namespace compiler {

enum class KernelUnitKind {
  Common,
  Metadata,
  VendorPrimitive,
  GeneratedKernel,
  HandwrittenException,
  BackendLowering,
};

struct HandwrittenKernelExceptionContract {
  std::string ticket;
  std::string reason;
  std::string removal_condition;

  bool valid() const noexcept {
    return !ticket.empty() && !reason.empty() && !removal_condition.empty();
  }
};

class KernelUnit final {
public:
  KernelUnit() = default;

  static KernelUnit from_route(LoweringRouteKind route_kind,
                               std::string unit_id);
  static KernelUnit describe(LoweringRouteKind route_kind, KernelUnitKind kind,
                             std::string unit_id, std::string backend_domain,
                             std::string op_family);
  static KernelUnit describe_handwritten_exception(
      std::string unit_id, std::string backend_domain, std::string op_family,
      HandwrittenKernelExceptionContract exception);

  const std::string &id() const noexcept { return m_id; }

  KernelUnitKind kind() const noexcept { return m_kind; }

  LoweringRouteKind route_kind() const noexcept { return m_route_kind; }

  const std::string &backend_domain() const noexcept {
    return m_backend_domain;
  }

  const std::string &op_family() const noexcept { return m_op_family; }

  const HandwrittenKernelExceptionContract &
  exception_contract() const noexcept {
    return m_exception;
  }

  bool valid() const noexcept {
    return m_route_kind != LoweringRouteKind::Unsupported && !m_id.empty();
  }

  std::string manifest_key() const;

private:
  KernelUnit(LoweringRouteKind route_kind, KernelUnitKind kind,
             std::string unit_id, std::string backend_domain,
             std::string op_family,
             HandwrittenKernelExceptionContract exception = {});

  KernelUnitKind m_kind = KernelUnitKind::BackendLowering;
  LoweringRouteKind m_route_kind = LoweringRouteKind::Unsupported;
  std::string m_id;
  std::string m_backend_domain;
  std::string m_op_family;
  HandwrittenKernelExceptionContract m_exception;
};

std::string_view kernel_unit_kind_to_string(KernelUnitKind kind) noexcept;
std::string_view kernel_unit_route_to_string(const KernelUnit &unit) noexcept;

} // namespace compiler
} // namespace gfx_plugin
} // namespace ov
