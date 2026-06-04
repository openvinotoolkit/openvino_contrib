// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//
#pragma once

#include <string_view>

namespace ov {
namespace gfx_plugin {

enum class KernelArtifactOrigin {
  Unknown,
  Common,
  Metadata,
  VendorPrimitive,
  Generated,
  HandwrittenException,
};

enum class KernelArtifactPayloadKind {
  None,
  VendorDescriptor,
  MslSource,
  OpenClSource,
};

class KernelArtifactPayload {
public:
  virtual ~KernelArtifactPayload() = default;

  virtual KernelArtifactPayloadKind payload_kind() const noexcept = 0;
  virtual std::string_view backend_domain() const noexcept = 0;
  virtual std::string_view source_id() const noexcept = 0;
  virtual std::string_view entry_point() const noexcept = 0;
  virtual bool valid() const noexcept = 0;
};

} // namespace gfx_plugin
} // namespace ov
