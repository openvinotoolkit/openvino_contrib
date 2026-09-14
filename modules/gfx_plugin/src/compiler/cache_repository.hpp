// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//
#pragma once

#include <string>
#include <string_view>
#include <vector>

#include "compiler/cache_envelope.hpp"
#include "compiler/operation_support.hpp"

namespace ov {
class Model;
} // namespace ov

namespace ov {
namespace gfx_plugin {
namespace compiler {

struct CacheLookupRequest {
  const ov::Model *model = nullptr;
  const BackendTarget *target = nullptr;
  const BackendCapabilities *capabilities = nullptr;
  bool enable_fusion = true;
};

enum class ArtifactCacheLookupStatus {
  Disabled,
  Miss,
  Hit,
  Rejected,
};

struct ArtifactCacheLookupResult {
  ArtifactCacheLookupStatus status = ArtifactCacheLookupStatus::Disabled;
  CacheEnvelope envelope;
  std::string request_key;
  std::vector<std::string> diagnostics;

  bool hit() const noexcept { return status == ArtifactCacheLookupStatus::Hit; }
  bool rejected() const noexcept {
    return status == ArtifactCacheLookupStatus::Rejected;
  }
};

class ArtifactCacheRepository final {
public:
  explicit ArtifactCacheRepository(std::string cache_dir);

  bool enabled() const noexcept { return m_store.enabled(); }

  ArtifactCacheLookupResult load(const CacheLookupRequest &request) const;
  ArtifactCacheStoreResult store(const CacheLookupRequest &request,
                                 const CacheEnvelope &envelope) const;

private:
  std::string index_path(std::string_view request_key) const;

  ArtifactCacheStore m_store;
};

std::string make_artifact_cache_request_key(const CacheLookupRequest &request);

} // namespace compiler
} // namespace gfx_plugin
} // namespace ov
