// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "compiler/cache_repository.hpp"

#include <filesystem>
#include <fstream>
#include <iomanip>
#include <sstream>
#include <string_view>
#include <utility>

namespace ov {
namespace gfx_plugin {
namespace compiler {
namespace {

constexpr std::string_view kRequestIndexVersion = "gfx-cache-request-index-v1";
constexpr std::string_view kCompilerRevision = "gfx-compiler-cache-v1";

uint64_t stable_hash64(std::string_view value) noexcept {
  uint64_t hash = 14695981039346656037ull;
  for (const unsigned char c : value) {
    hash ^= c;
    hash *= 1099511628211ull;
  }
  return hash;
}

std::string hex64(uint64_t value) {
  std::ostringstream os;
  os << std::hex << std::setw(16) << std::setfill('0') << value;
  return os.str();
}

void append_field(std::ostringstream &os, std::string_view value) {
  os << value.size() << ":" << value << ";";
}

std::string hash_material(std::string_view material) {
  return hex64(stable_hash64(material));
}

bool request_complete(const CacheLookupRequest &request,
                      std::vector<std::string> *diagnostics = nullptr) {
  if (!request.model) {
    if (diagnostics) {
      diagnostics->emplace_back("cache lookup request model is null");
    }
    return false;
  }
  if (!request.target || request.target->backend() == GpuBackend::Unknown) {
    if (diagnostics) {
      diagnostics->emplace_back("cache lookup request target is unavailable");
    }
    return false;
  }
  if (!request.capabilities) {
    if (diagnostics) {
      diagnostics->emplace_back(
          "cache lookup request capabilities are missing");
    }
    return false;
  }
  return true;
}

bool envelope_matches_request(const CacheEnvelope &envelope,
                              const CacheLookupRequest &request,
                              std::vector<std::string> &diagnostics) {
  if (!request_complete(request, &diagnostics)) {
    return false;
  }
  const auto model_fingerprint = make_model_cache_fingerprint(*request.model);
  const auto capabilities_fingerprint =
      make_backend_capabilities_fingerprint(*request.capabilities);
  bool matched = true;
  const auto require_equal = [&](std::string_view label,
                                 const std::string &actual,
                                 const std::string &expected) {
    if (actual != expected) {
      diagnostics.emplace_back(std::string("cache envelope ") +
                               std::string(label) + " mismatch");
      matched = false;
    }
  };
  require_equal("model fingerprint", envelope.key.model_fingerprint,
                model_fingerprint);
  require_equal("target fingerprint", envelope.key.target_fingerprint,
                request.target->fingerprint());
  require_equal("capabilities fingerprint",
                envelope.key.backend_capabilities_fingerprint,
                capabilities_fingerprint);
  require_equal("compiler revision", envelope.key.compiler_revision,
                std::string(kCompilerRevision));
  require_equal("backend compiler revision",
                envelope.key.backend_compiler_revision,
                request.target->compiler_id());
  require_equal("driver identity", envelope.key.driver_identity,
                request.target->driver_id());
  return matched;
}

bool is_store_miss(const CacheEnvelopeWireResult &loaded) {
  return !loaded.diagnostics.empty() &&
         loaded.diagnostics.front().find("cache envelope miss:") == 0;
}

} // namespace

ArtifactCacheRepository::ArtifactCacheRepository(std::string cache_dir)
    : m_store(std::move(cache_dir)) {}

std::string
ArtifactCacheRepository::index_path(std::string_view request_key) const {
  if (!enabled() || request_key.empty()) {
    return {};
  }
  return (std::filesystem::path(m_store.cache_dir()) /
          ("gfx_request_" + std::string(request_key) + ".idx"))
      .string();
}

std::string make_artifact_cache_request_key(const CacheLookupRequest &request) {
  std::vector<std::string> diagnostics;
  if (!request_complete(request, &diagnostics)) {
    return {};
  }
  std::ostringstream material;
  append_field(material, kRequestIndexVersion);
  append_field(material, make_model_cache_fingerprint(*request.model));
  append_field(material, request.target->fingerprint());
  append_field(material,
               make_backend_capabilities_fingerprint(*request.capabilities));
  append_field(material, kCompilerRevision);
  append_field(material, request.target->compiler_id());
  append_field(material, request.target->driver_id());
  append_field(material, request.target->cache_compatibility_id());
  append_field(material, request.enable_fusion ? "fusion:on" : "fusion:off");
  return hash_material(material.str());
}

ArtifactCacheLookupResult
ArtifactCacheRepository::load(const CacheLookupRequest &request) const {
  ArtifactCacheLookupResult result;
  if (!enabled()) {
    result.status = ArtifactCacheLookupStatus::Disabled;
    result.diagnostics.emplace_back("artifact cache repository is disabled");
    return result;
  }
  if (!request_complete(request, &result.diagnostics)) {
    result.status = ArtifactCacheLookupStatus::Rejected;
    return result;
  }

  result.request_key = make_artifact_cache_request_key(request);
  const auto path = index_path(result.request_key);
  std::ifstream index(path);
  if (!index) {
    result.status = ArtifactCacheLookupStatus::Miss;
    result.diagnostics.push_back("cache request index miss: " + path);
    return result;
  }

  std::string stable_envelope_key;
  std::getline(index, stable_envelope_key);
  if (stable_envelope_key.empty()) {
    result.status = ArtifactCacheLookupStatus::Rejected;
    result.diagnostics.push_back("cache request index is empty: " + path);
    return result;
  }

  CacheKey envelope_key;
  envelope_key.stable_key = stable_envelope_key;
  auto loaded = m_store.load(envelope_key);
  if (!loaded.valid()) {
    result.diagnostics.insert(result.diagnostics.end(),
                              loaded.diagnostics.begin(),
                              loaded.diagnostics.end());
    result.status = is_store_miss(loaded) ? ArtifactCacheLookupStatus::Miss
                                          : ArtifactCacheLookupStatus::Rejected;
    return result;
  }

  std::vector<std::string> match_diagnostics;
  if (!envelope_matches_request(loaded.envelope, request, match_diagnostics)) {
    result.status = ArtifactCacheLookupStatus::Rejected;
    result.diagnostics.insert(result.diagnostics.end(),
                              match_diagnostics.begin(),
                              match_diagnostics.end());
    return result;
  }

  result.status = ArtifactCacheLookupStatus::Hit;
  result.envelope = std::move(loaded.envelope);
  return result;
}

ArtifactCacheStoreResult
ArtifactCacheRepository::store(const CacheLookupRequest &request,
                               const CacheEnvelope &envelope) const {
  ArtifactCacheStoreResult result;
  if (!enabled()) {
    result.diagnostics.emplace_back("artifact cache repository is disabled");
    return result;
  }
  std::vector<std::string> match_diagnostics;
  if (!envelope_matches_request(envelope, request, match_diagnostics)) {
    result.diagnostics.insert(result.diagnostics.end(),
                              match_diagnostics.begin(),
                              match_diagnostics.end());
    return result;
  }

  result = m_store.store(envelope);
  if (!result.success) {
    return result;
  }

  const auto request_key = make_artifact_cache_request_key(request);
  const auto path = index_path(request_key);
  std::error_code ec;
  std::filesystem::create_directories(
      std::filesystem::path(m_store.cache_dir()), ec);
  if (ec) {
    result.success = false;
    result.diagnostics.push_back("failed to create cache index directory: " +
                                 ec.message());
    return result;
  }
  std::ofstream index(path, std::ios::binary | std::ios::trunc);
  if (!index) {
    result.success = false;
    result.diagnostics.push_back(
        "failed to open cache request index for write");
    return result;
  }
  index << envelope.key.stable_key << '\n';
  if (!index) {
    result.success = false;
    result.diagnostics.push_back("failed to write cache request index");
    return result;
  }
  result.location = path;
  return result;
}

} // namespace compiler
} // namespace gfx_plugin
} // namespace ov
