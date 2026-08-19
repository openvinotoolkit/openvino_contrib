// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "kernel_ir/gfx_kernel_source.hpp"

namespace ov {
namespace gfx_plugin {

namespace {

std::string_view safe_view(const char* value) noexcept {
    return value ? std::string_view(value) : std::string_view();
}

}  // namespace

std::string_view gfx_kernel_source_language_name(GfxKernelSourceLanguage language) noexcept {
    switch (language) {
    case GfxKernelSourceLanguage::OpenCL:
        return "opencl";
    case GfxKernelSourceLanguage::MetalShadingLanguage:
        return "metal_shading_language";
    }
    return "unknown";
}

KernelArtifactPayloadKind gfx_kernel_source_payload_kind(
    GfxKernelSourceLanguage language) noexcept {
    switch (language) {
    case GfxKernelSourceLanguage::OpenCL:
        return KernelArtifactPayloadKind::OpenClSource;
    case GfxKernelSourceLanguage::MetalShadingLanguage:
        return KernelArtifactPayloadKind::MslSource;
    }
    return KernelArtifactPayloadKind::None;
}

bool gfx_kernel_source_valid(const GfxKernelSource& source) noexcept {
    return source.kernel_id && source.kernel_id[0] != '\0' &&
           source.backend_domain && source.backend_domain[0] != '\0' &&
           source.entry_point && source.entry_point[0] != '\0' &&
           source.source && source.source[0] != '\0' &&
           gfx_kernel_source_payload_kind(source.source_language) !=
               KernelArtifactPayloadKind::None;
}

GfxKernelSourcePayload::GfxKernelSourcePayload(GfxKernelSource source)
    : m_source(source) {}

GfxKernelSourcePayload::GfxKernelSourcePayload(std::string kernel_id,
                                               std::string backend_domain,
                                               std::string entry_point,
                                               GfxKernelSourceLanguage source_language,
                                               std::string source)
    : m_owned_kernel_id(std::move(kernel_id)),
      m_owned_backend_domain(std::move(backend_domain)),
      m_owned_entry_point(std::move(entry_point)),
      m_owned_source(std::move(source)) {
    m_source.kernel_id = m_owned_kernel_id.c_str();
    m_source.backend_domain = m_owned_backend_domain.c_str();
    m_source.entry_point = m_owned_entry_point.c_str();
    m_source.source_language = source_language;
    m_source.source = m_owned_source.c_str();
}

GfxKernelSourcePayload::GfxKernelSourcePayload(
    std::string kernel_id,
    std::string backend_domain,
    std::string entry_point,
    GfxKernelSourceLanguage source_language,
    std::string source,
    GfxKernelStageManifest stage_manifest)
    : GfxKernelSourcePayload(std::move(kernel_id),
                             std::move(backend_domain),
                             std::move(entry_point),
                             source_language,
                             std::move(source)) {
    m_stage_manifest = std::move(stage_manifest);
}

GfxKernelSourcePayload::GfxKernelSourcePayload(
    std::string kernel_id,
    std::string backend_domain,
    std::string entry_point,
    GfxKernelSourceLanguage source_language,
    std::string source,
    GfxKernelStageManifest stage_manifest,
    GfxKernelSourceRuntimeBinding runtime_binding)
    : GfxKernelSourcePayload(std::move(kernel_id),
                             std::move(backend_domain),
                             std::move(entry_point),
                             source_language,
                             std::move(source),
                             std::move(stage_manifest)) {
    m_runtime_binding = std::move(runtime_binding);
}

KernelArtifactPayloadKind GfxKernelSourcePayload::payload_kind() const noexcept {
    return gfx_kernel_source_payload_kind(m_source.source_language);
}

std::string_view GfxKernelSourcePayload::backend_domain() const noexcept {
    return safe_view(m_source.backend_domain);
}

std::string_view GfxKernelSourcePayload::source_id() const noexcept {
    return safe_view(m_source.kernel_id);
}

std::string_view GfxKernelSourcePayload::entry_point() const noexcept {
    return safe_view(m_source.entry_point);
}

bool GfxKernelSourcePayload::valid() const noexcept {
    return gfx_kernel_source_valid(m_source);
}

}  // namespace gfx_plugin
}  // namespace ov
