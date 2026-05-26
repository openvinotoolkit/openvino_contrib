// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//
#pragma once

#include <string>
#include <string_view>

#include "compiler/executable_bundle.hpp"

namespace ov {
namespace gfx_plugin {

enum class GfxKernelSourceLanguage {
    OpenCL,
    MetalShadingLanguage,
};

struct GfxKernelSource {
    const char* kernel_id = nullptr;
    const char* backend_domain = nullptr;
    const char* entry_point = nullptr;
    GfxKernelSourceLanguage source_language = GfxKernelSourceLanguage::OpenCL;
    const char* source = nullptr;
};

std::string_view gfx_kernel_source_language_name(GfxKernelSourceLanguage language) noexcept;
compiler::KernelArtifactPayloadKind gfx_kernel_source_payload_kind(
    GfxKernelSourceLanguage language) noexcept;
bool gfx_kernel_source_valid(const GfxKernelSource& source) noexcept;

class GfxKernelSourcePayload final : public compiler::KernelArtifactPayload {
public:
    explicit GfxKernelSourcePayload(GfxKernelSource source);
    GfxKernelSourcePayload(std::string kernel_id,
                           std::string backend_domain,
                           std::string entry_point,
                           GfxKernelSourceLanguage source_language,
                           std::string source);

    GfxKernelSourcePayload(const GfxKernelSourcePayload&) = delete;
    GfxKernelSourcePayload& operator=(const GfxKernelSourcePayload&) = delete;
    GfxKernelSourcePayload(GfxKernelSourcePayload&&) = delete;
    GfxKernelSourcePayload& operator=(GfxKernelSourcePayload&&) = delete;

    compiler::KernelArtifactPayloadKind payload_kind() const noexcept override;
    std::string_view backend_domain() const noexcept override;
    std::string_view source_id() const noexcept override;
    std::string_view entry_point() const noexcept override;
    bool valid() const noexcept override;

    const GfxKernelSource& source() const noexcept {
        return m_source;
    }

private:
    GfxKernelSource m_source{};
    std::string m_owned_kernel_id;
    std::string m_owned_backend_domain;
    std::string m_owned_entry_point;
    std::string m_owned_source;
};

}  // namespace gfx_plugin
}  // namespace ov
