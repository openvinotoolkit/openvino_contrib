// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//
#pragma once

#include <cstddef>
#include <cstdint>
#include <string>
#include <string_view>
#include <vector>

#include "kernel_ir/gfx_kernel_manifest.hpp"
#include "common/artifact_payload.hpp"

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

struct GfxKernelSourceRuntimeBinding {
    std::vector<size_t> inputs;
    size_t input_arg_count = 0;
    std::vector<int32_t> operand_kinds;
    std::vector<int32_t> operand_arg_indices;
    std::vector<int32_t> scalar_args;

    bool valid() const noexcept {
        return operand_kinds.size() == operand_arg_indices.size();
    }
};

std::string_view gfx_kernel_source_language_name(GfxKernelSourceLanguage language) noexcept;
KernelArtifactPayloadKind gfx_kernel_source_payload_kind(
    GfxKernelSourceLanguage language) noexcept;
bool gfx_kernel_source_valid(const GfxKernelSource& source) noexcept;

class GfxKernelSourcePayload final : public KernelArtifactPayload {
public:
    explicit GfxKernelSourcePayload(GfxKernelSource source);
    GfxKernelSourcePayload(std::string kernel_id,
                           std::string backend_domain,
                           std::string entry_point,
                           GfxKernelSourceLanguage source_language,
                           std::string source);
    GfxKernelSourcePayload(std::string kernel_id,
                           std::string backend_domain,
                           std::string entry_point,
                           GfxKernelSourceLanguage source_language,
                           std::string source,
                           GfxKernelStageManifest stage_manifest);
    GfxKernelSourcePayload(std::string kernel_id,
                           std::string backend_domain,
                           std::string entry_point,
                           GfxKernelSourceLanguage source_language,
                           std::string source,
                           GfxKernelStageManifest stage_manifest,
                           GfxKernelSourceRuntimeBinding runtime_binding);

    GfxKernelSourcePayload(const GfxKernelSourcePayload&) = delete;
    GfxKernelSourcePayload& operator=(const GfxKernelSourcePayload&) = delete;
    GfxKernelSourcePayload(GfxKernelSourcePayload&&) = delete;
    GfxKernelSourcePayload& operator=(GfxKernelSourcePayload&&) = delete;

    KernelArtifactPayloadKind payload_kind() const noexcept override;
    std::string_view backend_domain() const noexcept override;
    std::string_view source_id() const noexcept override;
    std::string_view entry_point() const noexcept override;
    bool valid() const noexcept override;

    const GfxKernelSource& source() const noexcept {
        return m_source;
    }
    const GfxKernelStageManifest& stage_manifest() const noexcept {
        return m_stage_manifest;
    }
    bool has_stage_manifest() const noexcept {
        return m_stage_manifest.valid;
    }
    const GfxKernelSourceRuntimeBinding& runtime_binding() const noexcept {
        return m_runtime_binding;
    }
    bool has_runtime_binding() const noexcept {
        return m_runtime_binding.valid() &&
               (!m_runtime_binding.inputs.empty() ||
                !m_runtime_binding.operand_kinds.empty() ||
                !m_runtime_binding.scalar_args.empty());
    }

private:
    GfxKernelSource m_source{};
    GfxKernelStageManifest m_stage_manifest{};
    GfxKernelSourceRuntimeBinding m_runtime_binding{};
    std::string m_owned_kernel_id;
    std::string m_owned_backend_domain;
    std::string m_owned_entry_point;
    std::string m_owned_source;
};

}  // namespace gfx_plugin
}  // namespace ov
