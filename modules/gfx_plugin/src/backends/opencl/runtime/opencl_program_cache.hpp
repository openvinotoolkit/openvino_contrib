// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//
#pragma once

#include <memory>
#include <mutex>
#include <string>
#include <unordered_map>

#include "backends/opencl/runtime/opencl_api.hpp"
#include "runtime/gpu_backend_base.hpp"

namespace ov {
namespace gfx_plugin {

class OpenClKernel final : public ICompiledKernel {
public:
    OpenClKernel(std::shared_ptr<OpenClRuntimeContext> context,
                 cl_program program,
                 cl_kernel kernel,
                 std::string entry_point,
                 uint32_t arg_count = 0);
    ~OpenClKernel() override;

    uint32_t args_count() const override { return m_arg_count; }
    void set_args_count(uint32_t count) override;
    size_t clamp_threadgroup_size(size_t desired) const override;
    std::shared_ptr<ICompiledKernel> fork() const override;
    void execute(GpuCommandBufferHandle command_buffer,
                 const KernelDispatch& dispatch,
                 const std::vector<KernelArg>& args,
                 const KernelExecutionHooks* hooks = nullptr) override;

private:
    std::shared_ptr<OpenClRuntimeContext> m_context;
    cl_program m_program = nullptr;
    cl_kernel m_kernel = nullptr;
    std::string m_entry_point;
    uint32_t m_arg_count = 0;
};

class OpenClProgramCache {
public:
    explicit OpenClProgramCache(std::shared_ptr<OpenClRuntimeContext> context);
    std::shared_ptr<OpenClKernel> get_or_create(const std::string& source_id,
                                                const std::string& source,
                                                const std::string& entry_point,
                                                const std::string& build_options);

private:
    std::string build_log(cl_program program) const;

    std::shared_ptr<OpenClRuntimeContext> m_context;
    std::mutex m_mutex;
    std::unordered_map<std::string, std::weak_ptr<OpenClKernel>> m_cache;
};

}  // namespace gfx_plugin
}  // namespace ov
