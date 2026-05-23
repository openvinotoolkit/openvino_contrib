// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "backends/opencl/runtime/opencl_program_cache.hpp"

#include "openvino/core/except.hpp"

#include <algorithm>
#include <sstream>

namespace ov {
namespace gfx_plugin {

namespace {

cl_command_queue resolve_queue(const std::shared_ptr<OpenClRuntimeContext>& context,
                               GpuCommandBufferHandle command_buffer) {
    if (command_buffer) {
        return reinterpret_cast<cl_command_queue>(command_buffer);
    }
    return context->queue();
}

}  // namespace

OpenClKernel::OpenClKernel(std::shared_ptr<OpenClRuntimeContext> context,
                           cl_program program,
                           cl_kernel kernel,
                           std::string entry_point,
                           uint32_t arg_count)
    : m_context(std::move(context)),
      m_program(program),
      m_kernel(kernel),
      m_entry_point(std::move(entry_point)),
      m_arg_count(arg_count) {}

OpenClKernel::~OpenClKernel() {
    if (m_kernel) {
        m_context->api().fn().clReleaseKernel(m_kernel);
    }
    if (m_program) {
        m_context->api().fn().clReleaseProgram(m_program);
    }
}

void OpenClKernel::set_args_count(uint32_t count) {
    if (count == 0) {
        return;
    }
    if (m_arg_count == 0) {
        m_arg_count = count;
        return;
    }
    OPENVINO_ASSERT(m_arg_count == count,
                    "GFX OpenCL: kernel arg count mismatch (expected ",
                    m_arg_count,
                    ", got ",
                    count,
                    ")");
}

size_t OpenClKernel::clamp_threadgroup_size(size_t desired) const {
    const auto max_group = std::max<size_t>(m_context->selection().max_work_group_size, 1);
    return std::max<size_t>(1, std::min(desired, max_group));
}

std::shared_ptr<ICompiledKernel> OpenClKernel::fork() const {
    opencl_check(m_context->api().fn().clRetainProgram(m_program), "clRetainProgram");
    cl_int status = CL_SUCCESS;
    cl_kernel kernel = m_context->api().fn().clCreateKernel(m_program, m_entry_point.c_str(), &status);
    opencl_check(status, "clCreateKernel(fork)");
    return std::make_shared<OpenClKernel>(m_context, m_program, kernel, m_entry_point, m_arg_count);
}

void OpenClKernel::execute(GpuCommandBufferHandle command_buffer,
                           const KernelDispatch& dispatch,
                           const std::vector<KernelArg>& args,
                           const KernelExecutionHooks* hooks) {
    const auto arg_count = ensure_kernel_args_dense(args, "GFX OpenCL");
    set_args_count(arg_count);
    for (uint32_t i = 0; i < arg_count; ++i) {
        const auto it = std::find_if(args.begin(), args.end(), [i](const KernelArg& arg) {
            return arg.index == i;
        });
        OPENVINO_ASSERT(it != args.end(), "GFX OpenCL: missing kernel argument ", i);
        if (it->kind == KernelArg::Kind::Buffer) {
            OPENVINO_ASSERT(it->offset == 0,
                            "GFX OpenCL: non-zero buffer offsets require manifest/runtime-param lowering");
            cl_mem mem = reinterpret_cast<cl_mem>(it->buffer.buffer);
            opencl_check(m_context->api().fn().clSetKernelArg(m_kernel, i, sizeof(cl_mem), &mem), "clSetKernelArg(buffer)");
        } else {
            opencl_check(m_context->api().fn().clSetKernelArg(m_kernel, i, it->byte_size, it->bytes),
                         "clSetKernelArg(bytes)");
        }
    }

    if (hooks && hooks->on_begin) {
        hooks->on_begin(command_buffer);
    }
    const size_t global[3] = {dispatch.grid[0], dispatch.grid[1], dispatch.grid[2]};
    const size_t local[3] = {dispatch.threads_per_group[0],
                             dispatch.threads_per_group[1],
                             dispatch.threads_per_group[2]};
    opencl_check(m_context->api().fn().clEnqueueNDRangeKernel(resolve_queue(m_context, command_buffer),
                                                             m_kernel,
                                                             3,
                                                             nullptr,
                                                             global,
                                                             local,
                                                             0,
                                                             nullptr,
                                                             nullptr),
                 "clEnqueueNDRangeKernel");
    m_context->finish();
    if (hooks && hooks->on_end) {
        hooks->on_end(command_buffer);
    }
    if (hooks && hooks->on_complete) {
        hooks->on_complete();
    }
}

OpenClProgramCache::OpenClProgramCache(std::shared_ptr<OpenClRuntimeContext> context)
    : m_context(std::move(context)) {
    OPENVINO_ASSERT(m_context, "GFX OpenCL: program cache requires runtime context");
}

std::string OpenClProgramCache::build_log(cl_program program) const {
    size_t bytes = 0;
    const auto& cl = m_context->api().fn();
    cl.clGetProgramBuildInfo(program, m_context->device(), CL_PROGRAM_BUILD_LOG, 0, nullptr, &bytes);
    if (bytes == 0) {
        return {};
    }
    std::string log(bytes, '\0');
    cl.clGetProgramBuildInfo(program, m_context->device(), CL_PROGRAM_BUILD_LOG, log.size(), log.data(), nullptr);
    while (!log.empty() && log.back() == '\0') {
        log.pop_back();
    }
    return log;
}

std::shared_ptr<OpenClKernel> OpenClProgramCache::get_or_create(const std::string& source_id,
                                                                const std::string& source,
                                                                const std::string& entry_point,
                                                                const std::string& build_options) {
    const std::string key = source_id + "\n" + entry_point + "\n" + build_options + "\n" + source;
    std::lock_guard<std::mutex> lock(m_mutex);
    if (auto it = m_cache.find(key); it != m_cache.end()) {
        if (auto cached = it->second.lock()) {
            return cached;
        }
    }

    const char* source_ptr = source.c_str();
    const size_t source_size = source.size();
    cl_int status = CL_SUCCESS;
    cl_program program = m_context->api().fn().clCreateProgramWithSource(m_context->context(),
                                                                         1,
                                                                         &source_ptr,
                                                                         &source_size,
                                                                         &status);
    opencl_check(status, "clCreateProgramWithSource");
    status = m_context->api().fn().clBuildProgram(program,
                                                  1,
                                                  &m_context->selection().device,
                                                  build_options.empty() ? nullptr : build_options.c_str(),
                                                  nullptr,
                                                  nullptr);
    if (status != CL_SUCCESS) {
        const auto log = build_log(program);
        m_context->api().fn().clReleaseProgram(program);
        OPENVINO_THROW("GFX OpenCL: clBuildProgram failed for ",
                       entry_point,
                       ": ",
                       opencl_error_string(status),
                       log.empty() ? "" : "\n",
                       log);
    }
    cl_kernel kernel = m_context->api().fn().clCreateKernel(program, entry_point.c_str(), &status);
    if (status != CL_SUCCESS || !kernel) {
        m_context->api().fn().clReleaseProgram(program);
        opencl_check(status, "clCreateKernel");
        OPENVINO_THROW("GFX OpenCL: clCreateKernel returned null for ", entry_point);
    }

    auto compiled = std::make_shared<OpenClKernel>(m_context, program, kernel, entry_point);
    m_cache[key] = compiled;
    return compiled;
}

}  // namespace gfx_plugin
}  // namespace ov
