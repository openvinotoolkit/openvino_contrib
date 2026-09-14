// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//
#pragma once

#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>
#include <vector>

#include "backends/opencl/runtime/opencl_runtime_bundle.hpp"
#include "runtime/gpu_buffer_manager.hpp"

namespace ov {
namespace gfx_plugin {

struct _cl_platform_id;
struct _cl_device_id;
struct _cl_context;
struct _cl_command_queue;
struct _cl_mem;
struct _cl_program;
struct _cl_kernel;
struct _cl_event;

using cl_platform_id = _cl_platform_id*;
using cl_device_id = _cl_device_id*;
using cl_context = _cl_context*;
using cl_command_queue = _cl_command_queue*;
using cl_mem = _cl_mem*;
using cl_program = _cl_program*;
using cl_kernel = _cl_kernel*;
using cl_event = _cl_event*;
using cl_int = int32_t;
using cl_uint = uint32_t;
using cl_bool = cl_uint;
using cl_ulong = uint64_t;
using cl_bitfield = cl_ulong;
using cl_device_type = cl_bitfield;
using cl_mem_flags = cl_bitfield;
using cl_map_flags = cl_bitfield;
using cl_command_queue_properties = cl_bitfield;
using cl_context_properties = intptr_t;
using cl_queue_properties = intptr_t;

constexpr cl_int CL_SUCCESS = 0;
constexpr cl_bool CL_TRUE = 1;
constexpr cl_bool CL_FALSE = 0;

constexpr cl_device_type CL_DEVICE_TYPE_CPU = 1ull << 1;
constexpr cl_device_type CL_DEVICE_TYPE_GPU = 1ull << 2;
constexpr cl_device_type CL_DEVICE_TYPE_ACCELERATOR = 1ull << 3;

constexpr cl_mem_flags CL_MEM_READ_WRITE = 1ull << 0;
constexpr cl_mem_flags CL_MEM_WRITE_ONLY = 1ull << 1;
constexpr cl_mem_flags CL_MEM_READ_ONLY = 1ull << 2;

constexpr cl_map_flags CL_MAP_READ = 1ull << 0;
constexpr cl_map_flags CL_MAP_WRITE = 1ull << 1;

constexpr cl_uint CL_DEVICE_TYPE = 0x1000;
constexpr cl_uint CL_DEVICE_VENDOR_ID = 0x1001;
constexpr cl_uint CL_DEVICE_MAX_WORK_ITEM_DIMENSIONS = 0x1003;
constexpr cl_uint CL_DEVICE_MAX_WORK_GROUP_SIZE = 0x1004;
constexpr cl_uint CL_DEVICE_MAX_WORK_ITEM_SIZES = 0x1005;
constexpr cl_uint CL_DEVICE_MAX_COMPUTE_UNITS = 0x1002;
constexpr cl_uint CL_DEVICE_MEM_BASE_ADDR_ALIGN = 0x1019;
constexpr cl_uint CL_DEVICE_NAME = 0x102B;
constexpr cl_uint CL_DEVICE_VENDOR = 0x102C;
constexpr cl_uint CL_DRIVER_VERSION = 0x102D;
constexpr cl_uint CL_DEVICE_VERSION = 0x102F;
constexpr cl_uint CL_DEVICE_EXTENSIONS = 0x1030;

constexpr cl_uint CL_PLATFORM_NAME = 0x0902;

constexpr cl_uint CL_PROGRAM_BUILD_LOG = 0x1183;
constexpr cl_uint CL_PROGRAM_NUM_DEVICES = 0x1162;
constexpr cl_uint CL_PROGRAM_BINARY_SIZES = 0x1165;
constexpr cl_uint CL_PROGRAM_BINARIES = 0x1166;

constexpr cl_uint CL_MEM_SIZE = 0x1102;
constexpr cl_uint CL_MEM_CONTEXT = 0x1106;

struct OpenClFunctionTable {
    cl_int (*clGetPlatformIDs)(cl_uint, cl_platform_id*, cl_uint*) = nullptr;
    cl_int (*clGetPlatformInfo)(cl_platform_id, cl_uint, size_t, void*, size_t*) = nullptr;
    cl_int (*clGetDeviceIDs)(cl_platform_id, cl_device_type, cl_uint, cl_device_id*, cl_uint*) = nullptr;
    cl_int (*clGetDeviceInfo)(cl_device_id, cl_uint, size_t, void*, size_t*) = nullptr;
    cl_context (*clCreateContext)(const cl_context_properties*, cl_uint, const cl_device_id*,
                                  void (*)(const char*, const void*, size_t, void*), void*, cl_int*) = nullptr;
    cl_int (*clReleaseContext)(cl_context) = nullptr;
    cl_command_queue (*clCreateCommandQueueWithProperties)(cl_context, cl_device_id,
                                                           const cl_queue_properties*, cl_int*) = nullptr;
    cl_command_queue (*clCreateCommandQueue)(cl_context, cl_device_id, cl_command_queue_properties, cl_int*) = nullptr;
    cl_int (*clReleaseCommandQueue)(cl_command_queue) = nullptr;
    cl_int (*clFinish)(cl_command_queue) = nullptr;
    cl_mem (*clCreateBuffer)(cl_context, cl_mem_flags, size_t, void*, cl_int*) = nullptr;
    cl_int (*clGetMemObjectInfo)(cl_mem, cl_uint, size_t, void*, size_t*) = nullptr;
    cl_int (*clReleaseMemObject)(cl_mem) = nullptr;
    void* (*clEnqueueMapBuffer)(cl_command_queue, cl_mem, cl_bool, cl_map_flags, size_t, size_t,
                                cl_uint, const cl_event*, cl_event*, cl_int*) = nullptr;
    cl_int (*clEnqueueUnmapMemObject)(cl_command_queue, cl_mem, void*, cl_uint, const cl_event*, cl_event*) = nullptr;
    cl_int (*clEnqueueWriteBuffer)(cl_command_queue, cl_mem, cl_bool, size_t, size_t, const void*,
                                   cl_uint, const cl_event*, cl_event*) = nullptr;
    cl_int (*clEnqueueReadBuffer)(cl_command_queue, cl_mem, cl_bool, size_t, size_t, void*,
                                  cl_uint, const cl_event*, cl_event*) = nullptr;
    cl_int (*clEnqueueCopyBuffer)(cl_command_queue, cl_mem, cl_mem, size_t, size_t, size_t,
                                  cl_uint, const cl_event*, cl_event*) = nullptr;
    cl_program (*clCreateProgramWithSource)(cl_context, cl_uint, const char**, const size_t*, cl_int*) = nullptr;
    cl_program (*clCreateProgramWithBinary)(cl_context, cl_uint, const cl_device_id*, const size_t*,
                                            const unsigned char**, cl_int*, cl_int*) = nullptr;
    cl_int (*clBuildProgram)(cl_program, cl_uint, const cl_device_id*, const char*,
                             void (*)(cl_program, void*), void*) = nullptr;
    cl_int (*clGetProgramBuildInfo)(cl_program, cl_device_id, cl_uint, size_t, void*, size_t*) = nullptr;
    cl_int (*clGetProgramInfo)(cl_program, cl_uint, size_t, void*, size_t*) = nullptr;
    cl_int (*clRetainProgram)(cl_program) = nullptr;
    cl_int (*clReleaseProgram)(cl_program) = nullptr;
    cl_kernel (*clCreateKernel)(cl_program, const char*, cl_int*) = nullptr;
    cl_int (*clSetKernelArg)(cl_kernel, cl_uint, size_t, const void*) = nullptr;
    cl_int (*clEnqueueNDRangeKernel)(cl_command_queue, cl_kernel, cl_uint, const size_t*, const size_t*,
                                     const size_t*, cl_uint, const cl_event*, cl_event*) = nullptr;
    cl_int (*clReleaseKernel)(cl_kernel) = nullptr;
};

class OpenClApi {
public:
    static const OpenClApi& instance();

    const OpenClFunctionTable& fn() const { return m_fn; }
    const std::string& library_path() const { return m_library_path; }
    const OpenClRuntimeBundleInfo& bundle_info() const { return m_bundle_info; }

private:
    OpenClApi();
    ~OpenClApi();

    void* m_library = nullptr;
    std::string m_library_path;
    OpenClRuntimeBundleInfo m_bundle_info;
    OpenClFunctionTable m_fn;
};

struct OpenClDeviceSelection {
    cl_platform_id platform = nullptr;
    cl_device_id device = nullptr;
    cl_device_type device_type = 0;
    std::string platform_name;
    std::string device_name;
    std::string vendor_name;
    std::string driver_version;
    std::string device_version;
    std::string extensions;
    uint32_t vendor_id = 0;
    uint32_t compute_units = 1;
    uint32_t max_work_item_dimensions = 1;
    size_t max_work_group_size = 1;
    std::vector<size_t> max_work_item_sizes;
    uint64_t mem_base_addr_align_bits = 8;

    bool is_gpu() const {
        return (device_type & CL_DEVICE_TYPE_GPU) != 0;
    }
};

OpenClDeviceSelection select_opencl_gpu_device(const OpenClApi& api);
void validate_opencl_device_selection(const OpenClDeviceSelection& selection);
GpuExecutionDeviceInfo make_opencl_execution_device_info(const OpenClDeviceSelection& selection);
std::string opencl_error_string(cl_int error);
void opencl_check(cl_int status, const char* action);

class OpenClRuntimeContext {
public:
    static std::shared_ptr<OpenClRuntimeContext> instance();
    ~OpenClRuntimeContext();

    const OpenClApi& api() const { return m_api; }
    cl_context context() const { return m_context; }
    cl_command_queue queue() const { return m_queue; }
    cl_device_id device() const { return m_selection.device; }
    const OpenClDeviceSelection& selection() const { return m_selection; }
    GpuExecutionDeviceInfo execution_device_info() const;
    void finish() const;

private:
    OpenClRuntimeContext();

    const OpenClApi& m_api;
    OpenClDeviceSelection m_selection;
    cl_context m_context = nullptr;
    cl_command_queue m_queue = nullptr;
};

}  // namespace gfx_plugin
}  // namespace ov
