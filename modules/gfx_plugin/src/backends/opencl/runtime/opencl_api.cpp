// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "backends/opencl/runtime/opencl_api.hpp"

#include "openvino/core/except.hpp"

#include <algorithm>
#include <cctype>
#include <cstring>
#include <mutex>
#include <sstream>

#if !defined(_WIN32)
#    include <dlfcn.h>
#endif

namespace ov {
namespace gfx_plugin {

namespace {

template <typename Fn>
void load_required_symbol(void* library, const char* name, Fn& fn) {
#if defined(_WIN32)
    (void)library;
    (void)name;
    (void)fn;
    OPENVINO_THROW("GFX OpenCL: Windows OpenCL loader is not part of the supported target set");
#else
    fn = reinterpret_cast<Fn>(dlsym(library, name));
    OPENVINO_ASSERT(fn, "GFX OpenCL: missing required symbol ", name);
#endif
}

template <typename Fn>
void load_optional_symbol(void* library, const char* name, Fn& fn) {
#if defined(_WIN32)
    (void)library;
    (void)name;
    (void)fn;
#else
    fn = reinterpret_cast<Fn>(dlsym(library, name));
#endif
}

std::string read_device_string(const OpenClApi& api, cl_device_id device, cl_uint param) {
    size_t bytes = 0;
    const auto& cl = api.fn();
    cl_int status = cl.clGetDeviceInfo(device, param, 0, nullptr, &bytes);
    if (status != CL_SUCCESS || bytes == 0) {
        return {};
    }
    std::string value(bytes, '\0');
    status = cl.clGetDeviceInfo(device, param, value.size(), value.data(), nullptr);
    if (status != CL_SUCCESS) {
        return {};
    }
    while (!value.empty() && value.back() == '\0') {
        value.pop_back();
    }
    return value;
}

template <typename T>
T read_device_scalar(const OpenClApi& api, cl_device_id device, cl_uint param, T fallback) {
    T value{};
    const auto status = api.fn().clGetDeviceInfo(device, param, sizeof(T), &value, nullptr);
    return status == CL_SUCCESS ? value : fallback;
}

std::string lower_ascii(std::string value) {
    std::transform(value.begin(), value.end(), value.begin(), [](unsigned char c) {
        return static_cast<char>(std::tolower(c));
    });
    return value;
}

bool contains_token(std::string_view haystack, std::string_view needle) {
    return lower_ascii(std::string(haystack)).find(lower_ascii(std::string(needle))) != std::string::npos;
}

GpuDeviceFamily classify_opencl_device(const OpenClDeviceSelection& selection) {
    const std::string combined = selection.vendor_name + " " + selection.device_name + " " + selection.device_version;
    if (contains_token(combined, "qualcomm") || contains_token(combined, "adreno")) {
        return GpuDeviceFamily::QualcommAdreno;
    }
    if (contains_token(combined, "broadcom") || contains_token(combined, "v3d") ||
        contains_token(combined, "videocore")) {
        return GpuDeviceFamily::BroadcomV3D;
    }
    return GpuDeviceFamily::Generic;
}

void reject_cpu_or_pocl(const OpenClDeviceSelection& selection) {
    OPENVINO_ASSERT(selection.is_gpu(),
                    "GFX OpenCL: selected device is not a GPU; CPU/accelerator fallback is forbidden");
    const std::string combined = selection.vendor_name + " " + selection.device_name + " " + selection.device_version +
                                 " " + selection.extensions;
    OPENVINO_ASSERT(!contains_token(combined, "pocl"),
                    "GFX OpenCL: PoCL/CPU style OpenCL device is not allowed for GFX inference");
}

}  // namespace

OpenClApi::OpenClApi() {
#if defined(_WIN32)
    OPENVINO_THROW("GFX OpenCL: Windows is not a supported OpenCL target for this plugin");
#else
    const char* candidates[] = {
        "libOpenCL.so",
        "libOpenCL.so.1",
        "/vendor/lib64/libOpenCL.so",
        "/vendor/lib/libOpenCL.so",
        "/system/vendor/lib64/libOpenCL.so",
        "/system/vendor/lib/libOpenCL.so",
    };
    for (const char* path : candidates) {
        m_library = dlopen(path, RTLD_NOW | RTLD_LOCAL);
        if (m_library) {
            m_library_path = path;
            break;
        }
    }
    OPENVINO_ASSERT(m_library,
                    "GFX OpenCL: failed to load vendor OpenCL runtime. "
                    "Only GPU OpenCL is allowed; CPU fallback is forbidden.");

    load_required_symbol(m_library, "clGetPlatformIDs", m_fn.clGetPlatformIDs);
    load_required_symbol(m_library, "clGetDeviceIDs", m_fn.clGetDeviceIDs);
    load_required_symbol(m_library, "clGetDeviceInfo", m_fn.clGetDeviceInfo);
    load_required_symbol(m_library, "clCreateContext", m_fn.clCreateContext);
    load_required_symbol(m_library, "clReleaseContext", m_fn.clReleaseContext);
    load_optional_symbol(m_library, "clCreateCommandQueueWithProperties", m_fn.clCreateCommandQueueWithProperties);
    load_optional_symbol(m_library, "clCreateCommandQueue", m_fn.clCreateCommandQueue);
    OPENVINO_ASSERT(m_fn.clCreateCommandQueueWithProperties || m_fn.clCreateCommandQueue,
                    "GFX OpenCL: no command queue creation entry point found");
    load_required_symbol(m_library, "clReleaseCommandQueue", m_fn.clReleaseCommandQueue);
    load_required_symbol(m_library, "clFinish", m_fn.clFinish);
    load_required_symbol(m_library, "clCreateBuffer", m_fn.clCreateBuffer);
    load_required_symbol(m_library, "clReleaseMemObject", m_fn.clReleaseMemObject);
    load_required_symbol(m_library, "clEnqueueMapBuffer", m_fn.clEnqueueMapBuffer);
    load_required_symbol(m_library, "clEnqueueUnmapMemObject", m_fn.clEnqueueUnmapMemObject);
    load_required_symbol(m_library, "clEnqueueWriteBuffer", m_fn.clEnqueueWriteBuffer);
    load_required_symbol(m_library, "clEnqueueReadBuffer", m_fn.clEnqueueReadBuffer);
    load_required_symbol(m_library, "clEnqueueCopyBuffer", m_fn.clEnqueueCopyBuffer);
    load_required_symbol(m_library, "clCreateProgramWithSource", m_fn.clCreateProgramWithSource);
    load_optional_symbol(m_library, "clCreateProgramWithBinary", m_fn.clCreateProgramWithBinary);
    load_required_symbol(m_library, "clBuildProgram", m_fn.clBuildProgram);
    load_required_symbol(m_library, "clGetProgramBuildInfo", m_fn.clGetProgramBuildInfo);
    load_required_symbol(m_library, "clGetProgramInfo", m_fn.clGetProgramInfo);
    load_required_symbol(m_library, "clRetainProgram", m_fn.clRetainProgram);
    load_required_symbol(m_library, "clReleaseProgram", m_fn.clReleaseProgram);
    load_required_symbol(m_library, "clCreateKernel", m_fn.clCreateKernel);
    load_required_symbol(m_library, "clSetKernelArg", m_fn.clSetKernelArg);
    load_required_symbol(m_library, "clEnqueueNDRangeKernel", m_fn.clEnqueueNDRangeKernel);
    load_required_symbol(m_library, "clReleaseKernel", m_fn.clReleaseKernel);
#endif
}

OpenClApi::~OpenClApi() {
#if !defined(_WIN32)
    // Keep the ICD loaded for the process lifetime. Some GPU OpenCL stacks
    // register their own process-exit teardown and are not safe to unload from
    // a plugin-local singleton destructor after contexts/programs were used.
    // The OS reclaims this handle at process exit.
#endif
}

const OpenClApi& OpenClApi::instance() {
    static const OpenClApi api;
    return api;
}

OpenClDeviceSelection select_opencl_gpu_device(const OpenClApi& api) {
    const auto& cl = api.fn();
    cl_uint platform_count = 0;
    opencl_check(cl.clGetPlatformIDs(0, nullptr, &platform_count), "clGetPlatformIDs(count)");
    OPENVINO_ASSERT(platform_count > 0, "GFX OpenCL: no OpenCL platforms found");

    std::vector<cl_platform_id> platforms(platform_count);
    opencl_check(cl.clGetPlatformIDs(platform_count, platforms.data(), nullptr), "clGetPlatformIDs(list)");

    for (auto platform : platforms) {
        cl_uint device_count = 0;
        const auto count_status = cl.clGetDeviceIDs(platform, CL_DEVICE_TYPE_GPU, 0, nullptr, &device_count);
        if (count_status != CL_SUCCESS || device_count == 0) {
            continue;
        }
        std::vector<cl_device_id> devices(device_count);
        const auto list_status = cl.clGetDeviceIDs(platform, CL_DEVICE_TYPE_GPU, device_count, devices.data(), nullptr);
        if (list_status != CL_SUCCESS) {
            continue;
        }
        for (auto device : devices) {
            OpenClDeviceSelection selection;
            selection.platform = platform;
            selection.device = device;
            selection.device_type = read_device_scalar<cl_device_type>(api, device, CL_DEVICE_TYPE, 0);
            selection.device_name = read_device_string(api, device, CL_DEVICE_NAME);
            selection.vendor_name = read_device_string(api, device, CL_DEVICE_VENDOR);
            selection.driver_version = read_device_string(api, device, CL_DRIVER_VERSION);
            selection.device_version = read_device_string(api, device, CL_DEVICE_VERSION);
            selection.extensions = read_device_string(api, device, CL_DEVICE_EXTENSIONS);
            selection.vendor_id = read_device_scalar<cl_uint>(api, device, CL_DEVICE_VENDOR_ID, 0);
            selection.compute_units = read_device_scalar<cl_uint>(api, device, CL_DEVICE_MAX_COMPUTE_UNITS, 1);
            selection.max_work_item_dimensions =
                read_device_scalar<cl_uint>(api, device, CL_DEVICE_MAX_WORK_ITEM_DIMENSIONS, 1);
            selection.max_work_group_size =
                read_device_scalar<size_t>(api, device, CL_DEVICE_MAX_WORK_GROUP_SIZE, 1);
            selection.mem_base_addr_align_bits =
                read_device_scalar<cl_uint>(api, device, CL_DEVICE_MEM_BASE_ADDR_ALIGN, 8);
            const auto dims = std::max<uint32_t>(selection.max_work_item_dimensions, 1);
            selection.max_work_item_sizes.assign(dims, 1);
            cl.clGetDeviceInfo(device,
                               CL_DEVICE_MAX_WORK_ITEM_SIZES,
                               selection.max_work_item_sizes.size() * sizeof(size_t),
                               selection.max_work_item_sizes.data(),
                               nullptr);
            reject_cpu_or_pocl(selection);
            return selection;
        }
    }

    OPENVINO_THROW("GFX OpenCL: no allowed GPU OpenCL device found");
}

GpuExecutionDeviceInfo make_opencl_execution_device_info(const OpenClDeviceSelection& selection) {
    GpuExecutionDeviceInfo info;
    info.backend = GpuBackend::OpenCL;
    info.device_family = classify_opencl_device(selection);
    info.device_key = std::string("opencl:") + gpu_device_family_name(info.device_family);
    if (!selection.vendor_name.empty()) {
        info.device_key += ":" + lower_ascii(selection.vendor_name);
    }
    info.device_name = selection.device_name.empty() ? "OpenCL GPU" : selection.device_name;
    info.vendor_id = selection.vendor_id;
    info.preferred_simd_width = std::max<uint32_t>(selection.compute_units, 1);
    info.subgroup_size = 1;
    info.max_total_threads_per_group = static_cast<uint32_t>(std::max<size_t>(selection.max_work_group_size, 1));
    for (size_t i = 0; i < info.max_threads_per_group.size(); ++i) {
        const size_t dim = i < selection.max_work_item_sizes.size() ? selection.max_work_item_sizes[i] : 1;
        info.max_threads_per_group[i] = static_cast<uint32_t>(std::max<size_t>(dim, 1));
    }
    info.min_storage_buffer_offset_alignment = std::max<uint64_t>(selection.mem_base_addr_align_bits / 8, 1);
    info.non_coherent_atom_size = 1;
    info.supports_shader_float16 = contains_token(selection.extensions, "cl_khr_fp16");
    info.supports_storage_buffer_16bit = info.supports_shader_float16;
    info.supports_shader_int8 = true;
    info.supports_storage_buffer_8bit = true;
    info.supports_conv_output_channel_blocking = info.device_family == GpuDeviceFamily::QualcommAdreno;
    info.supports_conv_channel_block_spatial_tiling = info.device_family == GpuDeviceFamily::QualcommAdreno;
    return info;
}

std::string opencl_error_string(cl_int error) {
    std::ostringstream os;
    os << "OpenCL error " << error;
    return os.str();
}

void opencl_check(cl_int status, const char* action) {
    OPENVINO_ASSERT(status == CL_SUCCESS,
                    "GFX OpenCL: ",
                    action ? action : "OpenCL call",
                    " failed: ",
                    opencl_error_string(status));
}

OpenClRuntimeContext::OpenClRuntimeContext()
    : m_api(OpenClApi::instance()),
      m_selection(select_opencl_gpu_device(m_api)) {
    reject_cpu_or_pocl(m_selection);
    cl_int status = CL_SUCCESS;
    m_context = m_api.fn().clCreateContext(nullptr, 1, &m_selection.device, nullptr, nullptr, &status);
    opencl_check(status, "clCreateContext");
    OPENVINO_ASSERT(m_context, "GFX OpenCL: clCreateContext returned null");
    if (m_api.fn().clCreateCommandQueueWithProperties) {
        m_queue = m_api.fn().clCreateCommandQueueWithProperties(m_context, m_selection.device, nullptr, &status);
    } else {
        m_queue = m_api.fn().clCreateCommandQueue(m_context, m_selection.device, 0, &status);
    }
    opencl_check(status, "clCreateCommandQueue");
    OPENVINO_ASSERT(m_queue, "GFX OpenCL: command queue creation returned null");
}

OpenClRuntimeContext::~OpenClRuntimeContext() {
    if (m_queue) {
        m_api.fn().clFinish(m_queue);
        m_api.fn().clReleaseCommandQueue(m_queue);
    }
    if (m_context) {
        m_api.fn().clReleaseContext(m_context);
    }
}

std::shared_ptr<OpenClRuntimeContext> OpenClRuntimeContext::instance() {
    static std::weak_ptr<OpenClRuntimeContext> weak;
    static std::mutex mutex;
    std::lock_guard<std::mutex> lock(mutex);
    auto ctx = weak.lock();
    if (!ctx) {
        ctx = std::shared_ptr<OpenClRuntimeContext>(new OpenClRuntimeContext());
        weak = ctx;
    }
    return ctx;
}

GpuExecutionDeviceInfo OpenClRuntimeContext::execution_device_info() const {
    return make_opencl_execution_device_info(m_selection);
}

void OpenClRuntimeContext::finish() const {
    opencl_check(m_api.fn().clFinish(m_queue), "clFinish");
}

}  // namespace gfx_plugin
}  // namespace ov
