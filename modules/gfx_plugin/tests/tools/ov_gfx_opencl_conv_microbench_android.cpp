#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <dlfcn.h>
#include <iomanip>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>
#include <tuple>
#include <vector>

namespace {

using cl_bool = uint32_t;
using cl_bitfield = uint64_t;
using cl_command_queue_properties = cl_bitfield;
using cl_context_properties = intptr_t;
using cl_device_type = cl_bitfield;
using cl_int = int32_t;
using cl_platform_info = uint32_t;
using cl_device_info = uint32_t;
using cl_command_queue_info = uint32_t;
using cl_program_build_info = uint32_t;
using cl_kernel_work_group_info = uint32_t;
using cl_profiling_info = uint32_t;
using cl_mem_flags = cl_bitfield;
using cl_ulong = uint64_t;
using cl_uint = uint32_t;
using cl_context = struct _cl_context*;
using cl_command_queue = struct _cl_command_queue*;
using cl_mem = struct _cl_mem*;
using cl_program = struct _cl_program*;
using cl_kernel = struct _cl_kernel*;
using cl_event = struct _cl_event*;
using cl_platform_id = struct _cl_platform_id*;
using cl_device_id = struct _cl_device_id*;

constexpr cl_int CL_SUCCESS = 0;
constexpr cl_device_type CL_DEVICE_TYPE_GPU = 1u << 2u;
constexpr cl_bool CL_TRUE_VALUE = 1;
constexpr cl_mem_flags CL_MEM_READ_ONLY = 1u << 2u;
constexpr cl_mem_flags CL_MEM_WRITE_ONLY = 1u << 1u;
constexpr cl_command_queue_properties CL_QUEUE_PROFILING_ENABLE = 1u << 1u;
constexpr cl_device_info CL_DEVICE_NAME = 0x102B;
constexpr cl_device_info CL_DRIVER_VERSION = 0x102D;
constexpr cl_device_info CL_DEVICE_VERSION = 0x102F;
constexpr cl_device_info CL_DEVICE_OPENCL_C_VERSION = 0x103D;
constexpr cl_device_info CL_DEVICE_MAX_COMPUTE_UNITS = 0x1002;
constexpr cl_device_info CL_DEVICE_MAX_WORK_GROUP_SIZE = 0x1004;
constexpr cl_device_info CL_DEVICE_LOCAL_MEM_SIZE = 0x1023;
constexpr cl_device_info CL_DEVICE_PROFILING_TIMER_RESOLUTION = 0x1025;
constexpr cl_program_build_info CL_PROGRAM_BUILD_LOG = 0x1183;
constexpr cl_kernel_work_group_info CL_KERNEL_WORK_GROUP_SIZE = 0x11B0;
constexpr cl_kernel_work_group_info CL_KERNEL_PREFERRED_WORK_GROUP_SIZE_MULTIPLE = 0x11B3;
constexpr cl_kernel_work_group_info CL_KERNEL_PRIVATE_MEM_SIZE = 0x11B4;
constexpr cl_profiling_info CL_PROFILING_COMMAND_START = 0x1282;
constexpr cl_profiling_info CL_PROFILING_COMMAND_END = 0x1283;

struct ConvCase {
    const char* name;
    int h;
    int w;
    int ic;
    int oc;
    int oh;
    int ow;
    int stride;
    int pad;
};

struct Variant {
    const char* name;
    const char* kernel_name;
    int spatial;
};

struct LocalSize {
    size_t x;
    size_t y;
    size_t z;
};

struct ClApi {
    void* handle = nullptr;

#define CL_FN(name) decltype(&name) name##_ = nullptr
    using clGetPlatformIDs_t = cl_int (*)(cl_uint, cl_platform_id*, cl_uint*);
    using clGetPlatformInfo_t = cl_int (*)(cl_platform_id, cl_platform_info, size_t, void*, size_t*);
    using clGetDeviceIDs_t = cl_int (*)(cl_platform_id, cl_device_type, cl_uint, cl_device_id*, cl_uint*);
    using clGetDeviceInfo_t = cl_int (*)(cl_device_id, cl_device_info, size_t, void*, size_t*);
    using clCreateContext_t = cl_context (*)(const cl_context_properties*, cl_uint, const cl_device_id*, void*, void*, cl_int*);
    using clCreateCommandQueue_t = cl_command_queue (*)(cl_context, cl_device_id, cl_command_queue_properties, cl_int*);
    using clCreateBuffer_t = cl_mem (*)(cl_context, cl_mem_flags, size_t, void*, cl_int*);
    using clCreateProgramWithSource_t = cl_program (*)(cl_context, cl_uint, const char**, const size_t*, cl_int*);
    using clBuildProgram_t = cl_int (*)(cl_program, cl_uint, const cl_device_id*, const char*, void*, void*);
    using clGetProgramBuildInfo_t = cl_int (*)(cl_program, cl_device_id, cl_program_build_info, size_t, void*, size_t*);
    using clCreateKernel_t = cl_kernel (*)(cl_program, const char*, cl_int*);
    using clSetKernelArg_t = cl_int (*)(cl_kernel, cl_uint, size_t, const void*);
    using clEnqueueWriteBuffer_t = cl_int (*)(cl_command_queue, cl_mem, cl_bool, size_t, size_t, const void*, cl_uint, const cl_event*, cl_event*);
    using clEnqueueReadBuffer_t = cl_int (*)(cl_command_queue, cl_mem, cl_bool, size_t, size_t, void*, cl_uint, const cl_event*, cl_event*);
    using clEnqueueNDRangeKernel_t = cl_int (*)(cl_command_queue, cl_kernel, cl_uint, const size_t*, const size_t*, const size_t*, cl_uint, const cl_event*, cl_event*);
    using clFinish_t = cl_int (*)(cl_command_queue);
    using clGetEventProfilingInfo_t = cl_int (*)(cl_event, cl_profiling_info, size_t, void*, size_t*);
    using clReleaseEvent_t = cl_int (*)(cl_event);
    using clReleaseMemObject_t = cl_int (*)(cl_mem);
    using clReleaseKernel_t = cl_int (*)(cl_kernel);
    using clReleaseProgram_t = cl_int (*)(cl_program);
    using clReleaseCommandQueue_t = cl_int (*)(cl_command_queue);
    using clReleaseContext_t = cl_int (*)(cl_context);
    using clGetKernelWorkGroupInfo_t = cl_int (*)(cl_kernel, cl_device_id, cl_kernel_work_group_info, size_t, void*, size_t*);

    clGetPlatformIDs_t clGetPlatformIDs = nullptr;
    clGetPlatformInfo_t clGetPlatformInfo = nullptr;
    clGetDeviceIDs_t clGetDeviceIDs = nullptr;
    clGetDeviceInfo_t clGetDeviceInfo = nullptr;
    clCreateContext_t clCreateContext = nullptr;
    clCreateCommandQueue_t clCreateCommandQueue = nullptr;
    clCreateBuffer_t clCreateBuffer = nullptr;
    clCreateProgramWithSource_t clCreateProgramWithSource = nullptr;
    clBuildProgram_t clBuildProgram = nullptr;
    clGetProgramBuildInfo_t clGetProgramBuildInfo = nullptr;
    clCreateKernel_t clCreateKernel = nullptr;
    clSetKernelArg_t clSetKernelArg = nullptr;
    clEnqueueWriteBuffer_t clEnqueueWriteBuffer = nullptr;
    clEnqueueReadBuffer_t clEnqueueReadBuffer = nullptr;
    clEnqueueNDRangeKernel_t clEnqueueNDRangeKernel = nullptr;
    clFinish_t clFinish = nullptr;
    clGetEventProfilingInfo_t clGetEventProfilingInfo = nullptr;
    clReleaseEvent_t clReleaseEvent = nullptr;
    clReleaseMemObject_t clReleaseMemObject = nullptr;
    clReleaseKernel_t clReleaseKernel = nullptr;
    clReleaseProgram_t clReleaseProgram = nullptr;
    clReleaseCommandQueue_t clReleaseCommandQueue = nullptr;
    clReleaseContext_t clReleaseContext = nullptr;
    clGetKernelWorkGroupInfo_t clGetKernelWorkGroupInfo = nullptr;
#undef CL_FN
};

template <typename T>
T load_symbol(void* handle, const char* name) {
    void* sym = dlsym(handle, name);
    if (!sym) {
        throw std::runtime_error(std::string("missing OpenCL symbol: ") + name);
    }
    return reinterpret_cast<T>(sym);
}

ClApi load_opencl() {
    ClApi api;
    api.handle = dlopen("libOpenCL.so", RTLD_NOW | RTLD_LOCAL);
    if (!api.handle) {
        api.handle = dlopen("/vendor/lib64/libOpenCL.so", RTLD_NOW | RTLD_LOCAL);
    }
    if (!api.handle) {
        throw std::runtime_error(std::string("failed to dlopen OpenCL: ") + dlerror());
    }
#define LOAD(name) api.name = load_symbol<ClApi::name##_t>(api.handle, #name)
    LOAD(clGetPlatformIDs);
    LOAD(clGetPlatformInfo);
    LOAD(clGetDeviceIDs);
    LOAD(clGetDeviceInfo);
    LOAD(clCreateContext);
    LOAD(clCreateCommandQueue);
    LOAD(clCreateBuffer);
    LOAD(clCreateProgramWithSource);
    LOAD(clBuildProgram);
    LOAD(clGetProgramBuildInfo);
    LOAD(clCreateKernel);
    LOAD(clSetKernelArg);
    LOAD(clEnqueueWriteBuffer);
    LOAD(clEnqueueReadBuffer);
    LOAD(clEnqueueNDRangeKernel);
    LOAD(clFinish);
    LOAD(clGetEventProfilingInfo);
    LOAD(clReleaseEvent);
    LOAD(clReleaseMemObject);
    LOAD(clReleaseKernel);
    LOAD(clReleaseProgram);
    LOAD(clReleaseCommandQueue);
    LOAD(clReleaseContext);
    LOAD(clGetKernelWorkGroupInfo);
#undef LOAD
    return api;
}

void check(cl_int err, const char* what) {
    if (err != CL_SUCCESS) {
        throw std::runtime_error(std::string(what) + " failed: " + std::to_string(err));
    }
}

size_t round_up(size_t value, size_t step) {
    return ((value + step - 1) / step) * step;
}

std::string get_string(const ClApi& cl, cl_device_id dev, cl_device_info info) {
    size_t bytes = 0;
    check(cl.clGetDeviceInfo(dev, info, 0, nullptr, &bytes), "clGetDeviceInfo(size)");
    std::string result(bytes, '\0');
    check(cl.clGetDeviceInfo(dev, info, bytes, result.data(), nullptr), "clGetDeviceInfo");
    if (!result.empty() && result.back() == '\0') {
        result.pop_back();
    }
    return result;
}

template <typename T>
T get_scalar(const ClApi& cl, cl_device_id dev, cl_device_info info) {
    T value{};
    check(cl.clGetDeviceInfo(dev, info, sizeof(T), &value, nullptr), "clGetDeviceInfo(scalar)");
    return value;
}

double event_ms(const ClApi& cl, cl_event event) {
    cl_ulong start = 0;
    cl_ulong end = 0;
    check(cl.clGetEventProfilingInfo(event, CL_PROFILING_COMMAND_START, sizeof(start), &start, nullptr),
          "clGetEventProfilingInfo(start)");
    check(cl.clGetEventProfilingInfo(event, CL_PROFILING_COMMAND_END, sizeof(end), &end, nullptr),
          "clGetEventProfilingInfo(end)");
    return static_cast<double>(end - start) / 1.0e6;
}

std::vector<float> make_input(const ConvCase& c) {
    std::vector<float> data(static_cast<size_t>(c.ic) * c.h * c.w);
    for (size_t i = 0; i < data.size(); ++i) {
        data[i] = static_cast<float>((static_cast<int>(i * 17 + 13) % 97) - 48) / 53.0f;
    }
    return data;
}

std::vector<float> make_weights(const ConvCase& c) {
    std::vector<float> data(static_cast<size_t>(c.oc) * c.ic * 3 * 3);
    for (size_t i = 0; i < data.size(); ++i) {
        data[i] = static_cast<float>((static_cast<int>(i * 31 + 7) % 43) - 21) / 67.0f;
    }
    return data;
}

std::vector<float> make_bias(const ConvCase& c) {
    std::vector<float> data(c.oc);
    for (size_t i = 0; i < data.size(); ++i) {
        data[i] = static_cast<float>((static_cast<int>(i * 11 + 5) % 19) - 9) / 31.0f;
    }
    return data;
}

std::vector<float> pack_oc16(const ConvCase& c, const std::vector<float>& weights) {
    const int oc_blocks = (c.oc + 15) / 16;
    std::vector<float> packed(static_cast<size_t>(oc_blocks) * c.ic * 3 * 3 * 16, 0.0f);
    for (int ocb = 0; ocb < oc_blocks; ++ocb) {
        for (int ci = 0; ci < c.ic; ++ci) {
            for (int kh = 0; kh < 3; ++kh) {
                for (int kw = 0; kw < 3; ++kw) {
                    for (int lane = 0; lane < 16; ++lane) {
                        const int oc = ocb * 16 + lane;
                        const size_t dst = (((static_cast<size_t>(ocb) * c.ic + ci) * 3 + kh) * 3 + kw) * 16 + lane;
                        if (oc < c.oc) {
                            const size_t src = ((static_cast<size_t>(oc) * c.ic + ci) * 3 + kh) * 3 + kw;
                            packed[dst] = weights[src];
                        }
                    }
                }
            }
        }
    }
    return packed;
}

const char* kSource = R"CLC(
__kernel void conv_scalar(__global const float* input,
                          __global const float* weights,
                          __global const float* bias,
                          __global float* output,
                          int H, int W, int IC, int OC, int OH, int OW,
                          int stride, int pad) {
  int x = get_global_id(0);
  int y = get_global_id(1);
  int oc = get_global_id(2);
  if (x >= OW || y >= OH || oc >= OC) return;
  float acc = bias[oc];
  for (int ci = 0; ci < IC; ++ci) {
    for (int kh = 0; kh < 3; ++kh) {
      int ih = y * stride + kh - pad;
      if (ih < 0 || ih >= H) continue;
      for (int kw = 0; kw < 3; ++kw) {
        int iw = x * stride + kw - pad;
        if (iw < 0 || iw >= W) continue;
        acc = mad(input[(ci * H + ih) * W + iw],
                  weights[((oc * IC + ci) * 3 + kh) * 3 + kw],
                  acc);
      }
    }
  }
  output[(oc * OH + y) * OW + x] = acc;
}

inline void store16(float16 acc, __global float* output, int oc_base, int OHW, int xy, int OC) {
  if (oc_base + 15 < OC) {
    __global float* out = output + oc_base * OHW + xy;
    out[0 * OHW] = acc.s0;
    out[1 * OHW] = acc.s1;
    out[2 * OHW] = acc.s2;
    out[3 * OHW] = acc.s3;
    out[4 * OHW] = acc.s4;
    out[5 * OHW] = acc.s5;
    out[6 * OHW] = acc.s6;
    out[7 * OHW] = acc.s7;
    out[8 * OHW] = acc.s8;
    out[9 * OHW] = acc.s9;
    out[10 * OHW] = acc.sa;
    out[11 * OHW] = acc.sb;
    out[12 * OHW] = acc.sc;
    out[13 * OHW] = acc.sd;
    out[14 * OHW] = acc.se;
    out[15 * OHW] = acc.sf;
  } else {
    float tmp[16];
    vstore16(acc, 0, tmp);
    for (int lane = 0; lane < 16; ++lane) {
      int oc = oc_base + lane;
      if (oc < OC) output[oc * OHW + xy] = tmp[lane];
    }
  }
}

__kernel __attribute__((vec_type_hint(float16))) __attribute__((work_group_size_hint(8,8,1)))
void conv_oc16_ci4_ptr(__global const float* input,
                       __global const float* packed_weights,
                       __global const float* bias,
                       __global float* output,
                       int H, int W, int IC, int OC, int OH, int OW,
                       int stride, int pad) {
  int x = get_global_id(0);
  int y = get_global_id(1);
  int ocb = get_global_id(2);
  int oc_base = ocb * 16;
  if (x >= OW || y >= OH || oc_base >= OC) return;
  const int HW = H * W;
  const int OHW = OH * OW;
  const int xy = y * OW + x;
  const int y0 = y * stride - pad;
  const int x0 = x * stride - pad;
  const int ocb_ic_base = ocb * IC;
  float16 acc = vload16(0, bias + oc_base);
  for (int ci0 = 0; ci0 < IC; ci0 += 4) {
    const int input_base0 = (ci0 + 0) * HW;
    const int input_base1 = (ci0 + 1) * HW;
    const int input_base2 = (ci0 + 2) * HW;
    const int input_base3 = (ci0 + 3) * HW;
    const int w_ci_base = (ocb_ic_base + ci0) * 9 * 16;
    for (int kh = 0; kh < 3; ++kh) {
      int ih = y0 + kh;
      if (ih < 0 || ih >= H) continue;
      const int input_row_base = ih * W;
      const int w_kh_base = w_ci_base + kh * 3 * 16;
      for (int kw = 0; kw < 3; ++kw) {
        int iw = x0 + kw;
        if (iw < 0 || iw >= W) continue;
        const int in_base = input_row_base + iw;
        __global const float* w_ptr = packed_weights + w_kh_base + kw * 16;
        float in0 = input[input_base0 + in_base];
        acc = mad((float16)(in0), vload16(0, w_ptr + 0 * 9 * 16), acc);
        if (ci0 + 1 < IC) acc = mad((float16)(input[input_base1 + in_base]), vload16(0, w_ptr + 1 * 9 * 16), acc);
        if (ci0 + 2 < IC) acc = mad((float16)(input[input_base2 + in_base]), vload16(0, w_ptr + 2 * 9 * 16), acc);
        if (ci0 + 3 < IC) acc = mad((float16)(input[input_base3 + in_base]), vload16(0, w_ptr + 3 * 9 * 16), acc);
      }
    }
  }
  store16(acc, output, oc_base, OHW, xy, OC);
}

__kernel __attribute__((vec_type_hint(float16))) __attribute__((work_group_size_hint(8,8,1)))
void conv_oc16_ci4_x2(__global const float* input,
                      __global const float* packed_weights,
                      __global const float* bias,
                      __global float* output,
                      int H, int W, int IC, int OC, int OH, int OW,
                      int stride, int pad) {
  int x = get_global_id(0) * 2;
  int y = get_global_id(1);
  int ocb = get_global_id(2);
  int oc_base = ocb * 16;
  if (x >= OW || y >= OH || oc_base >= OC) return;
  const int HW = H * W;
  const int OHW = OH * OW;
  const int y0 = y * stride - pad;
  const int x0 = x * stride - pad;
  const int ocb_ic_base = ocb * IC;
  float16 acc0 = vload16(0, bias + oc_base);
  float16 acc1 = acc0;
  for (int ci0 = 0; ci0 < IC; ci0 += 4) {
    const int input_base0 = (ci0 + 0) * HW;
    const int input_base1 = (ci0 + 1) * HW;
    const int input_base2 = (ci0 + 2) * HW;
    const int input_base3 = (ci0 + 3) * HW;
    const int w_ci_base = (ocb_ic_base + ci0) * 9 * 16;
    for (int kh = 0; kh < 3; ++kh) {
      int ih = y0 + kh;
      if (ih < 0 || ih >= H) continue;
      const int input_row_base = ih * W;
      const int w_kh_base = w_ci_base + kh * 3 * 16;
      for (int kw = 0; kw < 3; ++kw) {
        int iw0 = x0 + kw;
        int iw1 = iw0 + stride;
        __global const float* w_ptr = packed_weights + w_kh_base + kw * 16;
        const float16 w0 = vload16(0, w_ptr + 0 * 9 * 16);
        const float16 w1 = vload16(0, w_ptr + 1 * 9 * 16);
        const float16 w2 = vload16(0, w_ptr + 2 * 9 * 16);
        const float16 w3 = vload16(0, w_ptr + 3 * 9 * 16);
        if (iw0 >= 0 && iw0 < W) {
          int in_base0 = input_row_base + iw0;
          acc0 = mad((float16)(input[input_base0 + in_base0]), w0, acc0);
          if (ci0 + 1 < IC) acc0 = mad((float16)(input[input_base1 + in_base0]), w1, acc0);
          if (ci0 + 2 < IC) acc0 = mad((float16)(input[input_base2 + in_base0]), w2, acc0);
          if (ci0 + 3 < IC) acc0 = mad((float16)(input[input_base3 + in_base0]), w3, acc0);
        }
        if (x + 1 < OW && iw1 >= 0 && iw1 < W) {
          int in_base1 = input_row_base + iw1;
          acc1 = mad((float16)(input[input_base0 + in_base1]), w0, acc1);
          if (ci0 + 1 < IC) acc1 = mad((float16)(input[input_base1 + in_base1]), w1, acc1);
          if (ci0 + 2 < IC) acc1 = mad((float16)(input[input_base2 + in_base1]), w2, acc1);
          if (ci0 + 3 < IC) acc1 = mad((float16)(input[input_base3 + in_base1]), w3, acc1);
        }
      }
    }
  }
  store16(acc0, output, oc_base, OHW, y * OW + x, OC);
  if (x + 1 < OW) {
    store16(acc1, output, oc_base, OHW, y * OW + x + 1, OC);
  }
}
)CLC";

cl_mem create_buffer(const ClApi& cl, cl_context ctx, cl_mem_flags flags, size_t bytes) {
    cl_int err = CL_SUCCESS;
    cl_mem mem = cl.clCreateBuffer(ctx, flags, bytes, nullptr, &err);
    check(err, "clCreateBuffer");
    if (!mem) {
        throw std::runtime_error("clCreateBuffer returned null");
    }
    return mem;
}

void set_common_args(const ClApi& cl, cl_kernel kernel, cl_mem input, cl_mem weights, cl_mem bias, cl_mem output,
                     const ConvCase& c) {
    int idx = 0;
    check(cl.clSetKernelArg(kernel, idx++, sizeof(cl_mem), &input), "clSetKernelArg(input)");
    check(cl.clSetKernelArg(kernel, idx++, sizeof(cl_mem), &weights), "clSetKernelArg(weights)");
    check(cl.clSetKernelArg(kernel, idx++, sizeof(cl_mem), &bias), "clSetKernelArg(bias)");
    check(cl.clSetKernelArg(kernel, idx++, sizeof(cl_mem), &output), "clSetKernelArg(output)");
    check(cl.clSetKernelArg(kernel, idx++, sizeof(int), &c.h), "clSetKernelArg(H)");
    check(cl.clSetKernelArg(kernel, idx++, sizeof(int), &c.w), "clSetKernelArg(W)");
    check(cl.clSetKernelArg(kernel, idx++, sizeof(int), &c.ic), "clSetKernelArg(IC)");
    check(cl.clSetKernelArg(kernel, idx++, sizeof(int), &c.oc), "clSetKernelArg(OC)");
    check(cl.clSetKernelArg(kernel, idx++, sizeof(int), &c.oh), "clSetKernelArg(OH)");
    check(cl.clSetKernelArg(kernel, idx++, sizeof(int), &c.ow), "clSetKernelArg(OW)");
    check(cl.clSetKernelArg(kernel, idx++, sizeof(int), &c.stride), "clSetKernelArg(stride)");
    check(cl.clSetKernelArg(kernel, idx++, sizeof(int), &c.pad), "clSetKernelArg(pad)");
}

double run_kernel(const ClApi& cl,
                  cl_command_queue queue,
                  cl_kernel kernel,
                  const ConvCase& c,
                  int spatial,
                  const LocalSize& local) {
    const size_t global[3] = {
        round_up(static_cast<size_t>((c.ow + spatial - 1) / spatial), local.x),
        round_up(static_cast<size_t>(c.oh), local.y),
        round_up(static_cast<size_t>((c.oc + 15) / 16), local.z),
    };
    const size_t lws[3] = {local.x, local.y, local.z};
    cl_event event = nullptr;
    check(cl.clEnqueueNDRangeKernel(queue, kernel, 3, nullptr, global, lws, 0, nullptr, &event),
          "clEnqueueNDRangeKernel");
    check(cl.clFinish(queue), "clFinish");
    const double ms = event_ms(cl, event);
    check(cl.clReleaseEvent(event), "clReleaseEvent");
    return ms;
}

double run_scalar_kernel(const ClApi& cl,
                         cl_command_queue queue,
                         cl_kernel kernel,
                         const ConvCase& c,
                         const LocalSize& local) {
    const size_t global[3] = {
        round_up(static_cast<size_t>(c.ow), local.x),
        round_up(static_cast<size_t>(c.oh), local.y),
        round_up(static_cast<size_t>(c.oc), local.z),
    };
    const size_t lws[3] = {local.x, local.y, local.z};
    cl_event event = nullptr;
    check(cl.clEnqueueNDRangeKernel(queue, kernel, 3, nullptr, global, lws, 0, nullptr, &event),
          "clEnqueueNDRangeKernel(scalar)");
    check(cl.clFinish(queue), "clFinish(scalar)");
    const double ms = event_ms(cl, event);
    check(cl.clReleaseEvent(event), "clReleaseEvent(scalar)");
    return ms;
}

double max_abs_diff(const std::vector<float>& a, const std::vector<float>& b) {
    double max_diff = 0.0;
    for (size_t i = 0; i < a.size(); ++i) {
        max_diff = std::max(max_diff, static_cast<double>(std::abs(a[i] - b[i])));
    }
    return max_diff;
}

std::vector<ConvCase> cases() {
    return {
        {"yolo26x_model_1_conv_s2", 320, 320, 96, 192, 160, 160, 2, 1},
        {"yolo26x_model_3_conv_s2", 160, 160, 384, 384, 80, 80, 2, 1},
        {"yolo26x_model_5_conv_s2", 80, 80, 768, 768, 40, 40, 2, 1},
        {"yolo26x_c2_48_48_k3_160", 160, 160, 48, 48, 160, 160, 1, 1},
        {"yolo26x_c4_96_96_k3_80", 80, 80, 96, 96, 80, 80, 1, 1},
        {"yolo26x_c6_192_192_k3_40", 40, 40, 192, 192, 40, 40, 1, 1},
    };
}

}  // namespace

int main() {
    try {
        ClApi cl = load_opencl();

        cl_uint platform_count = 0;
        check(cl.clGetPlatformIDs(0, nullptr, &platform_count), "clGetPlatformIDs(count)");
        std::vector<cl_platform_id> platforms(platform_count);
        check(cl.clGetPlatformIDs(platform_count, platforms.data(), nullptr), "clGetPlatformIDs");

        cl_device_id device = nullptr;
        for (cl_platform_id platform : platforms) {
            cl_uint device_count = 0;
            const cl_int err = cl.clGetDeviceIDs(platform, CL_DEVICE_TYPE_GPU, 0, nullptr, &device_count);
            if (err != CL_SUCCESS || device_count == 0) {
                continue;
            }
            std::vector<cl_device_id> devices(device_count);
            check(cl.clGetDeviceIDs(platform, CL_DEVICE_TYPE_GPU, device_count, devices.data(), nullptr),
                  "clGetDeviceIDs");
            device = devices.front();
            break;
        }
        if (!device) {
            throw std::runtime_error("no OpenCL GPU device found");
        }

        std::cout << "device_name," << get_string(cl, device, CL_DEVICE_NAME) << "\n";
        std::cout << "device_version," << get_string(cl, device, CL_DEVICE_VERSION) << "\n";
        std::cout << "opencl_c_version," << get_string(cl, device, CL_DEVICE_OPENCL_C_VERSION) << "\n";
        std::cout << "driver_version," << get_string(cl, device, CL_DRIVER_VERSION) << "\n";
        std::cout << "max_compute_units," << get_scalar<cl_uint>(cl, device, CL_DEVICE_MAX_COMPUTE_UNITS) << "\n";
        std::cout << "max_work_group_size," << get_scalar<size_t>(cl, device, CL_DEVICE_MAX_WORK_GROUP_SIZE) << "\n";
        std::cout << "local_mem_size," << get_scalar<cl_ulong>(cl, device, CL_DEVICE_LOCAL_MEM_SIZE) << "\n";
        std::cout << "profiling_timer_resolution_ns,"
                  << get_scalar<size_t>(cl, device, CL_DEVICE_PROFILING_TIMER_RESOLUTION) << "\n";

        cl_int err = CL_SUCCESS;
        cl_context ctx = cl.clCreateContext(nullptr, 1, &device, nullptr, nullptr, &err);
        check(err, "clCreateContext");
        cl_command_queue queue = cl.clCreateCommandQueue(ctx, device, CL_QUEUE_PROFILING_ENABLE, &err);
        check(err, "clCreateCommandQueue");

        const char* src = kSource;
        const size_t src_len = std::strlen(kSource);
        cl_program program = cl.clCreateProgramWithSource(ctx, 1, &src, &src_len, &err);
        check(err, "clCreateProgramWithSource");
        err = cl.clBuildProgram(program, 1, &device, "-cl-fast-relaxed-math", nullptr, nullptr);
        if (err != CL_SUCCESS) {
            size_t log_size = 0;
            cl.clGetProgramBuildInfo(program, device, CL_PROGRAM_BUILD_LOG, 0, nullptr, &log_size);
            std::string log(log_size, '\0');
            if (log_size) {
                cl.clGetProgramBuildInfo(program, device, CL_PROGRAM_BUILD_LOG, log_size, log.data(), nullptr);
            }
            throw std::runtime_error("clBuildProgram failed: " + std::to_string(err) + "\n" + log);
        }

        cl_kernel scalar = cl.clCreateKernel(program, "conv_scalar", &err);
        check(err, "clCreateKernel(scalar)");
        cl_kernel ptr = cl.clCreateKernel(program, "conv_oc16_ci4_ptr", &err);
        check(err, "clCreateKernel(ptr)");
        cl_kernel x2 = cl.clCreateKernel(program, "conv_oc16_ci4_x2", &err);
        check(err, "clCreateKernel(x2)");

        size_t kernel_wg = 0;
        size_t kernel_pref = 0;
        cl_ulong private_bytes = 0;
        check(cl.clGetKernelWorkGroupInfo(ptr, device, CL_KERNEL_WORK_GROUP_SIZE, sizeof(kernel_wg), &kernel_wg, nullptr),
              "clGetKernelWorkGroupInfo(wg)");
        check(cl.clGetKernelWorkGroupInfo(ptr, device, CL_KERNEL_PREFERRED_WORK_GROUP_SIZE_MULTIPLE, sizeof(kernel_pref),
                                          &kernel_pref, nullptr),
              "clGetKernelWorkGroupInfo(pref)");
        check(cl.clGetKernelWorkGroupInfo(ptr, device, CL_KERNEL_PRIVATE_MEM_SIZE, sizeof(private_bytes), &private_bytes,
                                          nullptr),
              "clGetKernelWorkGroupInfo(private)");
        std::cout << "kernel_ptr_max_wg," << kernel_wg << "\n";
        std::cout << "kernel_ptr_pref_wg_multiple," << kernel_pref << "\n";
        std::cout << "kernel_ptr_private_mem," << private_bytes << "\n";

        const std::vector<Variant> variants = {
            {"oc16_ci4_ptr", "conv_oc16_ci4_ptr", 1},
            {"oc16_ci4_x2", "conv_oc16_ci4_x2", 2},
        };
        const std::vector<LocalSize> locals = {{8, 8, 1}, {16, 4, 1}, {4, 8, 1}, {16, 8, 1}};

        std::cout << "case,variant,local_size,exec_median_ms,exec_min_ms,max_abs_diff\n";
        for (const auto& c : cases()) {
            const auto input = make_input(c);
            const auto weights = make_weights(c);
            const auto packed = pack_oc16(c, weights);
            const auto bias = make_bias(c);
            const size_t input_bytes = input.size() * sizeof(float);
            const size_t weights_bytes = weights.size() * sizeof(float);
            const size_t packed_bytes = packed.size() * sizeof(float);
            const size_t bias_bytes = bias.size() * sizeof(float);
            const size_t output_elems = static_cast<size_t>(c.oc) * c.oh * c.ow;
            const size_t output_bytes = output_elems * sizeof(float);

            cl_mem input_buf = create_buffer(cl, ctx, CL_MEM_READ_ONLY, input_bytes);
            cl_mem weights_buf = create_buffer(cl, ctx, CL_MEM_READ_ONLY, weights_bytes);
            cl_mem packed_buf = create_buffer(cl, ctx, CL_MEM_READ_ONLY, packed_bytes);
            cl_mem bias_buf = create_buffer(cl, ctx, CL_MEM_READ_ONLY, bias_bytes);
            cl_mem ref_buf = create_buffer(cl, ctx, CL_MEM_WRITE_ONLY, output_bytes);
            cl_mem out_buf = create_buffer(cl, ctx, CL_MEM_WRITE_ONLY, output_bytes);

            check(cl.clEnqueueWriteBuffer(queue, input_buf, CL_TRUE_VALUE, 0, input_bytes, input.data(), 0, nullptr, nullptr),
                  "clEnqueueWriteBuffer(input)");
            check(cl.clEnqueueWriteBuffer(queue, weights_buf, CL_TRUE_VALUE, 0, weights_bytes, weights.data(), 0, nullptr,
                                          nullptr),
                  "clEnqueueWriteBuffer(weights)");
            check(cl.clEnqueueWriteBuffer(queue, packed_buf, CL_TRUE_VALUE, 0, packed_bytes, packed.data(), 0, nullptr,
                                          nullptr),
                  "clEnqueueWriteBuffer(packed)");
            check(cl.clEnqueueWriteBuffer(queue, bias_buf, CL_TRUE_VALUE, 0, bias_bytes, bias.data(), 0, nullptr, nullptr),
                  "clEnqueueWriteBuffer(bias)");

            set_common_args(cl, scalar, input_buf, weights_buf, bias_buf, ref_buf, c);
            const LocalSize scalar_local{8, 8, 1};
            (void)run_scalar_kernel(cl, queue, scalar, c, scalar_local);

            std::vector<float> ref(output_elems);
            check(cl.clEnqueueReadBuffer(queue, ref_buf, CL_TRUE_VALUE, 0, output_bytes, ref.data(), 0, nullptr, nullptr),
                  "clEnqueueReadBuffer(ref)");

            for (const auto& variant : variants) {
                cl_kernel kernel = std::string(variant.kernel_name) == "conv_oc16_ci4_x2" ? x2 : ptr;
                set_common_args(cl, kernel, input_buf, packed_buf, bias_buf, out_buf, c);
                for (const auto& local : locals) {
                    std::vector<double> times;
                    for (int i = 0; i < 3; ++i) {
                        times.push_back(run_kernel(cl, queue, kernel, c, variant.spatial, local));
                    }
                    std::vector<float> out(output_elems);
                    check(cl.clEnqueueReadBuffer(queue, out_buf, CL_TRUE_VALUE, 0, output_bytes, out.data(), 0, nullptr,
                                                 nullptr),
                          "clEnqueueReadBuffer(out)");
                    std::sort(times.begin(), times.end());
                    const double diff = max_abs_diff(ref, out);
                    std::cout << c.name << "," << variant.name << "," << local.x << "x" << local.y << "x" << local.z
                              << "," << std::fixed << std::setprecision(3) << times[times.size() / 2] << ","
                              << times.front() << "," << std::setprecision(9) << diff << "\n";
                }
            }

            cl.clReleaseMemObject(out_buf);
            cl.clReleaseMemObject(ref_buf);
            cl.clReleaseMemObject(bias_buf);
            cl.clReleaseMemObject(packed_buf);
            cl.clReleaseMemObject(weights_buf);
            cl.clReleaseMemObject(input_buf);
        }

        cl.clReleaseKernel(x2);
        cl.clReleaseKernel(ptr);
        cl.clReleaseKernel(scalar);
        cl.clReleaseProgram(program);
        cl.clReleaseCommandQueue(queue);
        cl.clReleaseContext(ctx);
        return 0;
    } catch (const std::exception& e) {
        std::cerr << "fatal: " << e.what() << "\n";
        return 1;
    }
}
