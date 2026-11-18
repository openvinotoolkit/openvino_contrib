// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include "pp_runtime.hpp"

#include <cstdio>
#include <fstream>
#include <initializer_list>
#include <memory>
#include <sstream>
#include <stdexcept>
#include <type_traits>

#include "pp_postprocess.hpp"
#include "pp_tuning_config_generated.hpp"
#include "pp_voxelization.hpp"

namespace PpExtension {
namespace {

constexpr const char* kFp32RoundingOption = "-cl-fp32-correctly-rounded-divide-sqrt";
constexpr cl_uint kPlatformCapacity = 8;
constexpr cl_uint kOpenClWorkDimensions = 1;
constexpr cl_uint kSingleDevice = 1;
constexpr size_t kDriverSelectedLocalSize = 0;
constexpr size_t kCandidateHeaderRows = 1;

void check_status(cl_int status, const char* operation) {
    if (status != CL_SUCCESS)
        throw std::runtime_error(std::string(operation) + " failed: " + std::to_string(status));
}

std::string read_source_file(const std::string& path) {
    std::ifstream source_file(path);
    if (!source_file) throw std::runtime_error("cannot read " + path);
    std::stringstream text;
    text << source_file.rdbuf();
    return text.str();
}

std::string with_rounding_option(const char* options) {
    return std::string(options) + " " + kFp32RoundingOption;
}

// OpenCL handle wrappers, released by the unique_ptr deleter.
template <typename Handle, cl_int (*Release)(Handle)>
struct ClReleaser {
    void operator()(Handle handle) const { Release(handle); }
};

template <typename Handle, cl_int (*Release)(Handle)>
using ClHandle = std::unique_ptr<typename std::remove_pointer<Handle>::type, ClReleaser<Handle, Release>>;

using ContextPtr = ClHandle<cl_context, clReleaseContext>;
using QueuePtr = ClHandle<cl_command_queue, clReleaseCommandQueue>;
using ProgramPtr = ClHandle<cl_program, clReleaseProgram>;
using KernelPtr = ClHandle<cl_kernel, clReleaseKernel>;

}  // namespace

struct PointPillarsRuntime::Impl {
    // Declared before the OpenVINO members below so they outlive them; OpenVINO holds its own references.
    ContextPtr context;
    cl_device_id device = nullptr;
    QueuePtr queue;
    std::vector<ProgramPtr> kernel_programs;

    KernelPtr scatter_kernel;
    KernelPtr finalize_kernel;
    KernelPtr cells_kernel;
    KernelPtr scores_kernel;
    KernelPtr candidates_kernel;
    KernelPtr nms_kernel;
    KernelPtr gather_kernel;

    // Unified shared memory entry points.
    clSetKernelArgMemPointerINTEL_fn set_kernel_arg_mem_pointer = nullptr;
    clEnqueueMemcpyINTEL_fn enqueue_memcpy = nullptr;

    // OpenVINO model and inference request.
    ov::Core core;
    std::unique_ptr<ov::intel_gpu::ocl::ClContext> remote_context;
    ov::CompiledModel compiled_model;
    ov::InferRequest infer_request;

    // Device tensors and scratch buffers.
    std::vector<ov::intel_gpu::ocl::USMTensor> owned_tensors;
    void* point_buffer = nullptr;
    void* point_count_buffer = nullptr;
    void* metadata_buffer = nullptr;
    void* pillar_features = nullptr;
    void* cell_to_pillar = nullptr;
    void* classification_logits = nullptr;
    void* box_regression = nullptr;
    void* direction_logits = nullptr;
    void* selection_bitmap = nullptr;
    void* candidate_records = nullptr;
    void* nms_mask = nullptr;
    void* detections = nullptr;

    std::vector<float> host_detections;
    std::string kernel_dir;

    void* allocate_scratch(size_t bytes) {
        owned_tensors.push_back(remote_context->create_usm_device_tensor(ov::element::u8, {bytes}));
        return owned_tensors.back().get();
    }

    void* bind_tensor(const std::string& port, const ov::Output<const ov::Node>& tensor_info) {
        owned_tensors.push_back(remote_context->create_usm_device_tensor(tensor_info.get_element_type(),
                                                                        tensor_info.get_shape()));
        infer_request.set_tensor(port, owned_tensors.back());
        return owned_tensors.back().get();
    }

    KernelPtr build_kernel(const std::vector<std::string>& files, const char* entry,
                           const std::string& options) {
        std::vector<std::string> texts;
        std::vector<const char*> pointers;
        std::vector<size_t> lengths;
        for (const auto& file : files) {
            texts.push_back(read_source_file(kernel_dir + "/" + file));
            pointers.push_back(texts.back().c_str());
            lengths.push_back(texts.back().size());
        }

        cl_int status = CL_SUCCESS;
        ProgramPtr program(clCreateProgramWithSource(context.get(), static_cast<cl_uint>(pointers.size()),
                                                     pointers.data(), lengths.data(), &status));
        check_status(status, "clCreateProgramWithSource");
        kernel_programs.push_back(std::move(program));
        cl_program raw_program = kernel_programs.back().get();
        if (clBuildProgram(raw_program, kSingleDevice, &device, options.c_str(), nullptr, nullptr) != CL_SUCCESS) {
            size_t size = 0;
            clGetProgramBuildInfo(raw_program, device, CL_PROGRAM_BUILD_LOG, 0, nullptr, &size);
            std::string log(size, '\0');
            clGetProgramBuildInfo(raw_program, device, CL_PROGRAM_BUILD_LOG, size, log.data(), nullptr);
            throw std::runtime_error(std::string("building ") + entry + ":\n" + log);
        }

        KernelPtr kernel(clCreateKernel(raw_program, entry, &status));
        check_status(status, "clCreateKernel");
        return kernel;
    }

    void launch_kernel(const KernelPtr& kernel, std::initializer_list<void*> arguments, size_t global_size,
                       size_t local_size) {
        cl_uint argument_index = 0;
        for (void* argument : arguments)
            check_status(set_kernel_arg_mem_pointer(kernel.get(), argument_index++, argument),
                         "clSetKernelArgMemPointerINTEL");
        const size_t* local_size_ptr = local_size ? &local_size : nullptr;
        check_status(clEnqueueNDRangeKernel(queue.get(), kernel.get(), kOpenClWorkDimensions, nullptr, &global_size,
                                            local_size_ptr, 0, nullptr, nullptr),
                     "clEnqueueNDRangeKernel");
    }
};

PointPillarsRuntime::PointPillarsRuntime(const std::string& model_xml, const std::string& kernel_dir)
    : impl_(new Impl) {
    Impl& self = *impl_;
    self.kernel_dir = kernel_dir;

    // Create the shared OpenCL context and in-order queue.
    cl_platform_id platforms[kPlatformCapacity];
    cl_uint platform_count = 0;
    check_status(clGetPlatformIDs(kPlatformCapacity, platforms, &platform_count), "clGetPlatformIDs");
    cl_platform_id platform = nullptr;
    for (cl_uint platform_index = 0; platform_index < platform_count && !self.device; ++platform_index) {
        if (clGetDeviceIDs(platforms[platform_index], CL_DEVICE_TYPE_GPU, kSingleDevice, &self.device, nullptr) ==
            CL_SUCCESS)
            platform = platforms[platform_index];
    }
    if (!self.device) throw std::runtime_error("no OpenCL GPU device found");

    cl_int status = CL_SUCCESS;
    self.context.reset(clCreateContext(nullptr, kSingleDevice, &self.device, nullptr, nullptr, &status));
    check_status(status, "clCreateContext");
    self.queue.reset(clCreateCommandQueueWithProperties(self.context.get(), self.device, nullptr, &status));
    check_status(status, "clCreateCommandQueueWithProperties");

    self.set_kernel_arg_mem_pointer = reinterpret_cast<clSetKernelArgMemPointerINTEL_fn>(
        clGetExtensionFunctionAddressForPlatform(platform, "clSetKernelArgMemPointerINTEL"));
    self.enqueue_memcpy = reinterpret_cast<clEnqueueMemcpyINTEL_fn>(
        clGetExtensionFunctionAddressForPlatform(platform, "clEnqueueMemcpyINTEL"));
    if (!self.set_kernel_arg_mem_pointer || !self.enqueue_memcpy)
        throw std::runtime_error("driver does not expose unified shared memory");

    self.remote_context.reset(new ov::intel_gpu::ocl::ClContext(self.core, self.queue.get()));
    self.compiled_model = self.core.compile_model(self.core.read_model(model_xml), *self.remote_context,
                                                  {ov::hint::execution_mode(ov::hint::ExecutionMode::ACCURACY)});
    self.infer_request = self.compiled_model.create_infer_request();

    // Validate the model input capacity.
    const ov::Shape pillar_shape = self.compiled_model.input("pillars").get_shape();
    if (static_cast<int64_t>(pillar_shape[0]) != PP_MAX_VOXELS ||
        static_cast<int64_t>(pillar_shape[1]) != PP_MAX_POINTS_PER_VOXEL)
        throw std::runtime_error("model expects " + std::to_string(pillar_shape[0]) + " pillars of " +
                                 std::to_string(pillar_shape[1]) + " points, kernels were built for " +
                                 std::to_string(PP_MAX_VOXELS) + " of " +
                                 std::to_string(PP_MAX_POINTS_PER_VOXEL));

    self.pillar_features = self.bind_tensor("pillars", self.compiled_model.input("pillars"));
    self.cell_to_pillar = self.bind_tensor("cell_pillar", self.compiled_model.input("cell_pillar"));
    self.classification_logits = self.bind_tensor("cls", self.compiled_model.output("cls"));
    self.box_regression = self.bind_tensor("box", self.compiled_model.output("box"));
    self.direction_logits = self.bind_tensor("dir", self.compiled_model.output("dir"));

    self.point_buffer = self.allocate_scratch(PP_MAX_POINTS * PP_NUM_POINT_FEATURES * sizeof(float));
    self.point_count_buffer = self.allocate_scratch(sizeof(int32_t));
    self.metadata_buffer = self.allocate_scratch(PP_META_SIZE * sizeof(int32_t));
    self.selection_bitmap = self.allocate_scratch(PP_SELECT_WORDS * sizeof(int32_t));
    self.candidate_records =
        self.allocate_scratch((PP_MAX_CANDIDATES + kCandidateHeaderRows) * PP_DET_CHANNELS * sizeof(float));
    self.nms_mask = self.allocate_scratch(PP_MAX_CANDIDATES * PP_NMS_COL_BLOCKS * sizeof(int32_t));
    self.detections = self.allocate_scratch(PP_MAX_DETECTIONS * PP_DET_CHANNELS * sizeof(float));
    self.host_detections.resize(PP_MAX_DETECTIONS * PP_DET_CHANNELS);

    // Build kernels with the runtime tensor types and generated launch sizes.
    const std::string voxel_common = "pp_voxelization_common.cl";
    const std::string tuning_config = "pp_tuning_config_generated.cl";
    const std::string pillar_config = "pp_pillar_config_generated.cl";
    self.scatter_kernel = self.build_kernel({pillar_config, tuning_config, voxel_common, "pp_voxelization_scatter.cl"},
                                            "pp_voxelization_scatter",
                                            with_rounding_option("-DINPUT0_TYPE=float -DINPUT1_TYPE=int "
                                                                 "-DOUTPUT0_TYPE=float"));
    self.finalize_kernel =
        self.build_kernel({pillar_config, tuning_config, voxel_common, "pp_voxelization_features.cl"},
                          "pp_voxelization_finalize",
                          with_rounding_option("-DINPUT0_TYPE=float -DINPUT1_TYPE=float "
                                               "-DOUTPUT0_TYPE=half"));
    self.cells_kernel = self.build_kernel({pillar_config, tuning_config, voxel_common, "pp_voxelization_cells.cl"},
                                           "pp_voxelization_cells",
                                           "-DINPUT0_TYPE=float -DOUTPUT0_TYPE=int");

    const std::string postproc_common = "pp_postproc_common.cl";
    const std::vector<std::string> postproc_prefix = {"pp_decoder_config_generated.cl", tuning_config,
                                                      postproc_common};
    auto postproc = [&](const char* file) {
        std::vector<std::string> files = postproc_prefix;
        files.push_back(file);
        return files;
    };
    self.scores_kernel = self.build_kernel(postproc("pp_postproc_scores.cl"), "pp_postproc_scores",
                                           kFp32RoundingOption);
    self.candidates_kernel = self.build_kernel(postproc("pp_postproc_candidates.cl"), "pp_postproc_candidates",
                                               kFp32RoundingOption);
    self.nms_kernel = self.build_kernel(postproc("pp_postproc_nms.cl"), "pp_postproc_nms",
                                        kFp32RoundingOption);
    self.gather_kernel = self.build_kernel(postproc("pp_postproc_gather.cl"), "pp_postproc_gather",
                                           kFp32RoundingOption);
}

PointPillarsRuntime::~PointPillarsRuntime() = default;

void PointPillarsRuntime::forward_on_device(const float* points, int point_count) {
    Impl& self = *impl_;
    if (point_count < 0) throw std::invalid_argument("point_count must not be negative");
    if (point_count > PP_MAX_POINTS) point_count = static_cast<int>(PP_MAX_POINTS);

    // An empty cloud may pass a NULL pointer, which clEnqueueMemcpyINTEL rejects.
    const size_t point_bytes = static_cast<size_t>(point_count) * PP_NUM_POINT_FEATURES * sizeof(float);
    if (point_bytes > 0)
        check_status(self.enqueue_memcpy(self.queue.get(), CL_FALSE, self.point_buffer, points, point_bytes, 0,
                                         nullptr, nullptr),
                     "clEnqueueMemcpyINTEL points");
    check_status(self.enqueue_memcpy(self.queue.get(), CL_FALSE, self.point_count_buffer, &point_count,
                                     sizeof(point_count), 0, nullptr, nullptr),
                 "clEnqueueMemcpyINTEL count");

    self.launch_kernel(self.scatter_kernel, {self.point_buffer, self.point_count_buffer, self.metadata_buffer},
                       PP_SCATTER_WG, PP_SCATTER_WG);
    self.launch_kernel(self.finalize_kernel, {self.metadata_buffer, self.point_buffer, self.pillar_features},
                       PP_MAX_VOXELS, kDriverSelectedLocalSize);
    self.launch_kernel(self.cells_kernel, {self.metadata_buffer, self.cell_to_pillar}, PP_GRID_SIZE,
                       kDriverSelectedLocalSize);

    self.infer_request.infer();

    self.launch_kernel(self.scores_kernel, {self.classification_logits, self.selection_bitmap}, PP_ANCHOR_COUNT,
                       PP_SCORES_WG);
    self.launch_kernel(self.candidates_kernel,
                       {self.selection_bitmap, self.classification_logits, self.box_regression,
                        self.direction_logits, self.candidate_records},
                       PP_CAND_WG, PP_CAND_WG);
    self.launch_kernel(self.nms_kernel, {self.candidate_records, self.nms_mask},
                       PP_NMS_GROUPS * PP_NMS_TILE_WG, PP_NMS_TILE_WG);
    self.launch_kernel(self.gather_kernel, {self.candidate_records, self.nms_mask, self.detections},
                       PP_GATHER_WG, PP_GATHER_WG);
}

const float* PointPillarsRuntime::forward(const float* points, int point_count) {
    Impl& self = *impl_;
    forward_on_device(points, point_count);
    check_status(self.enqueue_memcpy(self.queue.get(), CL_TRUE, self.host_detections.data(), self.detections,
                                     self.host_detections.size() * sizeof(float), 0, nullptr, nullptr),
                 "clEnqueueMemcpyINTEL detections");
    return self.host_detections.data();
}

int PointPillarsRuntime::max_detections() { return static_cast<int>(PP_MAX_DETECTIONS); }
int PointPillarsRuntime::detection_stride() { return static_cast<int>(PP_DET_CHANNELS); }

}  // namespace PpExtension

// C entry points convert runtime errors into failure return values.
extern "C" {

void* pp_runtime_create(const char* model_xml, const char* kernel_dir) try {
    return new PpExtension::PointPillarsRuntime(model_xml, kernel_dir);
} catch (const std::exception& error) {
    std::fprintf(stderr, "pp_runtime_create: %s\n", error.what());
    return nullptr;
}

void pp_runtime_destroy(void* runtime) {
    delete static_cast<PpExtension::PointPillarsRuntime*>(runtime);
}

const float* pp_runtime_forward(void* runtime, const float* points, int point_count) try {
    return static_cast<PpExtension::PointPillarsRuntime*>(runtime)->forward(points, point_count);
} catch (const std::exception& error) {
    std::fprintf(stderr, "pp_runtime_forward: %s\n", error.what());
    return nullptr;
}

int pp_runtime_max_detections() { return PpExtension::PointPillarsRuntime::max_detections(); }
int pp_runtime_detection_stride() { return PpExtension::PointPillarsRuntime::detection_stride(); }

}  // extern "C"
