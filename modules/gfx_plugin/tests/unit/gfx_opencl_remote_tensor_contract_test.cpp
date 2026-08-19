// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <gtest/gtest.h>

#include "backends/opencl/plugin/remote_tensor.hpp"
#include "backends/opencl/runtime/opencl_api.hpp"
#include "openvino/core/except.hpp"
#include "openvino/gfx_plugin/properties.hpp"

namespace ov {
namespace gfx_plugin {
namespace {

TEST(GfxOpenClRemoteTensorContractTest,
     CreateOwnedTensorAllocatesOpenClBufferWithManifestProperties) {
    auto runtime = OpenClRuntimeContext::instance();

    auto created = create_opencl_remote_tensor(
        ov::element::f32, ov::Shape{2, 3}, {}, runtime, 2u * 3u * sizeof(float));

    EXPECT_TRUE(created.tensor.buf.valid());
    EXPECT_EQ(created.tensor.buf.backend, GpuBackend::OpenCL);
    EXPECT_TRUE(created.tensor.buf.owned);
    EXPECT_FALSE(created.tensor.buf.external);
    EXPECT_GE(created.tensor.buf.size, 2u * 3u * sizeof(float));
    EXPECT_EQ(created.properties.at(kGfxBufferProperty).as<void*>(),
              created.tensor.buf.buffer);
    EXPECT_EQ(created.properties.at(kGfxMemoryProperty).as<void*>(),
              created.tensor.buf.buffer);
    EXPECT_EQ(created.properties.at(kGfxBufferBytesProperty).as<size_t>(),
              created.tensor.buf.size);
    EXPECT_TRUE(created.release_fn);

    created.release_fn(created.tensor);
    EXPECT_FALSE(created.tensor.buf.valid());
}

TEST(GfxOpenClRemoteTensorContractTest,
     WrapsExternalClMemWithoutTakingOwnershipAndValidatesSize) {
    auto runtime = OpenClRuntimeContext::instance();
    cl_int status = CL_SUCCESS;
    cl_mem external = runtime->api().fn().clCreateBuffer(
        runtime->context(), CL_MEM_READ_WRITE, 64u, nullptr, &status);
    opencl_check(status, "clCreateBuffer(test_external)");
    ASSERT_TRUE(external);

    ov::AnyMap params;
    params[kGfxBufferProperty] = reinterpret_cast<void*>(external);
    params[kGfxBufferBytesProperty] = static_cast<size_t>(64u);

    auto created = create_opencl_remote_tensor(
        ov::element::f32, ov::Shape{4}, params, runtime, 4u * sizeof(float));

    EXPECT_EQ(created.tensor.buf.buffer, reinterpret_cast<void*>(external));
    EXPECT_EQ(created.tensor.buf.size, 64u);
    EXPECT_TRUE(created.tensor.buf.from_handle);
    EXPECT_TRUE(created.tensor.buf.external);
    EXPECT_FALSE(created.tensor.buf.owned);
    EXPECT_TRUE(created.release_fn);

    created.release_fn(created.tensor);
    EXPECT_EQ(created.tensor.buf.buffer, reinterpret_cast<void*>(external));
    opencl_check(runtime->api().fn().clReleaseMemObject(external),
                 "clReleaseMemObject(test_external)");
}

TEST(GfxOpenClRemoteTensorContractTest,
     RejectsExternalClMemThatIsSmallerThanTensorContract) {
    auto runtime = OpenClRuntimeContext::instance();
    cl_int status = CL_SUCCESS;
    cl_mem external = runtime->api().fn().clCreateBuffer(
        runtime->context(), CL_MEM_READ_WRITE, 8u, nullptr, &status);
    opencl_check(status, "clCreateBuffer(test_too_small)");
    ASSERT_TRUE(external);

    ov::AnyMap params;
    params[kGfxMemoryProperty] = reinterpret_cast<void*>(external);
    params[kGfxBufferBytesProperty] = static_cast<size_t>(8u);

    EXPECT_THROW(create_opencl_remote_tensor(
                     ov::element::f32, ov::Shape{4}, params, runtime,
                     4u * sizeof(float)),
                 ov::Exception);

    opencl_check(runtime->api().fn().clReleaseMemObject(external),
                 "clReleaseMemObject(test_too_small)");
}

}  // namespace
}  // namespace gfx_plugin
}  // namespace ov
