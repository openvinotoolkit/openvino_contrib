// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//
#include "graph.cuh"

#include <gmock/gmock.h>
#include <gtest/gtest.h>

#include <array>
#include <cuda/event.hpp>
#include <cuda/graph.hpp>
#include <gsl/gsl_util>

#include "openvino/core/except.hpp"

using namespace testing;

namespace {

class CudaGraphCaptureCppWrappersTest : public Test {
protected:
    std::array<int, 8> a{0, 1, 2, 3, 4, 5, 6, 7};
    std::array<int, 8> b{8, 9, 10, 11, 12, 13, 14, 15};
    std::array<int, 8> result{};
    std::size_t buffer_size{sizeof(decltype(a)::value_type) * a.size()};
    CUDA::Stream stream{};
    CUDA::Allocation devA = stream.malloc(buffer_size);
    CUDA::Allocation devB = stream.malloc(buffer_size);
    CUDA::Allocation devResult = stream.malloc(buffer_size);
};

TEST_F(CudaGraphCaptureCppWrappersTest, Capture) {
    CUDA::GraphCapture capture{stream};
    {
        auto scope = capture.getScope();
        stream.upload(devA, a.data(), buffer_size);
        stream.upload(devB, b.data(), buffer_size);
        enqueueVecAdd(stream,
                      dim3{},
                      dim3{gsl::narrow<unsigned>(a.size())},
                      static_cast<int*>(devA.get()),
                      static_cast<int*>(devB.get()),
                      static_cast<int*>(devResult.get()),
                      a.size());
        stream.download(result.data(), devResult, buffer_size);
    }
    CUDA::GraphExec exec{capture.getGraph()};
    stream.synchronize();
    ASSERT_THAT(result, ElementsAre(0, 0, 0, 0, 0, 0, 0, 0));
    exec.launch(stream);
    stream.synchronize();
    ASSERT_THAT(result, ElementsAre(8, 10, 12, 14, 16, 18, 20, 22));
}

TEST_F(CudaGraphCaptureCppWrappersTest, UpdateGraph) {
    CUDA::GraphCapture capture{stream};
    CUDA::Allocation dummyAllocation = stream.malloc(buffer_size);
    {
        auto scope = capture.getScope();
        stream.upload(dummyAllocation, reinterpret_cast<void*>(0xffffl), buffer_size);
        stream.upload(dummyAllocation, reinterpret_cast<void*>(0xffffl), buffer_size);
        enqueueVecAdd(stream,
                      dim3{},
                      dim3{gsl::narrow<unsigned>(a.size())},
                      static_cast<int*>(dummyAllocation.get()),
                      static_cast<int*>(dummyAllocation.get()),
                      static_cast<int*>(dummyAllocation.get()),
                      0);
        stream.download(reinterpret_cast<void*>(0xffffl), dummyAllocation, buffer_size);
    }
    CUDA::GraphExec exec{capture.getGraph()};
    CUDA::GraphCapture captureUpdate{stream};
    {
        auto scope = captureUpdate.getScope();
        stream.upload(devA, a.data(), buffer_size);
        stream.upload(devB, b.data(), buffer_size);
        enqueueVecAdd(stream,
                      dim3{},
                      dim3{gsl::narrow<unsigned>(a.size())},
                      static_cast<int*>(devA.get()),
                      static_cast<int*>(devB.get()),
                      static_cast<int*>(devResult.get()),
                      a.size());
        stream.download(result.data(), devResult, buffer_size);
    }
    exec.update(captureUpdate.getGraph());
    exec.launch(stream);
    stream.synchronize();
    ASSERT_THAT(result, ElementsAre(8, 10, 12, 14, 16, 18, 20, 22));
}

TEST_F(CudaGraphCaptureCppWrappersTest, EventCapture) {
    CUDA::Event event0{};
    CUDA::Event event1{};
    CUDA::GraphCapture capture{stream};
    {
        auto scope = capture.getScope();
        event0.record(stream, CUDA::Event::RecordMode::External);
        stream.upload(devA, a.data(), buffer_size);
        stream.upload(devB, b.data(), buffer_size);
        enqueueVecAdd(stream,
                      dim3{},
                      dim3{gsl::narrow<unsigned>(a.size())},
                      static_cast<int*>(devA.get()),
                      static_cast<int*>(devB.get()),
                      static_cast<int*>(devResult.get()),
                      a.size());
        stream.download(result.data(), devResult, buffer_size);
        event1.record(stream, CUDA::Event::RecordMode::External);
    }
    CUDA::GraphExec exec{capture.getGraph()};
    exec.launch(stream);
    stream.synchronize();
    EXPECT_GT(event1.elapsedSince(event0), 0);
}

TEST_F(CudaGraphCaptureCppWrappersTest, EventUpdate) {
    CUDA::Event event0{};
    CUDA::Event event1{};
    CUDA::GraphCapture capture0{stream};
    {
        auto scope = capture0.getScope();
        event0.record(stream, CUDA::Event::RecordMode::External);
        stream.upload(devA, a.data(), buffer_size);
        stream.upload(devB, b.data(), buffer_size);
        enqueueVecAdd(stream,
                      dim3{},
                      dim3{gsl::narrow<unsigned>(a.size())},
                      static_cast<int*>(devA.get()),
                      static_cast<int*>(devB.get()),
                      static_cast<int*>(devResult.get()),
                      a.size());
        stream.download(result.data(), devResult, buffer_size);
        event1.record(stream, CUDA::Event::RecordMode::External);
    }
    CUDA::GraphExec exec{capture0.getGraph()};
    CUDA::Event event2{};
    CUDA::Event event3{};
    CUDA::GraphCapture capture1{stream};
    {
        auto scope = capture1.getScope();
        event2.record(stream, CUDA::Event::RecordMode::External);
        stream.upload(devA, a.data(), buffer_size);
        stream.upload(devB, b.data(), buffer_size);
        enqueueVecAdd(stream,
                      dim3{},
                      dim3{gsl::narrow<unsigned>(a.size())},
                      static_cast<int*>(devA.get()),
                      static_cast<int*>(devB.get()),
                      static_cast<int*>(devResult.get()),
                      a.size());
        stream.download(result.data(), devResult, buffer_size);
        event3.record(stream, CUDA::Event::RecordMode::External);
    }
    exec.update(capture1.getGraph());
    exec.launch(stream);
    stream.synchronize();
    EXPECT_GT(event3.elapsedSince(event2), 0);
    EXPECT_THROW(event1.elapsedSince(event0), ov::Exception);
}

}  // namespace
