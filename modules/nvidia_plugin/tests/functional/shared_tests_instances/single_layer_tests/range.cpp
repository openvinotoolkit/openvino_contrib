// Copyright (C) 2021-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <cuda_runtime.h>
#include <gmock/gmock.h>
#include <gtest/gtest.h>

#include <cuda_config.hpp>
#include <cuda_graph_context.hpp>
#include <cuda_op_buffers_extractor.hpp>
#include <cuda_operation_registry.hpp>
#include <cuda_simple_execution_delegator.hpp>
#include <kernels/details/cuda_type_traits.hpp>
#include <openvino/op/constant.hpp>
#include <openvino/op/parameter.hpp>
#include <openvino/op/range.hpp>
#include <openvino/op/result.hpp>
#include <ops/parameter.hpp>

namespace {

using devptr_t = CUDA::DevicePointer<void*>;
using cdevptr_t = CUDA::DevicePointer<const void*>;
using dataType = float;
using Type_t = ov::element::Type_t;
using Type = ov::element::Type;

typedef std::tuple<dataType,  // start
                   dataType,  // stop
                   dataType,  // step
                   Type_t,    // start type
                   Type_t,    // step type
                   Type_t     // out type
                   >
    CudaRangeParams;

struct CudaRangeLayerTest : public testing::WithParamInterface<CudaRangeParams>, virtual public ::testing::Test {
    using TensorID = ov::nvidia_gpu::TensorID;
    CudaRangeParams param = GetParam();
    dataType start = std::get<0>(param);
    dataType stop = std::get<1>(param);
    dataType step = std::get<2>(param);
    Type_t start_type = std::get<3>(param);
    Type_t step_type = std::get<4>(param);
    Type_t out_type = std::get<5>(param);
    size_t outputSize = computeRangeOutputSize(start, stop, step, start_type, step_type, out_type);
    ov::nvidia_gpu::ThreadContext threadContext{{}};
    CUDA::Allocation startParamAlloc = threadContext.stream().malloc(Type(start_type).size());
    CUDA::Allocation stopParamAlloc = threadContext.stream().malloc(sizeof(dataType));
    CUDA::Allocation stepParamAlloc = threadContext.stream().malloc(Type(step_type).size());
    CUDA::Allocation outAlloc = threadContext.stream().malloc(outputSize * Type(out_type).size());
    std::vector<cdevptr_t> inputs = {startParamAlloc, stopParamAlloc, stepParamAlloc};
    std::vector<devptr_t> outputs{outAlloc};
    InferenceEngine::BlobMap empty;
    ov::nvidia_gpu::OperationBase::Ptr operation = createOperation();

    static std::string getTestCaseName(testing::TestParamInfo<CudaRangeParams> obj) {
        dataType start, stop, step;
        Type_t start_type, step_type, out_type;
        std::tie(start, stop, step, start_type, step_type, out_type) = obj.param;

        std::ostringstream result;
        const char separator = '_';
        result << "Start=" << start << separator;
        result << "Stop=" << stop << separator;
        result << "Step=" << step << separator;
        result << "StartType=" << Type(start_type).get_type_name() << separator;
        result << "StepType=" << Type(step_type).get_type_name() << separator;
        result << "OutType=" << Type(out_type).get_type_name();
        return result.str();
    }

    static std::tuple<double /*start*/, double /*stop*/, double /*step*/> truncateInput(
        double start, double stop, double step, Type_t start_type, Type_t step_type, Type_t out_type) {
        // all inputs must be casted to output_type before
        // the rounding for casting values are done towards zero
        bool is_output_integral_number = Type(out_type).is_integral_number();
        if (is_output_integral_number && Type(start_type).is_real()) {
            start = std::trunc(start);
        }
        if (is_output_integral_number && Type(Type_t::f32).is_real()) {
            stop = std::trunc(stop);
        }
        if (is_output_integral_number && Type(step_type).is_real()) {
            step = std::trunc(step);
        }
        return {start, stop, step};
    }

    static size_t computeRangeOutputSize(
        double start, double stop, double step, Type_t start_type, Type_t step_type, Type_t out_type) {
        std::tie(start, stop, step) = truncateInput(start, stop, step, start_type, step_type, out_type);

        // the number of elements is: max(ceil((stop − start) / step), 0)
        double span;
        if ((step > 0 && start >= stop) || (step < 0 && start <= stop)) {
            span = 0;
        } else {
            span = stop - start;
        }
        size_t outputSize = ceil(fabs(span) / fabs(step));
        assert(outputSize > 0);
        return outputSize;
    }

    std::vector<dataType> getRefOutput() const {
        std::vector<dataType> ref;
        dataType current = start;
        for (int c = 0; c < outputSize; ++c) {
            ref.push_back(current);
            current += step;
        }
        return ref;
    }

    template <ov::nvidia_gpu::kernel::Type_t T>
    static void upload(const CUDA::Stream& stream, CUDA::Allocation& dst, const dataType* src, size_t size) {
        using TOutput = ov::nvidia_gpu::kernel::cuda_type_traits_t<T>;
        std::vector<TOutput> data;
        data.reserve(size);
        for (size_t c = 0; c < size; ++c) {
            data.push_back(static_cast<TOutput>(src[c]));
        }
        stream.upload(dst, &data[0], size * sizeof(TOutput));
    }

    static void upload(
        const CUDA::Stream& stream, CUDA::Allocation& dst, const dataType* src, Type_t type, size_t size) {
        using Type_k = ov::nvidia_gpu::kernel::Type_t;
        switch (type) {
#if defined __CUDACC__
#ifdef CUDA_HAS_BF16_TYPE
            case Type_t::bf16:
                return upload<Type_k::bf16>(stream, dst, src, size);
#endif
            case Type_t::f16:
                return upload<Type_k::f16>(stream, dst, src, size);
#endif
            case Type_t::f32:
                return upload<Type_k::f32>(stream, dst, src, size);
            case Type_t::f64:
                return upload<Type_k::f64>(stream, dst, src, size);
            case Type_t::i4:
                return upload<Type_k::i4>(stream, dst, src, size);
            case Type_t::i8:
                return upload<Type_k::i8>(stream, dst, src, size);
            case Type_t::i16:
                return upload<Type_k::i16>(stream, dst, src, size);
            case Type_t::i32:
                return upload<Type_k::i32>(stream, dst, src, size);
            case Type_t::i64:
                return upload<Type_k::i64>(stream, dst, src, size);
            case Type_t::u8:
                return upload<Type_k::u8>(stream, dst, src, size);
            case Type_t::u16:
                return upload<Type_k::u16>(stream, dst, src, size);
            case Type_t::u32:
                return upload<Type_k::u32>(stream, dst, src, size);
            case Type_t::u64:
                return upload<Type_k::u64>(stream, dst, src, size);
            default:
                throw std::runtime_error("unsupported type");
        }
    }

    template <ov::nvidia_gpu::kernel::Type_t T>
    static void download(const CUDA::Stream& stream, dataType* dst, devptr_t src, size_t size) {
        using TOutput = ov::nvidia_gpu::kernel::cuda_type_traits_t<T>;
        std::vector<TOutput> data;
        data.resize(size);
        stream.download(&data[0], src, size * sizeof(TOutput));
        stream.synchronize();
        for (size_t c = 0; c < data.size(); ++c) {
            dst[c] = static_cast<dataType>(data[c]);
        }
    }

    static void download(const CUDA::Stream& stream, dataType* dst, devptr_t src, Type_t type, size_t size) {
        using Type_k = ov::nvidia_gpu::kernel::Type_t;
        switch (type) {
#if defined __CUDACC__
#ifdef CUDA_HAS_BF16_TYPE
            case Type_t::bf16:
                return download<Type_k::bf16>(stream, dst, src, size);
#endif
            case Type_t::f16:
                return download<Type_k::f16>(stream, dst, src, size);
#endif
            case Type_t::f32:
                return download<Type_k::f32>(stream, dst, src, size);
            case Type_t::f64:
                return download<Type_k::f64>(stream, dst, src, size);
            case Type_t::i4:
                return download<Type_k::i4>(stream, dst, src, size);
            case Type_t::i8:
                return download<Type_k::i8>(stream, dst, src, size);
            case Type_t::i16:
                return download<Type_k::i16>(stream, dst, src, size);
            case Type_t::i32:
                return download<Type_k::i32>(stream, dst, src, size);
            case Type_t::i64:
                return download<Type_k::i64>(stream, dst, src, size);
            case Type_t::u8:
                return download<Type_k::u8>(stream, dst, src, size);
            case Type_t::u16:
                return download<Type_k::u16>(stream, dst, src, size);
            case Type_t::u32:
                return download<Type_k::u32>(stream, dst, src, size);
            case Type_t::u64:
                return download<Type_k::u64>(stream, dst, src, size);
            default:
                throw std::runtime_error("unsupported type");
        }
    }

protected:
    ov::nvidia_gpu::OperationBase::Ptr createOperation() {
        using namespace ngraph;
        CUDA::Device device{};
        const bool optimizeOption = false;
        NodeVector params;
        params.push_back(std::make_shared<ov::op::v0::Constant>(ov::element::Type(start_type), ov::Shape(), start));
        params.push_back(std::make_shared<ov::op::v0::Constant>(ov::element::f32, ov::Shape(), stop));
        params.push_back(std::make_shared<ov::op::v0::Constant>(ov::element::Type(step_type), ov::Shape(), step));
        params[0]->set_friendly_name("start");
        params[1]->set_friendly_name("stop");
        params[2]->set_friendly_name("step");
        auto node = std::make_shared<ov::op::v4::Range>(params[0], params[1], params[2], Type(out_type));
        auto& registry = ov::nvidia_gpu::OperationRegistry::getInstance();
        assert(registry.hasOperation(node));
        auto op = registry.createOperation(ov::nvidia_gpu::CreationContext{device, optimizeOption},
                                           node,
                                           std::vector<TensorID>{TensorID{0}, TensorID{1}, TensorID{2}},
                                           std::vector<TensorID>{TensorID{0u}});
        return op;
    }
};

#if defined(__GNUC__)
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wsuggest-override"
#endif
MATCHER_P(FloatNearPointwise, tol, "Out of range") {
    return (std::get<0>(arg) > std::get<1>(arg) - tol && std::get<0>(arg) < std::get<1>(arg) + tol);
}
#if defined(__GNUC__)
#pragma GCC diagnostic pop
#endif

TEST_P(CudaRangeLayerTest, CompareWithRefs) {
    ASSERT_TRUE(outputSize > 0);
    ov::nvidia_gpu::CancellationToken token{};
    ov::nvidia_gpu::SimpleExecutionDelegator simpleExecutionDelegator{};
    std::vector<std::shared_ptr<ov::Tensor>> emptyTensor;
    std::map<std::string, std::size_t> emptyMapping;
    ov::nvidia_gpu::CudaGraphContext cudaGraphContext;
    ov::nvidia_gpu::InferenceRequestContext context{emptyTensor,
                                                    emptyMapping,
                                                    emptyTensor,
                                                    emptyMapping,
                                                    threadContext,
                                                    token,
                                                    simpleExecutionDelegator,
                                                    cudaGraphContext};
    auto& stream = context.getThreadContext().stream();
    CudaRangeLayerTest::upload(stream, startParamAlloc, &start, start_type, 1);
    CudaRangeLayerTest::upload(stream, stopParamAlloc, &stop, Type_t::f32, 1);
    CudaRangeLayerTest::upload(stream, stepParamAlloc, &step, step_type, 1);
    std::vector<dataType> out(outputSize, -1);
    CudaRangeLayerTest::upload(stream, outAlloc, out.data(), out_type, outputSize);
    ASSERT_TRUE(operation);
    operation->Execute(context, inputs, outputs, {});
    CudaRangeLayerTest::download(stream, out.data(), outputs[0], out_type, outputSize);
    std::tie(start, stop, step) = truncateInput(start, stop, step, start_type, step_type, out_type);
    auto ref = getRefOutput();
    EXPECT_THAT(out, ::testing::Pointwise(FloatNearPointwise(1e-5), ref));
}

const std::vector<dataType> start = {1.0f, 1.2f};
const std::vector<dataType> stop = {5.0f, 5.2f};
const std::vector<dataType> step = {1.0f, 0.1f};
const std::vector<Type_t> types = {
#if defined __CUDACC__
#ifdef CUDA_HAS_BF16_TYPE
    Type_t::bf16,
#endif
    Type_t::f16,
#endif
    Type_t::f32,
    Type_t::f64,
    Type_t::i8,
    Type_t::i16,
    Type_t::i32,
    Type_t::i64,
    Type_t::u8,
    Type_t::u16,
    Type_t::u32,
    Type_t::u64,
};

const std::vector<Type_t> int_types = {
    Type_t::i8,
    Type_t::i16,
    Type_t::i32,
    Type_t::i64,
    Type_t::u8,
    Type_t::u16,
    Type_t::u32,
    Type_t::u64,
};

const std::vector<Type_t> float_types = {
#if defined __CUDACC__
#ifdef CUDA_HAS_BF16_TYPE
    Type_t::bf16,
#endif
    Type_t::f16,
#endif
    Type_t::f32,
    Type_t::f64,
};

INSTANTIATE_TEST_CASE_P(smoke_Basic,
                        CudaRangeLayerTest,
                        ::testing::Combine(::testing::ValuesIn(start),
                                           ::testing::ValuesIn(stop),
                                           ::testing::ValuesIn(step),
                                           ::testing::Values(Type_t::f32),
                                           ::testing::Values(Type_t::f32),
                                           ::testing::Values(Type_t::f32)),
                        CudaRangeLayerTest::getTestCaseName);

INSTANTIATE_TEST_CASE_P(smoke_float_input,
                        CudaRangeLayerTest,
                        ::testing::Combine(::testing::Values(1.2f),
                                           ::testing::Values(5.6f),
                                           ::testing::Values(1.1f),
                                           ::testing::ValuesIn(float_types),
                                           ::testing::ValuesIn(float_types),
                                           ::testing::ValuesIn(types)),
                        CudaRangeLayerTest::getTestCaseName);

INSTANTIATE_TEST_CASE_P(smoke_int_input,
                        CudaRangeLayerTest,
                        ::testing::Combine(::testing::Values(1),
                                           ::testing::Values(5),
                                           ::testing::Values(1),
                                           ::testing::ValuesIn(int_types),
                                           ::testing::ValuesIn(int_types),
                                           ::testing::ValuesIn(types)),
                        CudaRangeLayerTest::getTestCaseName);
}  // namespace
