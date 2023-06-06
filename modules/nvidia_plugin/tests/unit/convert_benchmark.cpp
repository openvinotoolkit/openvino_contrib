// Copyright (C) 2021-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <gtest/gtest.h>

#include <chrono>
#include <cuda_config.hpp>
#include <cuda_operation_registry.hpp>
#include <cuda_profiler.hpp>
#include <iomanip>
#include <ngraph/node.hpp>
#include <openvino/op/convert.hpp>
#include <openvino/op/parameter.hpp>
#include <ops/parameter.hpp>
#include <random>

namespace {

struct ConvertTest : testing::Test {
    ov::nvidia_gpu::ThreadContext threadContext{CUDA::Device{}};
    const ov::Shape inputTensorShape{1, 1, 3, 1024, 1024};
    std::vector<std::shared_ptr<ov::Tensor>> emptyTensor;
    std::map<std::string, std::size_t> emptyMapping;

    auto create_operation(ov::element::Type_t input, ov::element::Type_t output) {
        auto param =
            std::make_shared<ov::op::v0::Parameter>(ov::element::Type(input), ov::PartialShape(inputTensorShape));
        const auto node = std::make_shared<ov::op::v0::Convert>(param->output(0), ov::element::Type(output));

        static constexpr bool optimizeOption = false;
        auto& registry = ov::nvidia_gpu::OperationRegistry::getInstance();
        return registry.createOperation(ov::nvidia_gpu::CreationContext{threadContext.device(), optimizeOption},
                                        node,
                                        std::vector<ov::nvidia_gpu::TensorID>{ov::nvidia_gpu::TensorID{0u}},
                                        std::vector<ov::nvidia_gpu::TensorID>{ov::nvidia_gpu::TensorID{0u}});
    }
};

TEST_F(ConvertTest, DISABLED_benchmark) {
    using microseconds = std::chrono::duration<double, std::micro>;
    constexpr int kNumAttempts = 200;

    auto& stream = threadContext.stream();
    ov::nvidia_gpu::ExecGraph graph{ov::nvidia_gpu::CreationContext{CUDA::Device{}, false}, {}};
    ov::nvidia_gpu::CancellationToken token{};
    ov::nvidia_gpu::Profiler profiler{false, graph};
    ov::nvidia_gpu::InferenceRequestContext context{
        emptyTensor, emptyMapping, emptyTensor, emptyMapping, threadContext, token, profiler};

    using Type_t = ov::element::Type_t;
    constexpr Type_t supported_types[] = {Type_t::boolean,
                                          Type_t::bf16,
                                          Type_t::f16,
                                          Type_t::f32,
                                          Type_t::f64,
                                          Type_t::i8,
                                          Type_t::i16,
                                          Type_t::i32,
                                          Type_t::i64,
                                          /*Type_t::u1, convert doesn't support it*/
                                          Type_t::u8,
                                          Type_t::u16,
                                          Type_t::u32,
                                          Type_t::u64};
    for (auto inputIdx : supported_types) {
        for (auto outputIdx : supported_types) {
            const auto inputType = Type_t(static_cast<std::underlying_type<Type_t>::type>(inputIdx));
            const auto outputType = Type_t(static_cast<std::underlying_type<Type_t>::type>(outputIdx));
            auto op = create_operation(inputType, outputType);
            const auto input_type = ov::element::Type(inputType);
            const auto output_type = ov::element::Type(outputType);
            const auto inputBufferSize = ov::shape_size(inputTensorShape) * input_type.size();
            const auto ouputBufferSize = ov::shape_size(inputTensorShape) * output_type.size();
            const CUDA::Allocation inAlloc = stream.malloc(inputBufferSize);
            const CUDA::Allocation outAlloc = stream.malloc(ouputBufferSize);
            std::vector<CUDA::DevicePointer<const void*>> inputs{inAlloc};
            std::vector<CUDA::DevicePointer<void*>> outputs{outAlloc};
            std::vector<uint8_t> in(inputBufferSize);
            std::random_device r_device;
            std::mt19937 mersenne_engine{r_device()};
            std::uniform_int_distribution<> dist{std::numeric_limits<uint8_t>::min(),
                                                 std::numeric_limits<uint8_t>::max()};
            auto gen = [&dist, &mersenne_engine]() {
                return 10.f * dist(mersenne_engine) / std::numeric_limits<uint8_t>::max();
            };
            std::generate(in.begin(), in.end(), gen);
            stream.upload(inAlloc, in.data(), inputBufferSize);

            auto start = std::chrono::steady_clock::now();
            for (int i = 0; i < kNumAttempts; i++) {
                ov::nvidia_gpu::Workbuffers workbuffers{};
                op->Execute(context, inputs, outputs, workbuffers);
                stream.synchronize();
            }
            auto end = std::chrono::steady_clock::now();
            microseconds average_exec_time = (end - start) / kNumAttempts;
            if (inputType == outputType) std::cout << "    ";
            std::cout << std::fixed << std::setfill('0') << "Input type:" << input_type.get_type_name()
                      << " Output type:" << output_type.get_type_name()
                      << " Strided slice execution time: " << average_exec_time.count() << " microseconds\n";
        }
        std::cout << std::endl;
    }
}
}  // namespace
