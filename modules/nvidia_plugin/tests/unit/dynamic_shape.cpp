// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <gtest/gtest.h>

#include <algorithm>
#include <cmath>
#include <future>
#include <memory>
#include <random>
#include <vector>

#include "cuda_compiled_model.hpp"
#include "cuda_dynamic_operation.hpp"
#include "cuda_plugin.hpp"
#include "openvino/op/abs.hpp"
#include "openvino/op/add.hpp"
#include "openvino/op/constant.hpp"
#include "openvino/op/matmul.hpp"
#include "openvino/op/multiply.hpp"
#include "openvino/op/parameter.hpp"
#include "openvino/op/relu.hpp"
#include "openvino/op/result.hpp"
#include "openvino/op/sigmoid.hpp"
#include "openvino/op/sqrt.hpp"
#include "openvino/op/tanh.hpp"
#include "openvino/runtime/make_tensor.hpp"
#include "transformations/init_node_info.hpp"

using namespace ov::nvidia_gpu;

namespace {

// ---- Model factories -------------------------------------------------------

// Parameter(f32, {-1, N}) -> Relu -> Result
std::shared_ptr<ov::Model> createDynamicReluModel(int64_t staticDim = 4) {
    auto param = std::make_shared<ov::op::v0::Parameter>(
        ov::element::f32, ov::PartialShape{-1, staticDim});
    param->set_friendly_name("input");
    auto relu = std::make_shared<ov::op::v0::Relu>(param);
    relu->set_friendly_name("relu");
    auto result = std::make_shared<ov::op::v0::Result>(relu);
    auto model = std::make_shared<ov::Model>(ov::ResultVector{result},
                                              ov::ParameterVector{param}, "DynamicRelu");
    ov::pass::InitNodeInfo().run_on_model(model);
    return model;
}

// Parameter(f32, {-1, N}) + Constant(f32, {1, N}) -> Result
std::shared_ptr<ov::Model> createDynamicAddConstModel(int64_t staticDim = 4) {
    auto param = std::make_shared<ov::op::v0::Parameter>(
        ov::element::f32, ov::PartialShape{-1, staticDim});
    param->set_friendly_name("input");
    std::vector<float> constData(staticDim);
    for (int64_t i = 0; i < staticDim; ++i) constData[i] = static_cast<float>(i + 1);
    auto constant = std::make_shared<ov::op::v0::Constant>(
        ov::element::f32, ov::Shape{1, static_cast<size_t>(staticDim)}, constData);
    auto add = std::make_shared<ov::op::v1::Add>(param, constant);
    add->set_friendly_name("add");
    auto result = std::make_shared<ov::op::v0::Result>(add);
    auto model = std::make_shared<ov::Model>(ov::ResultVector{result},
                                              ov::ParameterVector{param}, "DynamicAddConst");
    ov::pass::InitNodeInfo().run_on_model(model);
    return model;
}

// Parameter(f32, {-1, N}) -> Relu -> Sigmoid -> Tanh -> Result
std::shared_ptr<ov::Model> createDynamicActivationChainModel(int64_t staticDim = 4) {
    auto param = std::make_shared<ov::op::v0::Parameter>(
        ov::element::f32, ov::PartialShape{-1, staticDim});
    param->set_friendly_name("input");
    auto relu = std::make_shared<ov::op::v0::Relu>(param);
    relu->set_friendly_name("relu");
    auto sigmoid = std::make_shared<ov::op::v0::Sigmoid>(relu);
    sigmoid->set_friendly_name("sigmoid");
    auto tanh_op = std::make_shared<ov::op::v0::Tanh>(sigmoid);
    tanh_op->set_friendly_name("tanh");
    auto result = std::make_shared<ov::op::v0::Result>(tanh_op);
    auto model = std::make_shared<ov::Model>(ov::ResultVector{result},
                                              ov::ParameterVector{param}, "DynamicActivationChain");
    ov::pass::InitNodeInfo().run_on_model(model);
    return model;
}

// Parameter(f32, {-1, N}) x Constant(f32, {N, M}) + Constant(f32, {1, M}) -> Relu -> Result
std::shared_ptr<ov::Model> createDynamicDenseReluModel(int64_t inputDim = 4, int64_t outputDim = 8) {
    auto param = std::make_shared<ov::op::v0::Parameter>(
        ov::element::f32, ov::PartialShape{-1, inputDim});
    param->set_friendly_name("input");

    std::vector<float> weightData(inputDim * outputDim, 0.1f);
    auto weights = std::make_shared<ov::op::v0::Constant>(
        ov::element::f32, ov::Shape{static_cast<size_t>(inputDim), static_cast<size_t>(outputDim)}, weightData);

    auto matmul = std::make_shared<ov::op::v0::MatMul>(param, weights, false, false);
    matmul->set_friendly_name("matmul");

    std::vector<float> biasData(outputDim, 0.5f);
    auto bias = std::make_shared<ov::op::v0::Constant>(
        ov::element::f32, ov::Shape{1, static_cast<size_t>(outputDim)}, biasData);

    auto add = std::make_shared<ov::op::v1::Add>(matmul, bias);
    add->set_friendly_name("add_bias");

    auto relu = std::make_shared<ov::op::v0::Relu>(add);
    relu->set_friendly_name("relu");

    auto result = std::make_shared<ov::op::v0::Result>(relu);
    auto model = std::make_shared<ov::Model>(ov::ResultVector{result},
                                              ov::ParameterVector{param}, "DynamicDenseRelu");
    ov::pass::InitNodeInfo().run_on_model(model);
    return model;
}

// Parameter(f32, {-1, N}) * Constant(2.0) -> Abs -> Sqrt -> Result
std::shared_ptr<ov::Model> createDynamicMulAbsSqrtModel(int64_t staticDim = 4) {
    auto param = std::make_shared<ov::op::v0::Parameter>(
        ov::element::f32, ov::PartialShape{-1, staticDim});
    param->set_friendly_name("input");

    std::vector<float> mulData(staticDim, 2.0f);
    auto mulConst = std::make_shared<ov::op::v0::Constant>(
        ov::element::f32, ov::Shape{1, static_cast<size_t>(staticDim)}, mulData);

    auto mul = std::make_shared<ov::op::v1::Multiply>(param, mulConst);
    mul->set_friendly_name("mul");
    auto abs_op = std::make_shared<ov::op::v0::Abs>(mul);
    abs_op->set_friendly_name("abs");
    auto sqrt_op = std::make_shared<ov::op::v0::Sqrt>(abs_op);
    sqrt_op->set_friendly_name("sqrt");

    auto result = std::make_shared<ov::op::v0::Result>(sqrt_op);
    auto model = std::make_shared<ov::Model>(ov::ResultVector{result},
                                              ov::ParameterVector{param}, "DynamicMulAbsSqrt");
    ov::pass::InitNodeInfo().run_on_model(model);
    return model;
}

// Parameter_A(f32, {-1, N}) + Parameter_B(f32, {-1, N}) -> Sigmoid -> Result
std::shared_ptr<ov::Model> createDynamicTwoInputAddModel(int64_t staticDim = 4) {
    auto paramA = std::make_shared<ov::op::v0::Parameter>(
        ov::element::f32, ov::PartialShape{-1, staticDim});
    paramA->set_friendly_name("input_a");
    auto paramB = std::make_shared<ov::op::v0::Parameter>(
        ov::element::f32, ov::PartialShape{-1, staticDim});
    paramB->set_friendly_name("input_b");

    auto add = std::make_shared<ov::op::v1::Add>(paramA, paramB);
    add->set_friendly_name("add");
    auto sigmoid = std::make_shared<ov::op::v0::Sigmoid>(add);
    sigmoid->set_friendly_name("sigmoid");

    auto result = std::make_shared<ov::op::v0::Result>(sigmoid);
    auto model = std::make_shared<ov::Model>(ov::ResultVector{result},
                                              ov::ParameterVector{paramA, paramB},
                                              "DynamicTwoInputAdd");
    ov::pass::InitNodeInfo().run_on_model(model);
    return model;
}

// ---- Helpers ----------------------------------------------------------------

std::shared_ptr<ov::ICompiledModel> compileModel(const std::shared_ptr<ov::Model>& model) {
    auto plugin = std::make_shared<Plugin>();
    ov::AnyMap props = {
        {ov::device::id.name(), "0"},
        {ov::hint::inference_precision.name(), "f32"},
    };
    return plugin->compile_model(model, props);
}

void fillRandom(const ov::SoPtr<ov::ITensor>& tensor, float lo = -5.0f, float hi = 5.0f, int seed = 42) {
    std::mt19937 rng(seed);
    std::uniform_real_distribution<float> dist(lo, hi);
    auto* p = static_cast<float*>(tensor->data());
    for (size_t i = 0; i < tensor->get_size(); ++i) p[i] = dist(rng);
}

}  // namespace

// =============================================================================
// Compilation tests: verify exec sequence has DynamicOperation
// =============================================================================

class DynamicModelCompileTest : public testing::Test {};

TEST_F(DynamicModelCompileTest, ReluModel_HasDynamicOperationInExecSequence) {
    auto model = createDynamicReluModel();
    ASSERT_TRUE(model->is_dynamic());

    auto compiled = std::dynamic_pointer_cast<CompiledModel>(compileModel(model));
    ASSERT_TRUE(compiled);

    const auto& execSeq = compiled->get_topology_runner().GetSubGraph().getExecSequence();
    bool hasDynOp = false;
    for (const auto& op : execSeq) {
        if (dynamic_cast<const DynamicOperation*>(op.get())) {
            hasDynOp = true;
            break;
        }
    }
    EXPECT_TRUE(hasDynOp) << "Exec sequence should contain at least one DynamicOperation";
}

TEST_F(DynamicModelCompileTest, AddConstModel_HasDynamicOperationInExecSequence) {
    auto model = createDynamicAddConstModel();
    ASSERT_TRUE(model->is_dynamic());

    auto compiled = std::dynamic_pointer_cast<CompiledModel>(compileModel(model));
    ASSERT_TRUE(compiled);

    const auto& execSeq = compiled->get_topology_runner().GetSubGraph().getExecSequence();
    bool hasDynOp = false;
    for (const auto& op : execSeq) {
        if (dynamic_cast<const DynamicOperation*>(op.get())) {
            hasDynOp = true;
            break;
        }
    }
    EXPECT_TRUE(hasDynOp) << "Exec sequence should contain at least one DynamicOperation";
}

TEST_F(DynamicModelCompileTest, DynamicOperation_CudaGraphCompatibility_IsNone) {
    auto model = createDynamicReluModel();
    auto compiled = std::dynamic_pointer_cast<CompiledModel>(compileModel(model));
    ASSERT_TRUE(compiled);

    const auto& execSeq = compiled->get_topology_runner().GetSubGraph().getExecSequence();
    for (const auto& op : execSeq) {
        if (dynamic_cast<const DynamicOperation*>(op.get())) {
            EXPECT_EQ(op->GetCudaGraphCompatibility(), CudaGraphCompatibility::NONE);
        }
    }
}

// =============================================================================
// ReLU inference: parameterized by staticDim
// =============================================================================

class DynamicReluTest : public testing::TestWithParam<int64_t> {
protected:
    void runRelu(ov::Shape inputShape) {
        auto model = createDynamicReluModel(static_cast<int64_t>(inputShape.back()));
        auto compiled = compileModel(model);
        auto req = compiled->create_infer_request();

        auto input = ov::make_tensor(ov::element::f32, inputShape);
        fillRandom(input);
        req->set_tensor(model->get_parameters()[0]->output(0), input);

        req->infer();

        auto output = req->get_tensor(model->get_results()[0]->output(0));
        ASSERT_EQ(output->get_shape(), inputShape);

        const auto* inData = static_cast<const float*>(input->data());
        const auto* outData = static_cast<const float*>(output->data());
        for (size_t i = 0; i < input->get_size(); ++i) {
            ASSERT_NEAR(outData[i], std::max(0.0f, inData[i]), 1e-5f)
                << "Mismatch at index " << i;
        }
    }
};

TEST_P(DynamicReluTest, Batch1) { runRelu({1, static_cast<size_t>(GetParam())}); }
TEST_P(DynamicReluTest, Batch8) { runRelu({8, static_cast<size_t>(GetParam())}); }
TEST_P(DynamicReluTest, DifferentBatches) {
    auto dim = static_cast<size_t>(GetParam());
    runRelu({2, dim});
    runRelu({5, dim});
    runRelu({2, dim});  // cache hit
}
TEST_P(DynamicReluTest, ManyBatches) {
    auto dim = static_cast<size_t>(GetParam());
    for (size_t batch : {1, 2, 4, 8, 16, 32, 7, 13, 1, 4, 32}) {
        runRelu({batch, dim});
    }
}

INSTANTIATE_TEST_SUITE_P(StaticDimVariations, DynamicReluTest,
                         testing::Values(1, 4, 7, 16, 64, 128));

// =============================================================================
// AddConst inference: parameterized by staticDim
// =============================================================================

class DynamicAddConstTest : public testing::TestWithParam<int64_t> {
protected:
    void runAddConst(ov::Shape inputShape) {
        const int64_t staticDim = static_cast<int64_t>(inputShape.back());
        auto model = createDynamicAddConstModel(staticDim);
        auto compiled = compileModel(model);
        auto req = compiled->create_infer_request();

        auto input = ov::make_tensor(ov::element::f32, inputShape);
        fillRandom(input);
        req->set_tensor(model->get_parameters()[0]->output(0), input);

        req->infer();

        auto output = req->get_tensor(model->get_results()[0]->output(0));
        ASSERT_EQ(output->get_shape(), inputShape);

        const auto* inData = static_cast<const float*>(input->data());
        const auto* outData = static_cast<const float*>(output->data());
        for (size_t i = 0; i < input->get_size(); ++i) {
            float constVal = static_cast<float>((i % staticDim) + 1);
            ASSERT_NEAR(outData[i], inData[i] + constVal, 1e-5f)
                << "Mismatch at index " << i;
        }
    }
};

TEST_P(DynamicAddConstTest, Batch1) { runAddConst({1, static_cast<size_t>(GetParam())}); }
TEST_P(DynamicAddConstTest, Batch8) { runAddConst({8, static_cast<size_t>(GetParam())}); }
TEST_P(DynamicAddConstTest, DifferentBatches) {
    auto dim = static_cast<size_t>(GetParam());
    runAddConst({3, dim});
    runAddConst({7, dim});
    runAddConst({3, dim});  // cache hit
}

INSTANTIATE_TEST_SUITE_P(StaticDimVariations, DynamicAddConstTest,
                         testing::Values(1, 4, 7, 16, 64, 128));

// =============================================================================
// Activation Chain: ReLU -> Sigmoid -> Tanh
// =============================================================================

class DynamicActivationChainTest : public testing::TestWithParam<int64_t> {
protected:
    void runActivationChain(ov::Shape inputShape) {
        auto model = createDynamicActivationChainModel(static_cast<int64_t>(inputShape.back()));
        auto compiled = compileModel(model);
        auto req = compiled->create_infer_request();

        auto input = ov::make_tensor(ov::element::f32, inputShape);
        fillRandom(input);
        req->set_tensor(model->get_parameters()[0]->output(0), input);

        req->infer();

        auto output = req->get_tensor(model->get_results()[0]->output(0));
        ASSERT_EQ(output->get_shape(), inputShape);

        const auto* inData = static_cast<const float*>(input->data());
        const auto* outData = static_cast<const float*>(output->data());
        for (size_t i = 0; i < input->get_size(); ++i) {
            float relu_out = std::max(0.0f, inData[i]);
            float sigmoid_out = 1.0f / (1.0f + std::exp(-relu_out));
            float expected = std::tanh(sigmoid_out);
            ASSERT_NEAR(outData[i], expected, 1e-4f)
                << "Mismatch at index " << i;
        }
    }
};

TEST_P(DynamicActivationChainTest, Batch1) { runActivationChain({1, static_cast<size_t>(GetParam())}); }
TEST_P(DynamicActivationChainTest, Batch4) { runActivationChain({4, static_cast<size_t>(GetParam())}); }
TEST_P(DynamicActivationChainTest, MultipleBatches) {
    auto dim = static_cast<size_t>(GetParam());
    runActivationChain({2, dim});
    runActivationChain({8, dim});
    runActivationChain({2, dim});  // cache hit
}

INSTANTIATE_TEST_SUITE_P(Variations, DynamicActivationChainTest,
                         testing::Values(1, 4, 16, 64));

// =============================================================================
// Dense Layer: MatMul + Add(bias) + ReLU
// =============================================================================

class DynamicDenseReluTest : public testing::TestWithParam<std::tuple<int64_t, int64_t>> {
protected:
    void runDenseRelu(size_t batch) {
        auto [inputDim, outputDim] = GetParam();
        auto model = createDynamicDenseReluModel(inputDim, outputDim);
        auto compiled = compileModel(model);
        auto req = compiled->create_infer_request();

        ov::Shape inputShape{batch, static_cast<size_t>(inputDim)};
        ov::Shape outputShape{batch, static_cast<size_t>(outputDim)};

        auto input = ov::make_tensor(ov::element::f32, inputShape);
        fillRandom(input);
        req->set_tensor(model->get_parameters()[0]->output(0), input);

        req->infer();

        auto output = req->get_tensor(model->get_results()[0]->output(0));
        ASSERT_EQ(output->get_shape(), outputShape);

        const auto* inData = static_cast<const float*>(input->data());
        const auto* outData = static_cast<const float*>(output->data());
        for (size_t r = 0; r < batch; ++r) {
            for (int64_t j = 0; j < outputDim; ++j) {
                float matmul_val = 0.0f;
                for (int64_t k = 0; k < inputDim; ++k) {
                    matmul_val += inData[r * inputDim + k] * 0.1f;
                }
                float expected = std::max(0.0f, matmul_val + 0.5f);
                ASSERT_NEAR(outData[r * outputDim + j], expected, 1e-3f)
                    << "Mismatch at [" << r << "][" << j << "]";
            }
        }
    }
};

TEST_P(DynamicDenseReluTest, Batch1) { runDenseRelu(1); }
TEST_P(DynamicDenseReluTest, Batch4) { runDenseRelu(4); }
TEST_P(DynamicDenseReluTest, DifferentBatches) {
    runDenseRelu(2);
    runDenseRelu(8);
    runDenseRelu(2);  // cache hit
}

INSTANTIATE_TEST_SUITE_P(Variations, DynamicDenseReluTest,
                         testing::Values(
                             std::make_tuple(4, 8),
                             std::make_tuple(16, 32),
                             std::make_tuple(7, 16),
                             std::make_tuple(64, 128)));

// =============================================================================
// Elementwise Chain: Multiply(const) -> Abs -> Sqrt
// =============================================================================

class DynamicMulAbsSqrtTest : public testing::TestWithParam<int64_t> {
protected:
    void runMulAbsSqrt(ov::Shape inputShape) {
        auto model = createDynamicMulAbsSqrtModel(static_cast<int64_t>(inputShape.back()));
        auto compiled = compileModel(model);
        auto req = compiled->create_infer_request();

        auto input = ov::make_tensor(ov::element::f32, inputShape);
        fillRandom(input, 0.5f, 5.0f);  // positive range for stable sqrt
        req->set_tensor(model->get_parameters()[0]->output(0), input);

        req->infer();

        auto output = req->get_tensor(model->get_results()[0]->output(0));
        ASSERT_EQ(output->get_shape(), inputShape);

        const auto* inData = static_cast<const float*>(input->data());
        const auto* outData = static_cast<const float*>(output->data());
        for (size_t i = 0; i < input->get_size(); ++i) {
            float expected = std::sqrt(std::abs(inData[i] * 2.0f));
            ASSERT_NEAR(outData[i], expected, 1e-5f)
                << "Mismatch at index " << i;
        }
    }
};

TEST_P(DynamicMulAbsSqrtTest, Batch1) { runMulAbsSqrt({1, static_cast<size_t>(GetParam())}); }
TEST_P(DynamicMulAbsSqrtTest, Batch8) { runMulAbsSqrt({8, static_cast<size_t>(GetParam())}); }

INSTANTIATE_TEST_SUITE_P(Variations, DynamicMulAbsSqrtTest,
                         testing::Values(1, 4, 16, 64));

// =============================================================================
// Two-Input Model: Parameter_A + Parameter_B -> Sigmoid
// =============================================================================

class DynamicTwoInputTest : public testing::TestWithParam<int64_t> {
protected:
    void runTwoInput(ov::Shape inputShape) {
        auto model = createDynamicTwoInputAddModel(static_cast<int64_t>(inputShape.back()));
        auto compiled = compileModel(model);
        auto req = compiled->create_infer_request();

        auto inputA = ov::make_tensor(ov::element::f32, inputShape);
        fillRandom(inputA, -3.0f, 3.0f, 42);
        auto inputB = ov::make_tensor(ov::element::f32, inputShape);
        fillRandom(inputB, -3.0f, 3.0f, 123);
        req->set_tensor(model->get_parameters()[0]->output(0), inputA);
        req->set_tensor(model->get_parameters()[1]->output(0), inputB);

        req->infer();

        auto output = req->get_tensor(model->get_results()[0]->output(0));
        ASSERT_EQ(output->get_shape(), inputShape);

        const auto* dataA = static_cast<const float*>(inputA->data());
        const auto* dataB = static_cast<const float*>(inputB->data());
        const auto* outData = static_cast<const float*>(output->data());
        for (size_t i = 0; i < output->get_size(); ++i) {
            float expected = 1.0f / (1.0f + std::exp(-(dataA[i] + dataB[i])));
            ASSERT_NEAR(outData[i], expected, 1e-5f)
                << "Mismatch at index " << i;
        }
    }
};

TEST_P(DynamicTwoInputTest, Batch1) { runTwoInput({1, static_cast<size_t>(GetParam())}); }
TEST_P(DynamicTwoInputTest, Batch8) { runTwoInput({8, static_cast<size_t>(GetParam())}); }
TEST_P(DynamicTwoInputTest, DifferentBatches) {
    auto dim = static_cast<size_t>(GetParam());
    runTwoInput({3, dim});
    runTwoInput({7, dim});
    runTwoInput({3, dim});  // cache hit
}

INSTANTIATE_TEST_SUITE_P(Variations, DynamicTwoInputTest,
                         testing::Values(4, 16, 64));

// =============================================================================
// Concurrent async inference: multiple requests on the same compiled model
// =============================================================================

class DynamicConcurrentAsyncTest : public testing::Test {
protected:
    struct InferResult {
        bool success = false;
        std::string error;
    };

    InferResult runAsyncRelu(const std::shared_ptr<ov::IAsyncInferRequest>& req,
                             const std::shared_ptr<ov::Model>& model,
                             ov::Shape inputShape,
                             int seed) {
        InferResult result;
        try {
            auto input = ov::make_tensor(ov::element::f32, inputShape);
            fillRandom(input, -5.0f, 5.0f, seed);
            req->set_tensor(model->get_parameters()[0]->output(0), input);

            req->start_async();
            req->wait();

            auto output = req->get_tensor(model->get_results()[0]->output(0));
            if (output->get_shape() != inputShape) {
                result.error = "Output shape mismatch";
                return result;
            }
            const auto* inData = static_cast<const float*>(input->data());
            const auto* outData = static_cast<const float*>(output->data());
            for (size_t i = 0; i < input->get_size(); ++i) {
                float expected = std::max(0.0f, inData[i]);
                if (std::abs(outData[i] - expected) > 1e-5f) {
                    result.error = "Data mismatch at index " + std::to_string(i);
                    return result;
                }
            }
            result.success = true;
        } catch (const std::exception& e) {
            result.error = e.what();
        }
        return result;
    }
};

TEST_F(DynamicConcurrentAsyncTest, TwoRequestsSameBatch) {
    auto model = createDynamicReluModel(4);
    auto compiled = compileModel(model);
    auto req1 = compiled->create_infer_request();
    auto req2 = compiled->create_infer_request();

    auto f1 = std::async(std::launch::async, [&] {
        return runAsyncRelu(req1, model, {4, 4}, 42);
    });
    auto f2 = std::async(std::launch::async, [&] {
        return runAsyncRelu(req2, model, {4, 4}, 123);
    });

    auto r1 = f1.get();
    auto r2 = f2.get();
    EXPECT_TRUE(r1.success) << r1.error;
    EXPECT_TRUE(r2.success) << r2.error;
}

TEST_F(DynamicConcurrentAsyncTest, TwoRequestsDifferentBatches) {
    auto model = createDynamicReluModel(4);
    auto compiled = compileModel(model);
    auto req1 = compiled->create_infer_request();
    auto req2 = compiled->create_infer_request();

    auto f1 = std::async(std::launch::async, [&] {
        return runAsyncRelu(req1, model, {3, 4}, 42);
    });
    auto f2 = std::async(std::launch::async, [&] {
        return runAsyncRelu(req2, model, {8, 4}, 123);
    });

    auto r1 = f1.get();
    auto r2 = f2.get();
    EXPECT_TRUE(r1.success) << r1.error;
    EXPECT_TRUE(r2.success) << r2.error;
}

TEST_F(DynamicConcurrentAsyncTest, FourRequestsMixedBatches) {
    auto model = createDynamicReluModel(16);
    auto compiled = compileModel(model);

    std::vector<std::shared_ptr<ov::IAsyncInferRequest>> requests;
    for (int i = 0; i < 4; ++i) requests.push_back(compiled->create_infer_request());

    std::vector<ov::Shape> shapes = {{1, 16}, {4, 16}, {8, 16}, {16, 16}};
    std::vector<std::future<InferResult>> futures;
    for (size_t i = 0; i < shapes.size(); ++i) {
        futures.push_back(std::async(std::launch::async, [&, i] {
            return runAsyncRelu(requests[i], model, shapes[i], static_cast<int>(i));
        }));
    }

    for (size_t i = 0; i < futures.size(); ++i) {
        auto r = futures[i].get();
        EXPECT_TRUE(r.success) << "Request " << i << ": " << r.error;
    }
}

TEST_F(DynamicConcurrentAsyncTest, RepeatedConcurrentInferences) {
    auto model = createDynamicReluModel(4);
    auto compiled = compileModel(model);
    auto req1 = compiled->create_infer_request();
    auto req2 = compiled->create_infer_request();

    for (int round = 0; round < 5; ++round) {
        auto f1 = std::async(std::launch::async, [&, round] {
            return runAsyncRelu(req1, model, {static_cast<size_t>(round + 1), 4}, round * 2);
        });
        auto f2 = std::async(std::launch::async, [&, round] {
            return runAsyncRelu(req2, model, {static_cast<size_t>(round + 3), 4}, round * 2 + 1);
        });

        auto r1 = f1.get();
        auto r2 = f2.get();
        EXPECT_TRUE(r1.success) << "Round " << round << " req 1: " << r1.error;
        EXPECT_TRUE(r2.success) << "Round " << round << " req 2: " << r2.error;
    }
}
