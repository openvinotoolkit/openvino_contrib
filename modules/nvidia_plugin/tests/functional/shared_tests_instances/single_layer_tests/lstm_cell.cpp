// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "single_layer_tests/lstm_cell.hpp"

#include <cuda_profiler.hpp>
#include <cuda_test_constants.hpp>
#include <functional>
#include <vector>

#include "cuda/device_pointers.hpp"
#include "cuda_operation_registry.hpp"
#include "cuda_thread_context.hpp"

namespace LayerTestsDefinitions {

class CUDNNLSTMCellTest : public LSTMCellTest {
public:
    void SetUp() override {
        LSTMCellTest::SetUp();
        constexpr float up_to = 5.0f;
        constexpr float start_from = -5.0f;

        const auto& ops = function->get_ordered_ops();
        int seed = 1;
        for (const auto& op : ops) {
            if (std::dynamic_pointer_cast<ngraph::opset1::Constant>(op)) {
                const auto constant = ngraph::builder::makeConstant(
                    op->get_element_type(), op->get_shape(), std::vector<float>{}, true, up_to, start_from, seed);
                function->replace_node(op, constant);
                ++seed;
            }
        }
    }
};

TEST_P(CUDNNLSTMCellTest, CompareWithRefs) {
    SKIP_IF_CURRENT_TEST_IS_DISABLED()
    Run();
};

}  // namespace LayerTestsDefinitions

using namespace LayerTestsDefinitions;

namespace {
const bool should_decompose = false;
const std::vector<std::string> activations{"sigmoid", "tanh", "tanh"};
const std::vector<InferenceEngine::Precision> netPrecisions = {InferenceEngine::Precision::FP32,
                                                               InferenceEngine::Precision::FP16};

// ------------- Smoke shapes -------------
const std::vector<size_t> smoke_batch{1, 5};

// Currently LSTMCell cuDNN implementation doesn't support clipping
const std::vector<float> smoke_clip{0.0f};

// Currently LSTMCell cuDNN implementation doesn't support combination of input_size == 1 and hidden_size == 1
const std::vector<size_t> smoke_input_sizes_01{1, 2, 3, 30};
const std::vector<size_t> smoke_hidden_sizes_01{2, 3, 10};

INSTANTIATE_TEST_CASE_P(smoke_LSTMCell_01,
                        CUDNNLSTMCellTest,
                        ::testing::Combine(::testing::Values(should_decompose),
                                           ::testing::ValuesIn(smoke_batch),
                                           ::testing::ValuesIn(smoke_hidden_sizes_01),
                                           ::testing::ValuesIn(smoke_input_sizes_01),
                                           ::testing::Values(activations),
                                           ::testing::ValuesIn(smoke_clip),
                                           ::testing::ValuesIn(netPrecisions),
                                           ::testing::Values(CommonTestUtils::DEVICE_NVIDIA)),
                        CUDNNLSTMCellTest::getTestCaseName);

const std::vector<size_t> smoke_input_sizes_02{2, 3, 30};
size_t smoke_hidden_size_02 = 1;

INSTANTIATE_TEST_CASE_P(smoke_LSTMCell_02,
                        CUDNNLSTMCellTest,
                        ::testing::Combine(::testing::Values(should_decompose),
                                           ::testing::ValuesIn(smoke_batch),
                                           ::testing::Values(smoke_hidden_size_02),
                                           ::testing::ValuesIn(smoke_input_sizes_02),
                                           ::testing::Values(activations),
                                           ::testing::ValuesIn(smoke_clip),
                                           ::testing::ValuesIn(netPrecisions),
                                           ::testing::Values(CommonTestUtils::DEVICE_NVIDIA)),
                        CUDNNLSTMCellTest::getTestCaseName);

// ------------- Other shapes -------------
struct LSTMCellTestParams {
    size_t batch;
    size_t input_size;
    size_t hidden_size;
    float clip;
};

// ------------- Tacotron2 shapes -------------
const LSTMCellTestParams tacotron2_dec_01{1, 768, 1024, 0.0f};

INSTANTIATE_TEST_CASE_P(LSTMCell_Tacotron2_dec_01,
                        CUDNNLSTMCellTest,
                        ::testing::Combine(::testing::Values(should_decompose),
                                           ::testing::Values(tacotron2_dec_01.batch),
                                           ::testing::Values(tacotron2_dec_01.hidden_size),
                                           ::testing::Values(tacotron2_dec_01.input_size),
                                           ::testing::Values(activations),
                                           ::testing::Values(tacotron2_dec_01.clip),
                                           ::testing::ValuesIn(netPrecisions),
                                           ::testing::Values(CommonTestUtils::DEVICE_NVIDIA)),
                        CUDNNLSTMCellTest::getTestCaseName);

LSTMCellTestParams tacotron2_dec_02{1, 1536, 1024, 0.0f};

INSTANTIATE_TEST_CASE_P(LSTMCell_Tacotron2_dec_02,
                        CUDNNLSTMCellTest,
                        ::testing::Combine(::testing::Values(should_decompose),
                                           ::testing::Values(tacotron2_dec_02.batch),
                                           ::testing::Values(tacotron2_dec_02.hidden_size),
                                           ::testing::Values(tacotron2_dec_02.input_size),
                                           ::testing::Values(activations),
                                           ::testing::Values(tacotron2_dec_02.clip),
                                           ::testing::ValuesIn(netPrecisions),
                                           ::testing::Values(CommonTestUtils::DEVICE_NVIDIA)),
                        CUDNNLSTMCellTest::getTestCaseName);

LSTMCellTestParams tacotron2_enc_01{1, 512, 256, 0.0f};

INSTANTIATE_TEST_CASE_P(LSTMCell_Tacotron2_enc_01,
                        CUDNNLSTMCellTest,
                        ::testing::Combine(::testing::Values(should_decompose),
                                           ::testing::Values(tacotron2_enc_01.batch),
                                           ::testing::Values(tacotron2_enc_01.hidden_size),
                                           ::testing::Values(tacotron2_enc_01.input_size),
                                           ::testing::Values(activations),
                                           ::testing::Values(tacotron2_enc_01.clip),
                                           ::testing::ValuesIn(netPrecisions),
                                           ::testing::Values(CommonTestUtils::DEVICE_NVIDIA)),
                        CUDNNLSTMCellTest::getTestCaseName);

// ------------- Big shapes -------------
LSTMCellTestParams ov_doc_01{1, 16, 128, 0.0f};

INSTANTIATE_TEST_CASE_P(LSTMCell_OV_Doc_01,
                        CUDNNLSTMCellTest,
                        ::testing::Combine(::testing::Values(should_decompose),
                                           ::testing::Values(ov_doc_01.batch),
                                           ::testing::Values(ov_doc_01.hidden_size),
                                           ::testing::Values(ov_doc_01.input_size),
                                           ::testing::Values(activations),
                                           ::testing::Values(ov_doc_01.clip),
                                           ::testing::ValuesIn(netPrecisions),
                                           ::testing::Values(CommonTestUtils::DEVICE_NVIDIA)),
                        CUDNNLSTMCellTest::getTestCaseName);

LSTMCellTestParams big_01{10, 2048, 2048, 0.0f};

INSTANTIATE_TEST_CASE_P(LSTMCell_Big_01,
                        CUDNNLSTMCellTest,
                        ::testing::Combine(::testing::Values(should_decompose),
                                           ::testing::Values(big_01.batch),
                                           ::testing::Values(big_01.hidden_size),
                                           ::testing::Values(big_01.input_size),
                                           ::testing::Values(activations),
                                           ::testing::Values(big_01.clip),
                                           ::testing::ValuesIn(netPrecisions),
                                           ::testing::Values(CommonTestUtils::DEVICE_NVIDIA)),
                        CUDNNLSTMCellTest::getTestCaseName);

LSTMCellTestParams big_02{1, 8192, 4096, 0.0f};

INSTANTIATE_TEST_CASE_P(LSTMCell_Big_02,
                        CUDNNLSTMCellTest,
                        ::testing::Combine(::testing::Values(should_decompose),
                                           ::testing::Values(big_02.batch),
                                           ::testing::Values(big_02.hidden_size),
                                           ::testing::Values(big_02.input_size),
                                           ::testing::Values(activations),
                                           ::testing::Values(big_02.clip),
                                           ::testing::ValuesIn(netPrecisions),
                                           ::testing::Values(CommonTestUtils::DEVICE_NVIDIA)),
                        CUDNNLSTMCellTest::getTestCaseName);

LSTMCellTestParams big_03{3, 1781, 5003, 0.0f};

INSTANTIATE_TEST_CASE_P(LSTMCell_Big_03,
                        CUDNNLSTMCellTest,
                        ::testing::Combine(::testing::Values(should_decompose),
                                           ::testing::Values(big_03.batch),
                                           ::testing::Values(big_03.hidden_size),
                                           ::testing::Values(big_03.input_size),
                                           ::testing::Values(activations),
                                           ::testing::Values(big_03.clip),
                                           ::testing::ValuesIn(netPrecisions),
                                           ::testing::Values(CommonTestUtils::DEVICE_NVIDIA)),
                        CUDNNLSTMCellTest::getTestCaseName);

// ------------- Benchmark -------------
using ParamsVec = std::vector<std::reference_wrapper<const LSTMCellTestParams>>;
const ParamsVec all_params = {tacotron2_dec_01, tacotron2_dec_02, tacotron2_enc_01, ov_doc_01, big_01, big_02, big_03};

template <typename ElementType>
void testOneShape(const LSTMCellTestParams& params) {
    using CDevPtr = CUDA::DevicePointer<const void*>;
    using DevPtr = CUDA::DevicePointer<void*>;

    using microseconds = std::chrono::duration<double, std::micro>;
    using milliseconds = std::chrono::duration<double, std::milli>;

    constexpr int NUM_ATTEMPTS = 20;
    constexpr milliseconds WARMUP_TIME{2000.0};

    const auto x_size = params.batch * params.input_size;
    const auto hi_size = params.batch * params.hidden_size;
    const auto ci_size = params.batch * params.hidden_size;
    const auto w_size = 4 * params.hidden_size * params.input_size;
    const auto r_size = 4 * params.hidden_size * params.hidden_size;
    const auto b_size = 4 * params.hidden_size;
    const auto ho_size = params.batch * params.hidden_size;
    const auto co_size = params.batch * params.hidden_size;

    ov::nvidia_gpu::ThreadContext threadContext{{}};

    CUDA::Allocation x_alloc{threadContext.stream().malloc(x_size * sizeof(ElementType))};
    CUDA::Allocation hi_alloc{threadContext.stream().malloc(hi_size * sizeof(ElementType))};
    CUDA::Allocation ci_alloc{threadContext.stream().malloc(ci_size * sizeof(ElementType))};
    CUDA::Allocation w_alloc{threadContext.stream().malloc(w_size * sizeof(ElementType))};  // Not used
    CUDA::Allocation r_alloc{threadContext.stream().malloc(r_size * sizeof(ElementType))};  // Not used
    CUDA::Allocation b_alloc{threadContext.stream().malloc(b_size * sizeof(ElementType))};  // Not used
    CUDA::Allocation ho_alloc{threadContext.stream().malloc(ho_size * sizeof(ElementType))};
    CUDA::Allocation co_alloc{threadContext.stream().malloc(co_size * sizeof(ElementType))};
    std::vector<CDevPtr> inputs{x_alloc, hi_alloc, ci_alloc, w_alloc, r_alloc, b_alloc};
    std::vector<DevPtr> outputs{ho_alloc, co_alloc};

    std::vector<std::shared_ptr<ngraph::runtime::Tensor>> emptyTensor;
    std::map<std::string, std::size_t> emptyMapping;
    ov::nvidia_gpu::CudaGraph graph{ov::nvidia_gpu::CreationContext{CUDA::Device{}, false}, {}};
    ov::nvidia_gpu::CancellationToken token{};
    ov::nvidia_gpu::Profiler profiler{false, graph};
    ov::nvidia_gpu::InferenceRequestContext context{
        emptyTensor, emptyMapping, emptyTensor, emptyMapping, threadContext, token, profiler};
    std::vector<ElementType> x_host(x_size);
    std::vector<ElementType> hi_host(hi_size);
    std::vector<ElementType> ci_host(ci_size);
    std::vector<ElementType> w_host(w_size);
    std::vector<ElementType> r_host(r_size);
    std::vector<ElementType> b_host(b_size);

    std::random_device r_device;
    std::mt19937 mersenne_engine{r_device()};
    std::uniform_int_distribution<int> dist{std::numeric_limits<int>::min(), std::numeric_limits<int>::max()};
    auto generator = [&dist, &mersenne_engine] {
        return static_cast<ElementType>(10.f * dist(mersenne_engine) /
                                        static_cast<float>(std::numeric_limits<int>::max()));
    };

    // All the weights and biases are initialized from u(-sqrt(k), sqrt(k)), where k = 1 / hidden_size
    // https://pytorch.org/docs/stable/generated/torch.nn.LSTMCell.html
    const auto k_root = std::sqrt(1.0 / static_cast<double>(params.hidden_size));
    std::uniform_real_distribution<double> w_dist{-k_root, k_root};
    auto w_generator = [&w_dist, &mersenne_engine] { return static_cast<ElementType>(w_dist(mersenne_engine)); };

    std::generate(x_host.begin(), x_host.end(), generator);
    std::generate(hi_host.begin(), hi_host.end(), generator);
    std::generate(ci_host.begin(), ci_host.end(), generator);
    std::generate(w_host.begin(), w_host.end(), w_generator);
    std::generate(r_host.begin(), r_host.end(), w_generator);
    std::generate(b_host.begin(), b_host.end(), w_generator);

    auto& stream = context.getThreadContext().stream();
    stream.upload(x_alloc, x_host.data(), x_size * sizeof(ElementType));
    stream.upload(hi_alloc, hi_host.data(), hi_size * sizeof(ElementType));
    stream.upload(ci_alloc, ci_host.data(), ci_size * sizeof(ElementType));

    const auto& type = ov::element::from<ElementType>();
    auto w_constant =
        std::make_shared<ov::op::v0::Constant>(type, ov::Shape({4 * params.hidden_size, params.input_size}), w_host);
    auto r_constant =
        std::make_shared<ov::op::v0::Constant>(type, ov::Shape({4 * params.hidden_size, params.hidden_size}), r_host);
    auto b_constant = std::make_shared<ov::op::v0::Constant>(type, ov::Shape({4 * params.hidden_size}), b_host);

    ov::nvidia_gpu::OperationBase::Ptr operation = [&] {
        const bool optimizeOption = false;

        auto x_param = std::make_shared<ov::op::v0::Parameter>(type, ov::Shape{params.batch, params.input_size});
        auto hi_param = std::make_shared<ov::op::v0::Parameter>(type, ov::Shape{params.batch, params.hidden_size});
        auto ci_param = std::make_shared<ov::op::v0::Parameter>(type, ov::Shape{params.batch, params.hidden_size});
        auto node = std::make_shared<ov::op::v4::LSTMCell>(x_param->output(0),
                                                           hi_param->output(0),
                                                           ci_param->output(0),
                                                           w_constant,
                                                           r_constant,
                                                           b_constant,
                                                           params.hidden_size);

        Ensures(ho_size == ov::shape_size(node->get_output_shape(0)));
        Ensures(co_size == ov::shape_size(node->get_output_shape(1)));

        auto& registry = ov::nvidia_gpu::OperationRegistry::getInstance();
        auto op = registry.createOperation(ov::nvidia_gpu::CreationContext{threadContext.device(), optimizeOption},
                                           node,
                                           std::array{ov::nvidia_gpu::TensorID{0}},
                                           std::array{ov::nvidia_gpu::TensorID{0}});
        return op;
    }();

    ov::nvidia_gpu::WorkbufferRequest wb_request{operation->GetWorkBufferRequest()};
    Ensures(wb_request.immutable_sizes.size() != 0);
    Ensures(wb_request.mutable_sizes.size() != 0);

    const auto seq_length_array_size_bytes = wb_request.immutable_sizes[0];
    const auto weight_space_size = wb_request.immutable_sizes.size() > 1 ? wb_request.immutable_sizes[1] : 0;
    const auto y_size_bytes = wb_request.mutable_sizes[0];
    const auto work_space_size = wb_request.mutable_sizes.size() > 1 ? wb_request.mutable_sizes[1] : 0;

    auto sla_alloc = CUDA::DefaultStream::stream().malloc(seq_length_array_size_bytes);
    auto weight_space_alloc = CUDA::DefaultStream::stream().malloc(weight_space_size);
    auto y_alloc = CUDA::DefaultStream::stream().malloc(y_size_bytes);
    auto work_space_alloc = CUDA::DefaultStream::stream().malloc(work_space_size);

    ov::nvidia_gpu::IOperationExec::Buffers init_buffers;
    init_buffers.emplace_back(DevPtr{sla_alloc.get()});
    if (weight_space_size != 0) {
        init_buffers.emplace_back(DevPtr{weight_space_alloc.get()});
    }
    operation->InitSharedImmutableWorkbuffers(init_buffers);

    ov::nvidia_gpu::Workbuffers workbuffers{};
    workbuffers.immutable_buffers.emplace_back(CDevPtr{sla_alloc.get()});
    if (weight_space_size != 0) {
        workbuffers.immutable_buffers.emplace_back(CDevPtr{weight_space_alloc.get()});
    }
    workbuffers.mutable_buffers.emplace_back(DevPtr{y_alloc.get()});
    if (work_space_size != 0) {
        workbuffers.mutable_buffers.emplace_back(DevPtr{work_space_alloc.get()});
    }

    // Warmup
    auto warm_cur = std::chrono::steady_clock::now();
    const auto warm_end = warm_cur + WARMUP_TIME;
    while (warm_cur <= warm_end) {
        operation->Execute(context, inputs, outputs, workbuffers);
        stream.synchronize();
        warm_cur = std::chrono::steady_clock::now();
    }

    // Benchmark
    const auto start = std::chrono::steady_clock::now();
    for (int i = 0; i < NUM_ATTEMPTS; ++i) {
        operation->Execute(context, inputs, outputs, workbuffers);
        stream.synchronize();
    }
    const auto end = std::chrono::steady_clock::now();
    microseconds average_exec_time = (end - start) / NUM_ATTEMPTS;
    std::cout << std::fixed << std::setfill('0') << "LSTMCell batch = " << params.batch
              << ", input_size = " << params.input_size << ", hidden_size = " << params.hidden_size
              << ", clip = " << params.clip << ": " << average_exec_time.count() << " us\n";
}

template <typename ElementType>
void testAllShapes(const ParamsVec& all_params) {
    for (const auto& p : all_params) {
        testOneShape<ElementType>(p);
    }
}

struct LSTMCellBenchmark : testing::Test {};

TEST_F(LSTMCellBenchmark, DISABLED_benchmark) {
    std::cout << "-----FP32:-----\n";
    testAllShapes<float>(all_params);

    std::cout << "-----FP16:-----\n";
    testAllShapes<ov::float16>(all_params);
}

}  // namespace
