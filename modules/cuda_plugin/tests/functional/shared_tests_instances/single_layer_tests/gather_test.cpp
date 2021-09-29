// Copyright (C) 2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "single_layer_tests/gather.hpp"

#include <cuda_operation_registry.hpp>
#include <cuda_test_constants.hpp>

namespace LayerTestsDefinitions {

// The class was inherited to change getTestCaseName() method
// to disable printing of the indices tensors because of their large size in the tests
// printing of them sometimes leads to seg fault
class CudaGatherLayerTest : public GatherLayerTest {
public:
    static std::string getTestCaseName(const testing::TestParamInfo<gatherParamsTuple>& obj) {
        int axis;
        std::vector<int> indices;
        std::vector<size_t> indicesShape, inputShape;
        InferenceEngine::Precision netPrecision;
        InferenceEngine::Precision inPrc, outPrc;
        InferenceEngine::Layout inLayout, outLayout;
        std::string targetName;
        std::tie(
            indices, indicesShape, axis, inputShape, netPrecision, inPrc, outPrc, inLayout, outLayout, targetName) =
            obj.param;
        std::ostringstream result;
        result << "IS=" << CommonTestUtils::vec2str(inputShape) << "_";
        result << "axis=" << axis << "_";
        result << "indicesShape=" << CommonTestUtils::vec2str(indicesShape) << "_";
        result << "netPRC=" << netPrecision.name() << "_";
        result << "inPRC=" << inPrc.name() << "_";
        result << "outPRC=" << outPrc.name() << "_";
        result << "inL=" << inLayout << "_";
        result << "outL=" << outLayout << "_";
        result << "trgDev=" << targetName << "_";
        return result.str();
    }
};

TEST_P(CudaGatherLayerTest, CompareWithRefs) {
    SKIP_IF_CURRENT_TEST_IS_DISABLED()

    gatherParamsTuple params = GetParam();
    inPrc = std::get<5>(params);
    outPrc = std::get<6>(params);

    Run();
}

}  // namespace LayerTestsDefinitions

using namespace LayerTestsDefinitions;

namespace {

struct GatherTestParams {
    std::vector<size_t> params_shape_;
    std::vector<size_t> indices_shape_;
    int axis_ = 0;
    std::vector<InferenceEngine::Precision> net_precisions_ = {InferenceEngine::Precision::FP32,
                                                               InferenceEngine::Precision::FP16};
    InferenceEngine::Precision input_precision_ = InferenceEngine::Precision::UNSPECIFIED;
    InferenceEngine::Precision output_precision_ = InferenceEngine::Precision::UNSPECIFIED;
    InferenceEngine::Layout input_layout_ = InferenceEngine::Layout::ANY;
    InferenceEngine::Layout output_layout_ = InferenceEngine::Layout::ANY;
    LayerTestsUtils::TargetDevice device_ = CommonTestUtils::DEVICE_CUDA;
};

template <typename T>
std::vector<T> generate_indices(const GatherTestParams& test_params) {
    static std::random_device r_device;
    static std::default_random_engine r_engine{r_device()};

    const auto params_shape_size = test_params.params_shape_.size();
    const auto axis = test_params.axis_;
    const unsigned positive_axis = axis >= 0 ? axis : axis + params_shape_size;
    if (positive_axis >= params_shape_size) {
        THROW_IE_EXCEPTION << "positive_axis >= params_shape_size: " << positive_axis << " >= " << params_shape_size;
    }
    std::uniform_int_distribution<T> distr(0, test_params.params_shape_[positive_axis] - 1);

    const auto indices_size = ngraph::shape_size(test_params.indices_shape_);
    std::vector<T> indices(indices_size);
    std::generate(indices.begin(), indices.end(), [&]() { return distr(r_engine); });
    return indices;
}

const GatherTestParams smoke_01_params = {{4, 3}, {2}};

INSTANTIATE_TEST_CASE_P(smoke_Gather_01,
                        CudaGatherLayerTest,
                        ::testing::Combine(::testing::Values(generate_indices<int>(smoke_01_params)),
                                           ::testing::Values(smoke_01_params.indices_shape_),
                                           ::testing::Values(smoke_01_params.axis_),
                                           ::testing::Values(smoke_01_params.params_shape_),
                                           ::testing::ValuesIn(smoke_01_params.net_precisions_),
                                           ::testing::Values(smoke_01_params.input_precision_),
                                           ::testing::Values(smoke_01_params.output_precision_),
                                           ::testing::Values(smoke_01_params.input_layout_),
                                           ::testing::Values(smoke_01_params.output_layout_),
                                           ::testing::Values(smoke_01_params.device_)),
                        CudaGatherLayerTest::getTestCaseName);

const GatherTestParams smoke_02_params = {{1, 2, 3}, {1}, 1};

INSTANTIATE_TEST_CASE_P(smoke_Gather_02,
                        CudaGatherLayerTest,
                        ::testing::Combine(::testing::Values(generate_indices<int>(smoke_02_params)),
                                           ::testing::Values(smoke_02_params.indices_shape_),
                                           ::testing::Values(smoke_02_params.axis_),
                                           ::testing::Values(smoke_02_params.params_shape_),
                                           ::testing::ValuesIn(smoke_02_params.net_precisions_),
                                           ::testing::Values(smoke_02_params.input_precision_),
                                           ::testing::Values(smoke_02_params.output_precision_),
                                           ::testing::Values(smoke_02_params.input_layout_),
                                           ::testing::Values(smoke_02_params.output_layout_),
                                           ::testing::Values(smoke_02_params.device_)),
                        CudaGatherLayerTest::getTestCaseName);

const GatherTestParams smoke_03_params = {{1, 2, 3}, {7}, -2};

INSTANTIATE_TEST_CASE_P(smoke_Gather_03,
                        CudaGatherLayerTest,
                        ::testing::Combine(::testing::Values(generate_indices<int>(smoke_03_params)),
                                           ::testing::Values(smoke_03_params.indices_shape_),
                                           ::testing::Values(smoke_03_params.axis_),
                                           ::testing::Values(smoke_03_params.params_shape_),
                                           ::testing::ValuesIn(smoke_03_params.net_precisions_),
                                           ::testing::Values(smoke_03_params.input_precision_),
                                           ::testing::Values(smoke_03_params.output_precision_),
                                           ::testing::Values(smoke_03_params.input_layout_),
                                           ::testing::Values(smoke_03_params.output_layout_),
                                           ::testing::Values(smoke_03_params.device_)),
                        CudaGatherLayerTest::getTestCaseName);

const GatherTestParams smoke_04_params = {{1, 2, 3}, {7, 5}, 1};

INSTANTIATE_TEST_CASE_P(smoke_Gather_04,
                        CudaGatherLayerTest,
                        ::testing::Combine(::testing::Values(generate_indices<int>(smoke_04_params)),
                                           ::testing::Values(smoke_04_params.indices_shape_),
                                           ::testing::Values(smoke_04_params.axis_),
                                           ::testing::Values(smoke_04_params.params_shape_),
                                           ::testing::ValuesIn(smoke_04_params.net_precisions_),
                                           ::testing::Values(smoke_04_params.input_precision_),
                                           ::testing::Values(smoke_04_params.output_precision_),
                                           ::testing::Values(smoke_04_params.input_layout_),
                                           ::testing::Values(smoke_04_params.output_layout_),
                                           ::testing::Values(smoke_04_params.device_)),
                        CudaGatherLayerTest::getTestCaseName);

// ------------- Tacotron2 shapes -------------
const GatherTestParams tacotron2_enc_params = {{148, 512}, {1, 1000}};

INSTANTIATE_TEST_CASE_P(Gather_Tacotron2_shapes_enc,
                        CudaGatherLayerTest,
                        ::testing::Combine(::testing::Values(generate_indices<int>(tacotron2_enc_params)),
                                           ::testing::Values(tacotron2_enc_params.indices_shape_),
                                           ::testing::Values(tacotron2_enc_params.axis_),
                                           ::testing::Values(tacotron2_enc_params.params_shape_),
                                           ::testing::ValuesIn(tacotron2_enc_params.net_precisions_),
                                           ::testing::Values(tacotron2_enc_params.input_precision_),
                                           ::testing::Values(tacotron2_enc_params.output_precision_),
                                           ::testing::Values(tacotron2_enc_params.input_layout_),
                                           ::testing::Values(tacotron2_enc_params.output_layout_),
                                           ::testing::Values(tacotron2_enc_params.device_)),
                        CudaGatherLayerTest::getTestCaseName);

// ------------- LPCNet shapes -------------
const GatherTestParams lpcnet_enc_params = {{256, 64}, {64, 64, 1}};

INSTANTIATE_TEST_CASE_P(Gather_LPCnet_shapes_enc,
                        CudaGatherLayerTest,
                        ::testing::Combine(::testing::Values(generate_indices<int>(lpcnet_enc_params)),
                                           ::testing::Values(lpcnet_enc_params.indices_shape_),
                                           ::testing::Values(lpcnet_enc_params.axis_),
                                           ::testing::Values(lpcnet_enc_params.params_shape_),
                                           ::testing::ValuesIn(lpcnet_enc_params.net_precisions_),
                                           ::testing::Values(lpcnet_enc_params.input_precision_),
                                           ::testing::Values(lpcnet_enc_params.output_precision_),
                                           ::testing::Values(lpcnet_enc_params.input_layout_),
                                           ::testing::Values(lpcnet_enc_params.output_layout_),
                                           ::testing::Values(lpcnet_enc_params.device_)),
                        CudaGatherLayerTest::getTestCaseName);

const GatherTestParams lpcnet_dec_params = {{256, 128}, {64, 64, 3}};

INSTANTIATE_TEST_CASE_P(Gather_LPCnet_shapes_dec,
                        CudaGatherLayerTest,
                        ::testing::Combine(::testing::Values(generate_indices<int>(lpcnet_dec_params)),
                                           ::testing::Values(lpcnet_dec_params.indices_shape_),
                                           ::testing::Values(lpcnet_dec_params.axis_),
                                           ::testing::Values(lpcnet_dec_params.params_shape_),
                                           ::testing::ValuesIn(lpcnet_dec_params.net_precisions_),
                                           ::testing::Values(lpcnet_dec_params.input_precision_),
                                           ::testing::Values(lpcnet_dec_params.output_precision_),
                                           ::testing::Values(lpcnet_dec_params.input_layout_),
                                           ::testing::Values(lpcnet_dec_params.output_layout_),
                                           ::testing::Values(lpcnet_dec_params.device_)),
                        CudaGatherLayerTest::getTestCaseName);

// ------------- Big shapes -------------
const GatherTestParams ov_v1_params = {{6, 12, 10, 24}, {15, 4, 20, 28}, 1};

INSTANTIATE_TEST_CASE_P(Gather_OV_v1,
                        CudaGatherLayerTest,
                        ::testing::Combine(::testing::Values(generate_indices<int>(ov_v1_params)),
                                           ::testing::Values(ov_v1_params.indices_shape_),
                                           ::testing::Values(ov_v1_params.axis_),
                                           ::testing::Values(ov_v1_params.params_shape_),
                                           ::testing::ValuesIn(ov_v1_params.net_precisions_),
                                           ::testing::Values(ov_v1_params.input_precision_),
                                           ::testing::Values(ov_v1_params.output_precision_),
                                           ::testing::Values(ov_v1_params.input_layout_),
                                           ::testing::Values(ov_v1_params.output_layout_),
                                           ::testing::Values(ov_v1_params.device_)),
                        CudaGatherLayerTest::getTestCaseName);

const GatherTestParams big_01_params = {{2048, 32, 2048}, {2, 8, 1, 1}, 0, {InferenceEngine::Precision::FP32}};

INSTANTIATE_TEST_CASE_P(Gather_big_01,
                        CudaGatherLayerTest,
                        ::testing::Combine(::testing::Values(generate_indices<int>(big_01_params)),
                                           ::testing::Values(big_01_params.indices_shape_),
                                           ::testing::Values(big_01_params.axis_),
                                           ::testing::Values(big_01_params.params_shape_),
                                           ::testing::ValuesIn(big_01_params.net_precisions_),
                                           ::testing::Values(big_01_params.input_precision_),
                                           ::testing::Values(big_01_params.output_precision_),
                                           ::testing::Values(big_01_params.input_layout_),
                                           ::testing::Values(big_01_params.output_layout_),
                                           ::testing::Values(big_01_params.device_)),
                        CudaGatherLayerTest::getTestCaseName);

const GatherTestParams big_02_params = {{2048, 32, 2048}, {2, 8, 1, 1}, 2, {InferenceEngine::Precision::FP16}};

INSTANTIATE_TEST_CASE_P(Gather_big_02,
                        CudaGatherLayerTest,
                        ::testing::Combine(::testing::Values(generate_indices<int>(big_02_params)),
                                           ::testing::Values(big_02_params.indices_shape_),
                                           ::testing::Values(big_02_params.axis_),
                                           ::testing::Values(big_02_params.params_shape_),
                                           ::testing::ValuesIn(big_02_params.net_precisions_),
                                           ::testing::Values(big_02_params.input_precision_),
                                           ::testing::Values(big_02_params.output_precision_),
                                           ::testing::Values(big_02_params.input_layout_),
                                           ::testing::Values(big_02_params.output_layout_),
                                           ::testing::Values(big_02_params.device_)),
                        CudaGatherLayerTest::getTestCaseName);

using ParamsVec = std::vector<std::reference_wrapper<const GatherTestParams>>;
const ParamsVec all_params = {smoke_01_params,
                              smoke_02_params,
                              smoke_03_params,
                              smoke_04_params,
                              tacotron2_enc_params,
                              lpcnet_enc_params,
                              lpcnet_dec_params,
                              ov_v1_params,
                              big_01_params,
                              big_02_params};

template <typename ElementType, typename IndicesType>
void test_one_shape(const GatherTestParams& params) {
    using devptr_t = CUDA::DevicePointer<void*>;
    using cdevptr_t = CUDA::DevicePointer<const void*>;
    using microseconds = std::chrono::duration<double, std::micro>;
    using milliseconds = std::chrono::duration<double, std::milli>;

    constexpr int NUM_ATTEMPTS = 20;
    constexpr milliseconds WARMUP_TIME{2000.0};

    CUDA::ThreadContext threadContext{{}};
    int out_size = 0;
    CUDAPlugin::OperationBase::Ptr operation = [&] {
        const bool optimizeOption = false;
        auto dict_param = std::make_shared<ngraph::op::v0::Parameter>(ngraph::element::from<ElementType>(),
                                                                      ngraph::PartialShape{params.params_shape_});
        auto indices_param = std::make_shared<ngraph::op::v0::Parameter>(ngraph::element::from<IndicesType>(),
                                                                         ngraph::PartialShape{params.indices_shape_});
        auto axis_constant = std::make_shared<ngraph::op::v0::Constant>(
            ngraph::element::i64, ngraph::Shape({}), std::vector<int64_t>{params.axis_});
        auto node =
            std::make_shared<ngraph::op::v1::Gather>(dict_param->output(0), indices_param->output(0), axis_constant);
        out_size = ngraph::shape_size(node->get_output_shape(0));
        auto& registry = CUDAPlugin::OperationRegistry::getInstance();
        auto op = registry.createOperation(CUDA::CreationContext{threadContext.device(), optimizeOption},
                                           node,
                                           std::array{CUDAPlugin::TensorID{0}},
                                           std::array{CUDAPlugin::TensorID{0}});
        return op;
    }();
    const int dict_size = ngraph::shape_size(params.params_shape_);
    const int indices_size = ngraph::shape_size(params.indices_shape_);
    const auto dict_size_bytes = dict_size * sizeof(ElementType);
    const auto indices_size_bytes = indices_size * sizeof(IndicesType);
    const auto out_size_bytes = out_size * sizeof(ElementType);
    CUDA::Allocation dict_alloc = threadContext.stream().malloc(dict_size_bytes);
    CUDA::Allocation indices_alloc = threadContext.stream().malloc(indices_size_bytes);
    CUDA::Allocation axis_alloc = threadContext.stream().malloc(sizeof(int64_t));
    CUDA::Allocation out_alloc = threadContext.stream().malloc(out_size_bytes);
    std::vector<cdevptr_t> inputs{dict_alloc, indices_alloc, axis_alloc};
    std::vector<devptr_t> outputs{out_alloc};

    InferenceEngine::BlobMap empty;
    InferenceEngine::gpu::InferenceRequestContext context{empty, empty, threadContext};
    std::vector<IndicesType> indices = generate_indices<IndicesType>(params);
    std::vector<ElementType> dict(dict_size);
    std::random_device r_device;
    std::mt19937 mersenne_engine{r_device()};
    std::uniform_int_distribution<int> dist{std::numeric_limits<int>::min(), std::numeric_limits<int>::max()};
    auto gen_dict = [&dist, &mersenne_engine]() {
        return static_cast<IndicesType>(10.f * dist(mersenne_engine) / std::numeric_limits<int>::max());
    };
    std::generate(dict.begin(), dict.end(), gen_dict);

    auto& stream = context.getThreadContext().stream();
    stream.upload(dict_alloc, dict.data(), dict_size_bytes);
    stream.upload(indices_alloc, indices.data(), indices_size_bytes);
    CUDAPlugin::Workbuffers workbuffers{};

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
    std::cout << std::fixed << std::setfill('0') << "Gather " << CommonTestUtils::vec2str(params.params_shape_) << ", "
              << CommonTestUtils::vec2str(params.indices_shape_) << ", " << params.axis_ << ": "
              << average_exec_time.count() << " us\n";
}

template <typename ElementType, typename IndicesType>
void test_all_shapes(const ParamsVec& all_params) {
    for (const auto& p : all_params) {
        test_one_shape<ElementType, IndicesType>(p);
    }
}

struct GatherBenchmark : testing::Test {};

TEST_F(GatherBenchmark, DISABLED_benchmark) {
    std::cout << "---Dicts: float, Indices: int32_t:\n";
    test_all_shapes<float, int32_t>(all_params);

    std::cout << "---Dicts: float, Indices: int64_t:\n";
    test_all_shapes<float, int64_t>(all_params);

    std::cout << "---Dicts: ngraph::float16, Indices: int32_t:\n";
    test_all_shapes<ngraph::float16, int32_t>(all_params);
}

}  // namespace
