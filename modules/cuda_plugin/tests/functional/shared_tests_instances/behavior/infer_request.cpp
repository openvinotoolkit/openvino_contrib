// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

// TODO remove when GTest ASSERT_NE(nullptr, ptr) macro will be fixed
#if defined(_WIN32)
#include "fix_win32_gtest_assert_ne_macro.hpp"
#endif

#include <fmt/format.h>

#include <cuda_test_constants.hpp>
#include <error.hpp>
#include <inference_engine.hpp>
#include <vector>

#include "behavior/infer_request.hpp"
using namespace InferenceEngine;

/**
 * @brief Determine if InferenceEngine blob means image or not
 */
template<typename T>
static bool isImage(const T &blob) {
  auto descriptor = blob->getTensorDesc();
  if (descriptor.getLayout() != InferenceEngine::NCHW) {
    return false;
  }
  auto channels = descriptor.getDims()[1];
  return channels == 3;
}


/**
 * @brief Determine if InferenceEngine blob means image information or not
 */
template<typename T>
static bool isImageInfo(const T &blob) {
  auto descriptor = blob->getTensorDesc();
  if (descriptor.getLayout() != InferenceEngine::NC) {
    return false;
  }
  auto channels = descriptor.getDims()[1];
  return (channels >= 2);
}


/**
 * @brief Return height and width from provided InferenceEngine tensor description
 */
inline std::pair<size_t, size_t> getTensorHeightWidth(const InferenceEngine::TensorDesc& desc) {
  const auto& layout = desc.getLayout();
  const auto& dims = desc.getDims();
  const auto& size = dims.size();
  if ((size >= 2) &&
    (layout == InferenceEngine::Layout::NCHW  ||
     layout == InferenceEngine::Layout::NHWC  ||
     layout == InferenceEngine::Layout::NCDHW ||
     layout == InferenceEngine::Layout::NDHWC ||
     layout == InferenceEngine::Layout::OIHW  ||
     layout == InferenceEngine::Layout::GOIHW ||
     layout == InferenceEngine::Layout::OIDHW ||
     layout == InferenceEngine::Layout::GOIDHW ||
     layout == InferenceEngine::Layout::CHW  ||
     layout == InferenceEngine::Layout::HW)) {
    // Regardless of layout, dimensions are stored in fixed order
    return std::make_pair(dims.back(), dims.at(size - 2));
  } else {
      CUDAPlugin::throwIEException(
          "Tensor does not have height and width dimensions");
  }
}


/**
 * @brief Fill InferenceEngine blob with image information
 */
template<typename T>
void fillBlobImInfo(Blob::Ptr& inputBlob,
          const size_t& batchSize,
          std::pair<size_t, size_t> image_size) {
  MemoryBlob::Ptr minput = as<MemoryBlob>(inputBlob);
  // locked memory holder should be alive all time while access to its buffer happens
  auto minputHolder = minput->wmap();

  auto inputBlobData = minputHolder.as<T *>();
  for (size_t b = 0; b < batchSize; b++) {
    size_t iminfoSize = inputBlob->size()/batchSize;
    for (size_t i = 0; i < iminfoSize; i++) {
      size_t index = b*iminfoSize + i;
      if (0 == i)
        inputBlobData[index] = static_cast<T>(image_size.first);
      else if (1 == i)
        inputBlobData[index] = static_cast<T>(image_size.second);
      else
        inputBlobData[index] = 1;
    }
  }
}

/**
 * @brief Generate random values in given Limits
 */
template<typename T, class Limits = std::numeric_limits<T>>
T RandomGenerator() noexcept {
  static std::random_device rd;
  if constexpr(std::is_floating_point_v<T>) {
    static std::uniform_real_distribution<T> distribution { Limits::min(), Limits::max() };
    return distribution(rd);
  } else {
	// MSVC does not allow uint8_t and int8_t to parameterize uniform_int_distribution
# if defined(_WIN32)
	using upsized_int = std::conditional<sizeof(T)==1,
			std::conditional<std::is_signed_v<T>, short, unsigned short>::type, T>::type;
    static std::uniform_int_distribution<upsized_int> distribution { Limits::min(), Limits::max() };
# else
    static std::uniform_int_distribution<T> distribution { Limits::min(), Limits::max() };
# endif
    return static_cast<T>(distribution(rd));
  }
}

/**
 * @brief Functor type for generating random values
 */
template<typename T>
using RandomFunctionType = std::function<T()>;

/**
 * @brief A default random generating function
 */
template<typename T, class Limits = std::numeric_limits<T>>
struct RandomFunction {
  T operator()() const noexcept { return RandomGenerator<T, Limits>(); }
};

/**
 * @brief Custom limits 0..MAX
 */
template<typename T>
struct PositiveLimits : std::numeric_limits<T> {
  static constexpr T min() noexcept { return 0; }
};

/**
 * @brief Custom limits 0..1
 */
template<typename T>
struct NormalLimits {
  static constexpr T min() noexcept { return 0; }
  static constexpr T max() noexcept { return 1; }
};

/**
 * @brief Generate positive random values in  range 0..numeric_limits<T>::max
 */
template<typename T>
using PositiveRandoms = RandomFunction<T, PositiveLimits<T>>;

/**
 * @brief Generate positive random values in  range 0..numeric_limits<T>::max
 */
template<typename T>
using NormalizedRandoms = RandomFunction<T, NormalLimits<T>>;

/**
 * @brief Generate positive random values in  range numeric_limits<T>::min..numeric_limits<T>::max
 */
template <typename T>
using MaxMinTRandomFunction = RandomFunction<T>;

/**
 * @brief Fill InferenceEngine blob with random values, returned by function
 */
template<typename T>
void fillBlobRandom(Blob::Ptr& inputBlob, RandomFunctionType<T> function=&RandomGenerator<T>) {
  MemoryBlob::Ptr minput = as<MemoryBlob>(inputBlob);
  // locked memory holder should be alive all time while access to its buffer happens
  auto minputHolder = minput->wmap();

  auto inputBlobData = minputHolder.as<T *>();
  for (size_t i = 0; i < inputBlob->size(); i++) {
    inputBlobData[i] = function();
  }
}

/**
 * @brief Fill InferRequest blobs with random values or image information
 */
template<template<typename T> class Randomize = MaxMinTRandomFunction>
void fillBlobs(InferenceEngine::InferRequest inferRequest,
               const InferenceEngine::ConstInputsDataMap& inputsInfo,
               const size_t& batchSize) {
  std::vector<std::pair<size_t, size_t>> input_image_sizes;
  for (const ConstInputsDataMap::value_type& item : inputsInfo) {
    if (isImage(item.second))
      input_image_sizes.push_back(getTensorHeightWidth(item.second->getTensorDesc()));
  }

  for (const ConstInputsDataMap::value_type& item : inputsInfo) {
    Blob::Ptr inputBlob = inferRequest.GetBlob(item.first);
    if (isImageInfo(inputBlob) && (input_image_sizes.size() == 1)) {
      // Fill image information
      auto image_size = input_image_sizes.at(0);
      if (item.second->getPrecision() == InferenceEngine::Precision::FP32) {
        fillBlobImInfo<float>(inputBlob, batchSize, image_size);
      } else if (item.second->getPrecision() == InferenceEngine::Precision::FP16) {
        fillBlobImInfo<short>(inputBlob, batchSize, image_size);
      } else if (item.second->getPrecision() == InferenceEngine::Precision::I32) {
        fillBlobImInfo<int32_t>(inputBlob, batchSize, image_size);
      } else {
          CUDAPlugin::throwIEException(
              "Input precision is not supported for image info!");
      }
      continue;
    }
    // Fill random
    if (item.second->getPrecision() == InferenceEngine::Precision::FP32) {
      fillBlobRandom<float>(inputBlob, Randomize<float>{});
    } else if (item.second->getPrecision() == InferenceEngine::Precision::FP16) {
      fillBlobRandom<short>(inputBlob, Randomize<short>{});
    } else if (item.second->getPrecision() == InferenceEngine::Precision::I32) {
      fillBlobRandom<int32_t>(inputBlob, Randomize<int32_t>{});
    } else if (item.second->getPrecision() == InferenceEngine::Precision::U8) {
      fillBlobRandom<uint8_t>(inputBlob, Randomize<uint8_t>{});
    } else if (item.second->getPrecision() == InferenceEngine::Precision::I8) {
      fillBlobRandom<int8_t>(inputBlob, Randomize<int8_t>{});
    } else if (item.second->getPrecision() == InferenceEngine::Precision::U16) {
      fillBlobRandom<uint16_t>(inputBlob, Randomize<uint16_t>{});
    } else if (item.second->getPrecision() == InferenceEngine::Precision::I16) {
      fillBlobRandom<int16_t>(inputBlob, Randomize<int16_t>{});
    } else {
        CUDAPlugin::throwIEException(
            fmt::format("Input precision is not supported for {}", item.first));
    }
  }
}

using namespace BehaviorTestsDefinitions;

namespace {

const std::vector<InferenceEngine::Precision> netPrecisions = {
    InferenceEngine::Precision::FP32,
    InferenceEngine::Precision::FP16
};

const std::vector<std::map<std::string, std::string>> configs = {
    {}
};

INSTANTIATE_TEST_CASE_P(smoke_BehaviorTests, InferRequestTests,
                        ::testing::Combine(
                                ::testing::ValuesIn(netPrecisions),
                                ::testing::Values(CommonTestUtils::DEVICE_CUDA),
                                ::testing::ValuesIn(configs)),
                        InferRequestTests::getTestCaseName);

class smoke_InferenceRequestTest : public testing::Test {
  void SetUp() override {
  }

  void TearDown() override {
  }
 public:
  const std::string model10 = R"V0G0N(
   <net name="Network" version="10">
         <layers>
             <layer name="in" type="Parameter" id="0" version="opset1">
                 <data element_type="f32" shape="4096"/>
                 <output>
                     <port id="0" precision="FP32">
                         <dim>4096</dim>
                     </port>
                 </output>
             </layer>
             <layer name="out" type="Convert" id="1" version="opset1">
                 <data destination_type="f32"/>
                 <input>
                     <port id="0" precision="FP32">
                         <dim>4096</dim>
                     </port>
                 </input>
                 <output>
                     <port id="1" precision="FP32">
                         <dim>4096</dim>
                     </port>
                 </output>
             </layer>
             <layer name="result" type="Result" id="2" version="opset1">
                 <input>
                     <port id="0" precision="FP32">
                         <dim>4096</dim>
                     </port>
                 </input>
             </layer>
         </layers>
         <edges>
             <edge from-layer="0" from-port="0" to-layer="1" to-port="0"/>
             <edge from-layer="1" from-port="1" to-layer="2" to-port="0"/>
         </edges>
     </net>
     )V0G0N";
};

struct InferenceRequestBasicTest : smoke_InferenceRequestTest {
  const std::string heavyModel10 = R"V0G0N(
  <net name="Network" version="10">
        <layers>
            <layer name="Parameter_790" type="Parameter" id="0" version="opset1">
                <data element_type="f32" shape="1024,4,10,16"/>
                <output>
                    <port id="0" precision="FP32">
                        <dim>1024</dim>
                        <dim>4</dim>
                        <dim>10</dim>
                        <dim>16</dim>
                    </port>
                </output>
            </layer>
            <layer name="activation" type="Sigmoid" id="1" version="opset1">
                <input>
                    <port id="0" precision="FP32">
                        <dim>1024</dim>
                        <dim>4</dim>
                        <dim>10</dim>
                        <dim>16</dim>
                    </port>
                </input>
                <output>
                    <port id="1" precision="FP32">
                        <dim>1024</dim>
                        <dim>4</dim>
                        <dim>10</dim>
                        <dim>16</dim>
                    </port>
                </output>
            </layer>
            <layer name="output" type="Result" id="2" version="opset1">
                <input>
                    <port id="0" precision="FP32">
                        <dim>1024</dim>
                        <dim>4</dim>
                        <dim>10</dim>
                        <dim>16</dim>
                    </port>
                </input>
            </layer>
        </layers>
        <edges>
            <edge from-layer="0" from-port="0" to-layer="1" to-port="0"/>
            <edge from-layer="1" from-port="1" to-layer="2" to-port="0"/>
        </edges>
    </net>
    )V0G0N";
};

TEST_F(smoke_InferenceRequestTest, ParameterResult) {
    InferenceEngine::Core ie{};
    InferenceEngine::Blob::Ptr a{};
    auto testNet = ie.ReadNetwork(model10, a);
    auto execNet = ie.LoadNetwork(testNet, "CUDA");
    InferenceEngine::InferRequest request { execNet.CreateInferRequest() };

    const InferenceEngine::ConstInputsDataMap inputsInfo{execNet.GetInputsInfo()};
    fillBlobs<NormalizedRandoms>(request, inputsInfo, 1);
    ASSERT_NO_THROW(request.Infer());
    auto inp = request.GetBlob("in");
    auto out = request.GetBlob("out");
    auto inp_data = inp->as<MemoryBlob>()->rmap();
    auto out_data = out->as<MemoryBlob>()->rmap();
    ASSERT_NE(inp_data, out_data);
    FuncTestUtils::compareBlobs(inp, out, 0.0);
}

TEST_F(smoke_InferenceRequestTest, AsyncParameterResult) {
    InferenceEngine::Core ie{};
    InferenceEngine::Blob::Ptr a{};
    auto testNet = ie.ReadNetwork(model10, a);
    auto execNet = ie.LoadNetwork(testNet, CommonTestUtils::DEVICE_CUDA);
    InferenceEngine::InferRequest inferRequest { execNet.CreateInferRequest() };
    const InferenceEngine::ConstInputsDataMap inputsInfo{execNet.GetInputsInfo()};
    fillBlobs(inferRequest, inputsInfo, 1);
    std::atomic<bool> isCallbackCalled{false};
    inferRequest.SetCompletionCallback(
        [&] {
            isCallbackCalled.store(true, std::memory_order_release);
        });
    ASSERT_NO_THROW(inferRequest.StartAsync());
    ASSERT_NO_THROW(inferRequest.Wait(5000));
    ASSERT_EQ(isCallbackCalled.load(std::memory_order_acquire), true);
}

TEST_F(InferenceRequestBasicTest, AsyncParameterResultCancel) {
    InferenceEngine::Core ie{};
    InferenceEngine::Blob::Ptr a{};
    auto testNet = ie.ReadNetwork(heavyModel10, a);
    auto execNet = ie.LoadNetwork(testNet, CommonTestUtils::DEVICE_CUDA);
    InferenceEngine::InferRequest inferRequest { execNet.CreateInferRequest() };
    const InferenceEngine::ConstInputsDataMap inputsInfo{execNet.GetInputsInfo()};
    fillBlobs(inferRequest, inputsInfo, 1);
    ASSERT_NO_THROW(inferRequest.StartAsync());
    ASSERT_NO_THROW(inferRequest.Cancel());
    ASSERT_THROW(inferRequest.Wait(5000), std::exception);
}

TEST_F(smoke_InferenceRequestTest, PerformanceCounters) {
    InferenceEngine::Core ie{};
    InferenceEngine::Blob::Ptr a{};
    std::map<std::string, std::string> config = {{ InferenceEngine::PluginConfigParams::KEY_PERF_COUNT,
                                                   InferenceEngine::PluginConfigParams::YES }};

    auto testNet = ie.ReadNetwork(model10, a);
    auto execNet = ie.LoadNetwork(testNet, CommonTestUtils::DEVICE_CUDA, config);
    InferenceEngine::InferRequest request { execNet.CreateInferRequest() };

    const InferenceEngine::ConstInputsDataMap inputsInfo{execNet.GetInputsInfo()};
    fillBlobs<NormalizedRandoms>(request, inputsInfo, 1);
    ASSERT_NO_THROW(request.Infer());
    auto perfMap = request.GetPerformanceCounts();
    ASSERT_NE(perfMap.size(), 0);
}

}  // namespace
