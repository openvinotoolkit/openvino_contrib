// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <vector>

#include <cuda_test_constants.hpp>
#include <stress_tests/common/ie_utils.h>

#include "behavior/infer_request.hpp"

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

class InferenceEngineBasicTest : public testing::Test {
    void SetUp() override {
    }

    void TearDown() override {
    }

 public:
    const std::string model10 = R"V0G0N(
    <net name="Network" version="10">
          <layers>
              <layer name="Parameter_790" type="Parameter" id="0" version="opset1">
                  <data element_type="f32" shape="4"/>
                  <output>
                      <port id="0" precision="FP32">
                          <dim>4</dim>
                      </port>
                  </output>
              </layer>
              <layer name="activation" type="Sigmoid" id="1" version="opset1">
                  <input>
                      <port id="0" precision="FP32">
                          <dim>4</dim>
                      </port>
                  </input>
                  <output>
                      <port id="1" precision="FP32">
                          <dim>4</dim>
                      </port>
                  </output>
              </layer>
              <layer name="output" type="Result" id="2" version="opset1">
                  <input>
                      <port id="0" precision="FP32">
                          <dim>4</dim>
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

TEST_F(InferenceEngineBasicTest, ParameterAndResult) {
    InferenceEngine::Core ie{};
    InferenceEngine::Blob::Ptr a{};
    auto testNet = ie.ReadNetwork(model10, a);
    auto execNet = ie.LoadNetwork(testNet, "CUDA");
    InferenceEngine::InferRequest inferRequest { execNet.CreateInferRequest() };

    const InferenceEngine::ConstInputsDataMap inputsInfo{execNet.GetInputsInfo()};
    fillBlobs(inferRequest, inputsInfo, 1);
    ASSERT_NO_THROW(inferRequest.Infer());
}

TEST_F(InferenceEngineBasicTest, AsyncParameterAndResult) {
    InferenceEngine::Core ie{};
    InferenceEngine::Blob::Ptr a{};
    auto testNet = ie.ReadNetwork(model10, a);
    auto execNet = ie.LoadNetwork(testNet, "CUDA");
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

TEST_F(InferenceEngineBasicTest, AsyncParameterAndResultCancel) {
    InferenceEngine::Core ie{};
    InferenceEngine::Blob::Ptr a{};
    auto testNet = ie.ReadNetwork(heavyModel10, a);
    auto execNet = ie.LoadNetwork(testNet, "CUDA");
    InferenceEngine::InferRequest inferRequest { execNet.CreateInferRequest() };
    const InferenceEngine::ConstInputsDataMap inputsInfo{execNet.GetInputsInfo()};
    fillBlobs(inferRequest, inputsInfo, 1);
    ASSERT_NO_THROW(inferRequest.StartAsync());
    ASSERT_NO_THROW(inferRequest.Cancel());
    ASSERT_NO_THROW(inferRequest.Wait(5000));
}

}  // namespace
