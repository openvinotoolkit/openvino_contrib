// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "unit/gfx_backend_architecture_contract_common.hpp"

namespace ov {
namespace gfx_plugin {
namespace {

TEST_F(GfxBackendArchitectureContractTest,
       AppleTransposedSdpaVendorContractOwnsShapeAndTypeContract) {
  GfxAppleMpsVendorPrimitiveContract contract;
  EXPECT_TRUE(gfx_apple_make_mps_transposed_sdpa_contract(
      "attention", ov::element::f32, {1, 2, 3, 4}, {1, 2, 3, 5}, {1, 2, 6, 5},
      {1, 2, 6, 4}, 0.5f, contract));
  EXPECT_TRUE(contract.valid);
  EXPECT_EQ(contract.descriptor.kind, GfxAppleMpsVendorPrimitiveKind::Sdpa);
  EXPECT_EQ(contract.descriptor.sdpa.layout, GfxMpsrtSdpaLayoutTransposedBHDN);
  EXPECT_FLOAT_EQ(contract.descriptor.sdpa.scale, 0.5f);
  ASSERT_TRUE(contract.external_buffer_abi.valid);
  EXPECT_EQ(contract.external_buffer_abi.buffer_count, 4u);
  EXPECT_EQ(contract.external_buffer_abi.output_buffer_count, 1u);

  EXPECT_FALSE(gfx_apple_make_mps_transposed_sdpa_contract(
      "attention", ov::element::i32, {1, 2, 3, 4}, {1, 2, 3, 5}, {1, 2, 6, 5},
      {1, 2, 6, 4}, 0.5f, contract));
  EXPECT_FALSE(gfx_apple_make_mps_transposed_sdpa_contract(
      "attention", ov::element::f32, {1, 2, 3, 4}, {1, 2, 3, 5}, {1, 2, 6, 5},
      {1, 2, 7, 4}, 0.5f, contract));
  EXPECT_FALSE(gfx_apple_make_mps_transposed_sdpa_contract(
      "attention", ov::element::f32, {1, 2, 3, 4}, {1, 2, 3}, {1, 2, 6, 5},
      {1, 2, 6, 4}, 0.5f, contract));
}

} // namespace
} // namespace gfx_plugin
} // namespace ov
