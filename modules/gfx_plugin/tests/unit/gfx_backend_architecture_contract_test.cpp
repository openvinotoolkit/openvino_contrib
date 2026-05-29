// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <gtest/gtest.h>

#include "unit/gfx_backend_contracts.hpp"

using ov::gfx_plugin::compiler::KernelUnitKind;
using ov::gfx_plugin::compiler::LoweringRouteKind;

namespace ov {
namespace gfx_plugin {
namespace {

class GfxBackendArchitectureContractTest : public ::testing::Test {
protected:
    test::BackendContractCatalog backend_catalog;
    test::ModelContractFactory models;
};

TEST_F(GfxBackendArchitectureContractTest,
       KnownTargetsUseConcreteOopIdentityWithoutInverseBuckets) {
    for (const auto& target_contract :
         backend_catalog.known_target_contracts()) {
        EXPECT_TRUE(target_contract.has_concrete_oop_identity())
            << target_contract.target().debug_string();
        EXPECT_TRUE(target_contract.avoids_inverse_apple_bucket())
            << target_contract.target().debug_string();
    }
}

TEST_F(GfxBackendArchitectureContractTest,
       KernelRegistriesRequireExplicitOpOwnedUnits) {
    const auto opencl_registry = test::KernelRegistryContract::for_opencl();
    ASSERT_TRUE(opencl_registry.audit_is_valid());

    EXPECT_TRUE(opencl_registry.rejects_unit(LoweringRouteKind::GeneratedKernel,
                                             "opencl_generated_kernel"));
    const auto matmul_unit =
        opencl_registry.resolve_unit(LoweringRouteKind::GeneratedKernel,
                                     "opencl/generated/matmul_f32");
    ASSERT_TRUE(matmul_unit.valid());
    EXPECT_EQ(matmul_unit.kind(), KernelUnitKind::GeneratedKernel);
    EXPECT_EQ(matmul_unit.backend_domain(), "opencl");
    EXPECT_EQ(matmul_unit.op_family(), "MatMul");
    const auto eltwise_unit =
        opencl_registry.resolve_unit(LoweringRouteKind::GeneratedKernel,
                                     "opencl/generated/eltwise_binary_f32");
    ASSERT_TRUE(eltwise_unit.valid());
    EXPECT_EQ(eltwise_unit.kind(), KernelUnitKind::GeneratedKernel);
    EXPECT_EQ(eltwise_unit.backend_domain(), "opencl");
    EXPECT_EQ(eltwise_unit.op_family(), "Eltwise");
    const auto activation_unit =
        opencl_registry.resolve_unit(LoweringRouteKind::GeneratedKernel,
                                     "opencl/generated/activation_f32");
    ASSERT_TRUE(activation_unit.valid());
    EXPECT_EQ(activation_unit.kind(), KernelUnitKind::GeneratedKernel);
    EXPECT_EQ(activation_unit.backend_domain(), "opencl");
    EXPECT_EQ(activation_unit.op_family(), "Activation");
    const auto pool_unit =
        opencl_registry.resolve_unit(LoweringRouteKind::GeneratedKernel,
                                     "opencl/generated/pool2d_f32");
    ASSERT_TRUE(pool_unit.valid());
    EXPECT_EQ(pool_unit.kind(), KernelUnitKind::GeneratedKernel);
    EXPECT_EQ(pool_unit.backend_domain(), "opencl");
    EXPECT_EQ(pool_unit.op_family(), "Pooling");

    const auto metal_registry = test::KernelRegistryContract::for_metal();
    ASSERT_TRUE(metal_registry.audit_is_valid());
    EXPECT_TRUE(metal_registry
                    .resolve_unit(LoweringRouteKind::VendorPrimitive,
                                  "metal/vendor/mps_gemm")
                    .valid());
    EXPECT_TRUE(metal_registry
                    .resolve_unit(LoweringRouteKind::GeneratedKernel,
                                  "metal/generated/slice")
                    .valid());
    const auto metal_eltwise_unit =
        metal_registry.resolve_unit(LoweringRouteKind::GeneratedKernel,
                                    "metal/generated/eltwise");
    ASSERT_TRUE(metal_eltwise_unit.valid());
    EXPECT_EQ(metal_eltwise_unit.op_family(), "Eltwise");
    const auto metal_activation_unit =
        metal_registry.resolve_unit(LoweringRouteKind::GeneratedKernel,
                                    "metal/generated/activation");
    ASSERT_TRUE(metal_activation_unit.valid());
    EXPECT_EQ(metal_activation_unit.op_family(), "Activation");
    const auto metal_pool_unit =
        metal_registry.resolve_unit(LoweringRouteKind::VendorPrimitive,
                                    "metal/vendor/mps_pool2d");
    ASSERT_TRUE(metal_pool_unit.valid());
    EXPECT_EQ(metal_pool_unit.op_family(), "Pooling");
}

TEST_F(GfxBackendArchitectureContractTest,
       RegisteredBackendModulesShareTheSamePassthroughContract) {
    const auto contracts = backend_catalog.compiled_module_contracts();
    ASSERT_FALSE(contracts.empty());

    for (const auto& module_contract : contracts) {
        const auto model = models.passthrough(
            ov::PartialShape{ov::Dimension::dynamic(), 3});
        const auto compile_result =
            module_contract.compile_without_graph_pipeline(model);
        EXPECT_TRUE(module_contract.compile_result_obeys_manifest_contract(
            compile_result))
            << module_contract.target().debug_string() << ": "
            << compile_result.unsupported_message();
    }
}

}  // namespace
}  // namespace gfx_plugin
}  // namespace ov
