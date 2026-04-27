// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <gtest/gtest.h>

#include <filesystem>

#include "tools/ov_gfx_microbench_calibration.hpp"

namespace ov {
namespace gfx_plugin {
namespace microbench {
namespace {

TEST(OvGfxMicrobenchCalibrationTest, ArtifactRoundTripPreservesCoreFields) {
    CalibrationArtifact artifact;
    artifact.device_key = "0x5143:0x44050001:2150760449";
    artifact.backend = "vulkan";
    artifact.device_name = "Adreno (TM) 830";
    artifact.platform = "linux_or_android";
    artifact.vendor_id = "0x5143";
    artifact.device_id = "0x44050001";
    artifact.driver_version = "2150760449";
    artifact.fixed_overhead_us = 416.25;
    artifact.bandwidth_estimate_gbps = 14.123;
    artifact.compute_estimate_tflops = 0.073;
    artifact.gpu_bandwidth_estimate_gbps = 0.0;
    artifact.gpu_compute_estimate_tflops = 0.075;
    artifact.triage_hints = {"mb0_high_fixed_overhead", "mb3_sync_bound"};
    artifact.assumptions = {"fixed_overhead_us is taken from MB0 median wall time with explicit sync."};

    CalibrationProbe probe;
    probe.name = "MB3";
    probe.actual_backend = "vulkan";
    probe.arithmetic_intensity = 341.333;
    probe.overhead_subtracted_ms = 29.53;
    probe.adjusted_gbps = 0.213;
    probe.adjusted_tflops = 0.073;
    probe.gpu_gbps = 0.219;
    probe.gpu_tflops = 0.075;
    probe.first_to_steady_ratio = 1.276;
    probe.wait_share_of_wall = 0.990;
    probe.transfer_share_of_wall = 0.007;
    probe.submit_count = 1;
    probe.barrier_count = 2;
    probe.hints = {"sync_heavy", "compute_pressure_candidate"};
    artifact.probes.push_back(probe);

    const auto json = calibration_artifact_to_json(artifact);
    const auto parsed = parse_calibration_artifact(json);

    EXPECT_EQ(parsed.schema_version, 1u);
    EXPECT_EQ(parsed.microbench_schema_version, 2u);
    EXPECT_EQ(parsed.device_key, artifact.device_key);
    EXPECT_EQ(parsed.backend, artifact.backend);
    EXPECT_EQ(parsed.device_name, artifact.device_name);
    EXPECT_EQ(parsed.triage_hints, artifact.triage_hints);
    ASSERT_EQ(parsed.probes.size(), 1u);
    EXPECT_EQ(parsed.probes[0].name, "MB3");
    EXPECT_EQ(parsed.probes[0].submit_count, 1u);
    EXPECT_EQ(parsed.probes[0].barrier_count, 2u);
    EXPECT_EQ(parsed.probes[0].hints.size(), 2u);
}

TEST(OvGfxMicrobenchCalibrationTest, ArtifactCanBeWrittenAndLoadedFromFile) {
    CalibrationArtifact artifact;
    artifact.device_key = "apple:metal_default:metal";
    artifact.backend = "metal";
    artifact.device_name = "Apple M1 Max";
    artifact.fixed_overhead_us = 33.416;

    const auto path = std::filesystem::temp_directory_path() / "ov_gfx_microbench_calibration_test.json";
    std::error_code ec;
    std::filesystem::remove(path, ec);

    ASSERT_TRUE(write_calibration_artifact_file(artifact, path.string()));

    CalibrationArtifact loaded;
    ASSERT_TRUE(read_calibration_artifact_file(path.string(), loaded));
    EXPECT_EQ(loaded.device_key, artifact.device_key);
    EXPECT_EQ(loaded.backend, artifact.backend);
    EXPECT_EQ(loaded.device_name, artifact.device_name);
    EXPECT_DOUBLE_EQ(loaded.fixed_overhead_us, artifact.fixed_overhead_us);

    std::filesystem::remove(path, ec);
}

}  // namespace
}  // namespace microbench
}  // namespace gfx_plugin
}  // namespace ov
