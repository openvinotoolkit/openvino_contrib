// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <gtest/gtest.h>

#include <cstdlib>
#include <filesystem>
#include <fstream>

#include "plugin/gfx_profiling_utils.hpp"
#include "runtime/gfx_compile_profiling.hpp"
#include "runtime/gfx_profiling_report.hpp"

namespace ov {
namespace gfx_plugin {
namespace {

TEST(GfxProfilingReportTest, TraceAggregatesCountersSegmentsAndTransfers) {
    GfxProfilingTrace trace;
    trace.reset(ProfilingLevel::Detailed);
    trace.set_backend("vulkan");
    trace.set_counter_capability(true, true);
    trace.set_total_gpu_us(42);
    trace.set_total_cpu_us(24);
    trace.set_total_wall_us(64);
    trace.increment_counter("submit_count");
    trace.increment_counter("submit_count", 2);
    trace.set_counter("barrier_count", 5);
    trace.set_counter("pipeline_creation_count", 1);
    trace.set_counter("descriptor_update_count", 2);
    trace.set_counter("descriptor_write_count", 8);
    trace.add_segment("submit", "window#0", 7, 0, 0, 256, 128, 128, 256);
    trace.add_transfer("input_h2d", 256, true, 3);
    trace.add_transfer("output_d2h", 512, false, 4);
    trace.add_allocation("scratch", 1024, false, 2);

    const auto& report = trace.report();
    ASSERT_EQ(report.level, ProfilingLevel::Detailed);
    EXPECT_EQ(report.schema_version, 2u);
    EXPECT_EQ(report.backend, "vulkan");
    EXPECT_TRUE(report.counters_supported);
    EXPECT_TRUE(report.counters_used);
    EXPECT_EQ(report.total_gpu_us, 42u);
    EXPECT_EQ(report.total_cpu_us, 24u);
    EXPECT_EQ(report.total_wall_us, 64u);
    EXPECT_EQ(report.total_h2d_bytes, 256u);
    EXPECT_EQ(report.total_d2h_bytes, 512u);
    ASSERT_EQ(report.segments.size(), 1u);
    EXPECT_EQ(report.segments[0].phase, "submit");
    EXPECT_EQ(report.segments[0].name, "window#0");
    EXPECT_EQ(report.segments[0].bytes_out, 128u);
    ASSERT_EQ(report.counters.size(), 5u);
    EXPECT_EQ(report.counters[0].name, "submit_count");
    EXPECT_EQ(report.counters[0].value, 3u);
    EXPECT_EQ(report.counters[1].name, "barrier_count");
    EXPECT_EQ(report.counters[1].value, 5u);
    const auto json = trace.to_json();
    EXPECT_NE(json.find("\"schema_version\":2"), std::string::npos);
    EXPECT_NE(json.find("\"backend\":\"vulkan\""), std::string::npos);
    EXPECT_NE(json.find("\"summary\""), std::string::npos);
    EXPECT_NE(json.find("\"phase_totals\""), std::string::npos);
    EXPECT_NE(json.find("\"roofline\""), std::string::npos);
    EXPECT_NE(json.find("\"arithmetic_intensity\""), std::string::npos);
    EXPECT_NE(json.find("\"dominant_regime\":\"memory\""), std::string::npos);
    EXPECT_NE(json.find("\"transfer_totals\""), std::string::npos);
    EXPECT_NE(json.find("\"submit_count\""), std::string::npos);
    EXPECT_NE(json.find("\"pipeline_creation_count\":1"), std::string::npos);
    EXPECT_NE(json.find("\"descriptor_update_count\":2"), std::string::npos);
    EXPECT_NE(json.find("\"category\":\"pipeline_creation\""), std::string::npos);
    EXPECT_NE(json.find("\"category\":\"descriptor_update\""), std::string::npos);
    EXPECT_NE(json.find("\"category\":\"roofline\""), std::string::npos);
    EXPECT_NE(json.find("\"input_h2d\""), std::string::npos);
    EXPECT_NE(json.find("\"window#0\""), std::string::npos);
}

TEST(GfxProfilingReportTest, RootJsonCanEmbedCompileAndExtendedReports) {
    const std::string compile_json = R"({"schema_version":2,"backend":"vulkan","level":"detailed","segments":[{"phase":"compile","name":"build_op_pipeline"}]})";
    const std::string extended_json = R"({"schema_version":2,"backend":"vulkan","level":"detailed","segments":[{"phase":"infer","name":"submit"}]})";
    const auto json = build_profiling_report_json("vulkan", ProfilingLevel::Detailed, {}, extended_json, compile_json);
    EXPECT_NE(json.find("\"compile\":{\"schema_version\":2"), std::string::npos);
    EXPECT_NE(json.find("\"extended\":{\"schema_version\":2"), std::string::npos);
    EXPECT_NE(json.find("\"backend\":\"vulkan\""), std::string::npos);
}

TEST(GfxProfilingReportTest, PerfettoTraceEventsExportWhenRequested) {
    const char* old_value = std::getenv("OV_GFX_PROFILE_TRACE");
    const std::string saved = old_value ? old_value : "";
    setenv("OV_GFX_PROFILE_TRACE", "perfetto", 1);

    GfxProfilingTrace trace;
    trace.reset(ProfilingLevel::Detailed);
    trace.set_backend("vulkan");
    trace.increment_counter("submit_count", 3);
    trace.add_segment("submit", "window#1", 25, 0, 0, 0, 0, 0, 0, 2, 7, 9);
    trace.add_transfer("input_h2d", 128, true, 6);
    trace.add_allocation("scratch", 512, false, 4);
    trace.set_total_wall_us(64);
    const auto json = trace.to_json();

    if (old_value) {
        setenv("OV_GFX_PROFILE_TRACE", saved.c_str(), 1);
    } else {
        unsetenv("OV_GFX_PROFILE_TRACE");
    }

    EXPECT_NE(json.find("\"trace_sink\":\"perfetto\""), std::string::npos);
    EXPECT_NE(json.find("\"traceEvents\":["), std::string::npos);
    EXPECT_NE(json.find("\"ph\":\"X\""), std::string::npos);
    EXPECT_NE(json.find("\"cat\":\"submit\""), std::string::npos);
    EXPECT_NE(json.find("\"name\":\"window#1\""), std::string::npos);
    EXPECT_NE(json.find("\"cat\":\"transfer\""), std::string::npos);
    EXPECT_NE(json.find("\"name\":\"input_h2d\""), std::string::npos);
    EXPECT_NE(json.find("\"cat\":\"allocation\""), std::string::npos);
    EXPECT_NE(json.find("\"name\":\"scratch\""), std::string::npos);
    EXPECT_NE(json.find("\"ph\":\"C\""), std::string::npos);
    EXPECT_NE(json.find("\"name\":\"counters\""), std::string::npos);
}

TEST(GfxProfilingReportTest, RootProfilingReportCanExportMergedTraceFile) {
    const auto trace_path = std::filesystem::temp_directory_path() / "ov_gfx_profile_trace_test.json";
    std::error_code ec;
    std::filesystem::remove(trace_path, ec);

    const char* old_path = std::getenv("OV_GFX_PROFILE_TRACE_FILE");
    const std::string saved_path = old_path ? old_path : "";
    setenv("OV_GFX_PROFILE_TRACE_FILE", trace_path.string().c_str(), 1);

    const std::string compile_json =
        R"({"schema_version":2,"backend":"vulkan","level":"detailed","trace_sink":"perfetto","traceEvents":[{"name":"compile.stage","cat":"compile","ph":"X","ts":1,"dur":2,"pid":1,"tid":11,"args":{"backend":"vulkan"}}]})";
    const std::string extended_json =
        R"({"schema_version":2,"backend":"vulkan","level":"detailed","trace_sink":"perfetto","traceEvents":[{"name":"infer.submit","cat":"submit","ph":"X","ts":3,"dur":4,"pid":1,"tid":22,"args":{"backend":"vulkan"}}]})";
    const auto root_json = build_profiling_report_json("vulkan", ProfilingLevel::Detailed, {}, extended_json, compile_json);

    if (old_path) {
        setenv("OV_GFX_PROFILE_TRACE_FILE", saved_path.c_str(), 1);
    } else {
        unsetenv("OV_GFX_PROFILE_TRACE_FILE");
    }

    ASSERT_NE(root_json.find("\"compile\":{"), std::string::npos);
    ASSERT_NE(root_json.find("\"extended\":{"), std::string::npos);
    ASSERT_TRUE(std::filesystem::exists(trace_path));

    std::ifstream trace_file(trace_path);
    ASSERT_TRUE(trace_file.is_open());
    const std::string trace_json((std::istreambuf_iterator<char>(trace_file)), std::istreambuf_iterator<char>());
    EXPECT_NE(trace_json.find("\"traceEvents\":["), std::string::npos);
    EXPECT_NE(trace_json.find("\"name\":\"compile.stage\""), std::string::npos);
    EXPECT_NE(trace_json.find("\"name\":\"infer.submit\""), std::string::npos);
    EXPECT_NE(trace_json.find("\"ts\":6"), std::string::npos);
    EXPECT_NE(trace_json.find("\"displayTimeUnit\":\"ms\""), std::string::npos);

    std::filesystem::remove(trace_path, ec);
}

TEST(GfxProfilingReportTest, CompileProfilingContextPrefixesNestedCompileEvents) {
    GfxProfilingTrace trace;
    trace.reset(ProfilingLevel::Detailed);
    trace.set_backend("metal");

    {
        ScopedCompileProfilingContext scope(&trace, "stage.conv0");
        increment_compile_counter("kernel_cache_miss_count");
        add_compile_segment("metal_resolve_msl", 7);
        add_compile_segment("metal_pipeline_state_create", 11);
    }

    ASSERT_EQ(trace.report().counters.size(), 1u);
    EXPECT_EQ(trace.report().counters[0].name, "kernel_cache_miss_count");
    EXPECT_EQ(trace.report().counters[0].value, 1u);
    ASSERT_EQ(trace.report().segments.size(), 2u);
    EXPECT_EQ(trace.report().segments[0].phase, "compile");
    EXPECT_EQ(trace.report().segments[0].name, "stage.conv0::metal_resolve_msl");
    EXPECT_EQ(trace.report().segments[1].name, "stage.conv0::metal_pipeline_state_create");
}

}  // namespace
}  // namespace gfx_plugin
}  // namespace ov
