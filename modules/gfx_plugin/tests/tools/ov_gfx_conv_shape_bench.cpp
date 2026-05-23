#include <openvino/op/add.hpp>
#include <openvino/op/concat.hpp>
#include <openvino/op/constant.hpp>
#include <openvino/op/convolution.hpp>
#include <openvino/op/matmul.hpp>
#include <openvino/op/parameter.hpp>
#include <openvino/op/reshape.hpp>
#include <openvino/op/result.hpp>
#include <openvino/op/transpose.hpp>
#include <openvino/openvino.hpp>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <exception>
#include <iomanip>
#include <iostream>
#include <numeric>
#include <optional>
#include <string>
#include <string_view>
#include <vector>

#include "../gfx_accuracy_tolerance.hpp"
#include "common_test_utils/file_utils.hpp"
#include "common_test_utils/ov_plugin_cache.hpp"
#include "openvino/util/file_util.hpp"

namespace {

const char *resolve_gfx_plugin_path() {
  if (const char *env_path = std::getenv("GFX_PLUGIN_PATH")) {
    if (*env_path) {
      return env_path;
    }
  }
#ifdef GFX_PLUGIN_PATH
  return GFX_PLUGIN_PATH;
#else
  return nullptr;
#endif
}

void register_gfx_plugin(ov::Core &core) {
  if (const char *path = resolve_gfx_plugin_path()) {
    try {
      core.register_plugin(path, "GFX");
    } catch (...) {
    }
  }
}

void register_reference_plugin(ov::Core &core,
                               const std::string &reference_device_name,
                               const std::string &reference_plugin_path) {
  if (!reference_plugin_path.empty()) {
    try {
      core.register_plugin(reference_plugin_path, reference_device_name);
    } catch (...) {
    }
  }
  if (reference_device_name == "TEMPLATE") {
    try {
      const auto plugin_path = ov::util::make_plugin_library_name(
          ov::test::utils::getExecutableDirectory(),
          std::string(ov::test::utils::TEMPLATE_LIB) + OV_BUILD_POSTFIX);
      core.register_plugin(plugin_path, "TEMPLATE");
    } catch (...) {
    }
  }
}

struct ShapeCase {
  std::string name;
  ov::Shape input;
  ov::Shape weights;
  ov::Strides strides;
  ov::CoordinateDiff pads_begin;
  ov::CoordinateDiff pads_end;
  ov::Strides dilations;
};

struct Options {
  size_t warmup = 1;
  size_t iterations = 3;
  std::vector<std::string> devices{"GFX"};
  std::vector<std::string> case_filters;
  bool list_cases = false;
  bool compare_template = false;
  bool compare_pointwise_matmul = false;
  bool compare_input_split_conv = false;
  bool compare_output_split_conv = false;
  bool compare_stride2_sublattice_conv = false;
  bool compare_stride2_row_split_conv = false;
  size_t input_split_parts = 2;
  size_t output_split_parts = 2;
  std::string reference_device = "TEMPLATE";
  std::string reference_plugin_path;
  std::optional<double> abs_threshold;
  std::optional<double> rel_threshold;
  bool dump_gfx_profile = false;
  std::string gfx_profiling_level;
};

Options parse_options(int argc, char **argv) {
  Options options;
  for (int i = 1; i < argc; ++i) {
    const std::string arg = argv[i];
    if (arg == "--warmup" && i + 1 < argc) {
      options.warmup = static_cast<size_t>(std::stoul(argv[++i]));
    } else if (arg == "--iterations" && i + 1 < argc) {
      options.iterations = static_cast<size_t>(std::stoul(argv[++i]));
    } else if (arg == "--device" && i + 1 < argc) {
      options.devices = {argv[++i]};
    } else if (arg == "--case" && i + 1 < argc) {
      options.case_filters.push_back(argv[++i]);
    } else if (arg == "--list-cases") {
      options.list_cases = true;
    } else if (arg == "--compare-template") {
      options.compare_template = true;
    } else if (arg == "--compare-pointwise-matmul") {
      options.compare_pointwise_matmul = true;
    } else if (arg == "--compare-input-split-conv") {
      options.compare_input_split_conv = true;
    } else if (arg == "--input-split-parts" && i + 1 < argc) {
      options.input_split_parts = static_cast<size_t>(std::stoul(argv[++i]));
    } else if (arg == "--compare-output-split-conv") {
      options.compare_output_split_conv = true;
    } else if (arg == "--output-split-parts" && i + 1 < argc) {
      options.output_split_parts = static_cast<size_t>(std::stoul(argv[++i]));
    } else if (arg == "--compare-stride2-sublattice-conv") {
      options.compare_stride2_sublattice_conv = true;
    } else if (arg == "--compare-stride2-row-split-conv") {
      options.compare_stride2_row_split_conv = true;
    } else if (arg == "--reference-device" && i + 1 < argc) {
      options.reference_device = argv[++i];
    } else if (arg == "--reference-plugin" && i + 1 < argc) {
      options.reference_plugin_path = argv[++i];
    } else if (arg == "--abs-threshold" && i + 1 < argc) {
      options.abs_threshold = std::stod(argv[++i]);
    } else if (arg == "--rel-threshold" && i + 1 < argc) {
      options.rel_threshold = std::stod(argv[++i]);
    } else if (arg == "--dump-gfx-profile") {
      options.dump_gfx_profile = true;
    } else if (arg == "--gfx-profiling-level" && i + 1 < argc) {
      options.gfx_profiling_level = argv[++i];
    } else if (arg == "--help") {
      std::cout
          << "Usage: ov_gfx_conv_shape_bench [--warmup N] [--iterations N]"
             " [--device GFX|CPU] [--case SUBSTRING] [--list-cases]"
             " [--dump-gfx-profile] [--gfx-profiling-level LEVEL]\n"
             "       ov_gfx_conv_shape_bench --compare-template"
             " [--reference-device TEMPLATE] [--reference-plugin PATH]"
             " [--abs-threshold V] [--rel-threshold V] [--case SUBSTRING]\n"
             "       ov_gfx_conv_shape_bench --compare-pointwise-matmul"
             " [--case SUBSTRING] [--warmup N] [--iterations N]\n"
             "       ov_gfx_conv_shape_bench --compare-input-split-conv"
             " [--case SUBSTRING] [--input-split-parts N] [--warmup N]"
             " [--iterations N]\n"
             "       ov_gfx_conv_shape_bench --compare-output-split-conv"
             " [--case SUBSTRING] [--output-split-parts N] [--warmup N]"
             " [--iterations N]\n"
             "       ov_gfx_conv_shape_bench --compare-stride2-sublattice-conv"
             " [--case SUBSTRING] [--warmup N] [--iterations N]\n"
             "       ov_gfx_conv_shape_bench --compare-stride2-row-split-conv"
             " [--case SUBSTRING] [--warmup N] [--iterations N]\n"
             "Default benchmark device is GFX. CPU is an explicit performance"
             " orienter only, not a plugin inference fallback.\n";
      std::exit(0);
    } else {
      throw std::runtime_error("unknown argument: " + arg);
    }
  }
  if (options.iterations == 0) {
    throw std::runtime_error("--iterations must be > 0");
  }
  if (options.reference_device == "CPU") {
    throw std::runtime_error(
        "CPU reference device is not supported; use TEMPLATE");
  }
  if (options.input_split_parts < 2) {
    throw std::runtime_error("--input-split-parts must be >= 2");
  }
  if (options.output_split_parts < 2) {
    throw std::runtime_error("--output-split-parts must be >= 2");
  }
  return options;
}

bool case_matches_filters(const ShapeCase &c,
                          const std::vector<std::string> &filters) {
  if (filters.empty()) {
    return true;
  }
  return std::any_of(
      filters.begin(), filters.end(), [&](const std::string &filter) {
        return !filter.empty() && c.name.find(filter) != std::string::npos;
      });
}

std::vector<float> make_data(size_t count, size_t salt, float scale) {
  std::vector<float> data(count);
  for (size_t i = 0; i < count; ++i) {
    const int value = static_cast<int>((i * 131u + salt * 17u) % 257u) - 128;
    data[i] = static_cast<float>(value) * scale;
  }
  return data;
}

std::shared_ptr<ov::Model> make_conv_model(const ShapeCase &c) {
  auto input =
      std::make_shared<ov::op::v0::Parameter>(ov::element::f32, c.input);
  const auto weight_count = ov::shape_size(c.weights);
  auto weights = ov::op::v0::Constant::create(
      ov::element::f32, c.weights,
      make_data(weight_count, c.input[1] + c.weights[0], 0.01f));
  auto conv = std::make_shared<ov::op::v1::Convolution>(
      input, weights, c.strides, c.pads_begin, c.pads_end, c.dilations);
  const ov::Shape bias_shape{1, c.weights[0], 1, 1};
  auto bias = ov::op::v0::Constant::create(
      ov::element::f32, bias_shape,
      make_data(c.weights[0], c.weights[0], 0.001f));
  auto add = std::make_shared<ov::op::v1::Add>(
      conv, bias, ov::op::AutoBroadcastType::NUMPY);
  auto result = std::make_shared<ov::op::v0::Result>(add);
  return std::make_shared<ov::Model>(ov::ResultVector{result},
                                     ov::ParameterVector{input}, c.name);
}

bool is_pointwise_matmul_candidate(const ShapeCase &c) {
  return c.input.size() == 4 && c.weights.size() == 4 && c.weights[2] == 1 &&
         c.weights[3] == 1 && c.strides == ov::Strides{1, 1} &&
         c.pads_begin == ov::CoordinateDiff{0, 0} &&
         c.pads_end == ov::CoordinateDiff{0, 0} &&
         c.dilations == ov::Strides{1, 1};
}

std::shared_ptr<ov::Model> make_pointwise_matmul_model(const ShapeCase &c) {
  if (!is_pointwise_matmul_candidate(c)) {
    throw std::runtime_error(
        "pointwise MatMul study expects stride-1 pad-0 dilation-1 1x1 Conv");
  }

  const size_t n = c.input[0];
  const size_t c_in = c.input[1];
  const size_t h = c.input[2];
  const size_t w = c.input[3];
  const size_t c_out = c.weights[0];
  const auto conv_weights =
      make_data(ov::shape_size(c.weights), c.input[1] + c.weights[0], 0.01f);
  std::vector<float> rhs(c_in * c_out);
  for (size_t co = 0; co < c_out; ++co) {
    for (size_t ci = 0; ci < c_in; ++ci) {
      rhs[ci * c_out + co] = conv_weights[co * c_in + ci];
    }
  }

  auto input =
      std::make_shared<ov::op::v0::Parameter>(ov::element::f32, c.input);
  auto to_nhwc_order =
      ov::op::v0::Constant::create(ov::element::i64, {4}, {0, 2, 3, 1});
  auto nhwc = std::make_shared<ov::op::v1::Transpose>(input, to_nhwc_order);
  auto flat_shape = ov::op::v0::Constant::create(
      ov::element::i64, {2},
      {static_cast<int64_t>(n * h * w), static_cast<int64_t>(c_in)});
  auto flat = std::make_shared<ov::op::v1::Reshape>(nhwc, flat_shape, false);
  auto weights =
      ov::op::v0::Constant::create(ov::element::f32, {c_in, c_out}, rhs);
  auto matmul =
      std::make_shared<ov::op::v0::MatMul>(flat, weights, false, false);
  auto nhwc_out_shape = ov::op::v0::Constant::create(
      ov::element::i64, {4},
      {static_cast<int64_t>(n), static_cast<int64_t>(h),
       static_cast<int64_t>(w), static_cast<int64_t>(c_out)});
  auto nhwc_out =
      std::make_shared<ov::op::v1::Reshape>(matmul, nhwc_out_shape, false);
  const ov::Shape bias_shape{1, 1, 1, c_out};
  auto bias = ov::op::v0::Constant::create(ov::element::f32, bias_shape,
                                           make_data(c_out, c_out, 0.001f));
  auto add = std::make_shared<ov::op::v1::Add>(
      nhwc_out, bias, ov::op::AutoBroadcastType::NUMPY);
  auto to_nchw_order =
      ov::op::v0::Constant::create(ov::element::i64, {4}, {0, 3, 1, 2});
  auto nchw_out = std::make_shared<ov::op::v1::Transpose>(add, to_nchw_order);
  auto result = std::make_shared<ov::op::v0::Result>(nchw_out);
  return std::make_shared<ov::Model>(ov::ResultVector{result},
                                     ov::ParameterVector{input},
                                     c.name + "_pointwise_matmul_candidate");
}

bool is_input_split_conv_candidate(const ShapeCase &c, size_t split_parts) {
  return c.input.size() == 4 && c.weights.size() == 4 && split_parts >= 2 &&
         c.input[1] == c.weights[1] && c.weights[1] % split_parts == 0;
}

std::shared_ptr<ov::Model> make_input_split_conv_model(const ShapeCase &c,
                                                       size_t split_parts) {
  if (!is_input_split_conv_candidate(c, split_parts)) {
    throw std::runtime_error(
        "input-split Conv study expects divisible static Conv channels");
  }

  const size_t c_out = c.weights[0];
  const size_t c_in = c.weights[1];
  const size_t kh = c.weights[2];
  const size_t kw = c.weights[3];
  const size_t part_ic = c_in / split_parts;
  const auto conv_weights =
      make_data(ov::shape_size(c.weights), c.input[1] + c.weights[0], 0.01f);

  ov::ParameterVector inputs;
  std::shared_ptr<ov::Node> sum;
  for (size_t part = 0; part < split_parts; ++part) {
    ov::Shape part_input_shape = c.input;
    part_input_shape[1] = part_ic;
    auto input = std::make_shared<ov::op::v0::Parameter>(ov::element::f32,
                                                         part_input_shape);
    inputs.push_back(input);

    std::vector<float> part_weights(c_out * part_ic * kh * kw);
    for (size_t co = 0; co < c_out; ++co) {
      for (size_t ci = 0; ci < part_ic; ++ci) {
        const size_t full_ci = part * part_ic + ci;
        for (size_t y = 0; y < kh; ++y) {
          for (size_t x = 0; x < kw; ++x) {
            const size_t src = ((co * c_in + full_ci) * kh + y) * kw + x;
            const size_t dst = ((co * part_ic + ci) * kh + y) * kw + x;
            part_weights[dst] = conv_weights[src];
          }
        }
      }
    }
    auto weights = ov::op::v0::Constant::create(
        ov::element::f32, {c_out, part_ic, kh, kw}, part_weights);
    auto conv = std::make_shared<ov::op::v1::Convolution>(
        input, weights, c.strides, c.pads_begin, c.pads_end, c.dilations);
    if (!sum) {
      sum = conv;
    } else {
      sum = std::make_shared<ov::op::v1::Add>(sum, conv,
                                              ov::op::AutoBroadcastType::NUMPY);
    }
  }

  const ov::Shape bias_shape{1, c_out, 1, 1};
  auto bias = ov::op::v0::Constant::create(ov::element::f32, bias_shape,
                                           make_data(c_out, c_out, 0.001f));
  auto add = std::make_shared<ov::op::v1::Add>(
      sum, bias, ov::op::AutoBroadcastType::NUMPY);
  auto result = std::make_shared<ov::op::v0::Result>(add);
  return std::make_shared<ov::Model>(ov::ResultVector{result}, inputs,
                                     c.name + "_input_split_conv_candidate");
}

bool is_output_split_conv_candidate(const ShapeCase &c, size_t split_parts) {
  return c.input.size() == 4 && c.weights.size() == 4 && split_parts >= 2 &&
         c.weights[0] % split_parts == 0;
}

std::shared_ptr<ov::Model> make_output_split_conv_model(const ShapeCase &c,
                                                        size_t split_parts) {
  if (!is_output_split_conv_candidate(c, split_parts)) {
    throw std::runtime_error(
        "output-split Conv study expects divisible static output channels");
  }

  const size_t c_out = c.weights[0];
  const size_t c_in = c.weights[1];
  const size_t kh = c.weights[2];
  const size_t kw = c.weights[3];
  const size_t part_oc = c_out / split_parts;
  const auto conv_weights =
      make_data(ov::shape_size(c.weights), c.input[1] + c.weights[0], 0.01f);
  const auto full_bias = make_data(c_out, c_out, 0.001f);

  auto input =
      std::make_shared<ov::op::v0::Parameter>(ov::element::f32, c.input);
  ov::NodeVector parts;
  parts.reserve(split_parts);
  for (size_t part = 0; part < split_parts; ++part) {
    std::vector<float> part_weights(part_oc * c_in * kh * kw);
    for (size_t co = 0; co < part_oc; ++co) {
      const size_t full_co = part * part_oc + co;
      for (size_t ci = 0; ci < c_in; ++ci) {
        for (size_t y = 0; y < kh; ++y) {
          for (size_t x = 0; x < kw; ++x) {
            const size_t src = ((full_co * c_in + ci) * kh + y) * kw + x;
            const size_t dst = ((co * c_in + ci) * kh + y) * kw + x;
            part_weights[dst] = conv_weights[src];
          }
        }
      }
    }
    auto weights = ov::op::v0::Constant::create(
        ov::element::f32, {part_oc, c_in, kh, kw}, part_weights);
    auto conv = std::make_shared<ov::op::v1::Convolution>(
        input, weights, c.strides, c.pads_begin, c.pads_end, c.dilations);

    std::vector<float> part_bias(part_oc);
    std::copy_n(full_bias.begin() + static_cast<std::ptrdiff_t>(part * part_oc),
                part_oc, part_bias.begin());
    auto bias = ov::op::v0::Constant::create(ov::element::f32,
                                             {1, part_oc, 1, 1}, part_bias);
    parts.push_back(std::make_shared<ov::op::v1::Add>(
        conv, bias, ov::op::AutoBroadcastType::NUMPY));
  }

  auto concat = std::make_shared<ov::op::v0::Concat>(parts, 1);
  auto result = std::make_shared<ov::op::v0::Result>(concat);
  return std::make_shared<ov::Model>(ov::ResultVector{result},
                                     ov::ParameterVector{input},
                                     c.name + "_output_split_conv_candidate");
}

bool is_stride2_sublattice_conv_candidate(const ShapeCase &c) {
  return c.input.size() == 4 && c.weights.size() == 4 && c.weights[2] == 3 &&
         c.weights[3] == 3 && c.strides == ov::Strides{2, 2} &&
         c.dilations == ov::Strides{1, 1};
}

ShapeCase make_stride2_valid_core_case(const ShapeCase &c) {
  ShapeCase valid = c;
  valid.name += "_valid_core";
  valid.pads_begin = ov::CoordinateDiff{0, 0};
  valid.pads_end = ov::CoordinateDiff{0, 0};
  return valid;
}

std::shared_ptr<ov::Model>
make_stride2_sublattice_conv_model(const ShapeCase &c) {
  if (!is_stride2_sublattice_conv_candidate(c)) {
    throw std::runtime_error(
        "stride2 sublattice Conv study expects 3x3 stride-2 dilation-1 Conv");
  }

  const size_t n = c.input[0];
  const size_t c_in = c.input[1];
  const size_t h_out =
      (c.input[2] + static_cast<size_t>(c.pads_begin[0] + c.pads_end[0]) -
       c.weights[2]) /
          c.strides[0] +
      1;
  const size_t w_out =
      (c.input[3] + static_cast<size_t>(c.pads_begin[1] + c.pads_end[1]) -
       c.weights[3]) /
          c.strides[1] +
      1;
  const size_t c_out = c.weights[0];
  const auto conv_weights =
      make_data(ov::shape_size(c.weights), c.input[1] + c.weights[0], 0.01f);

  ov::ParameterVector inputs;
  inputs.reserve(9);

  std::shared_ptr<ov::Node> sum;
  for (size_t kh = 0; kh < 3; ++kh) {
    for (size_t kw = 0; kw < 3; ++kw) {
      auto sub = std::make_shared<ov::op::v0::Parameter>(
          ov::element::f32, ov::Shape{n, c_in, h_out, w_out});
      inputs.push_back(sub);

      std::vector<float> tap_weights(c_out * c_in);
      for (size_t co = 0; co < c_out; ++co) {
        for (size_t ci = 0; ci < c_in; ++ci) {
          const size_t src = ((co * c_in + ci) * 3 + kh) * 3 + kw;
          tap_weights[co * c_in + ci] = conv_weights[src];
        }
      }
      auto weights = ov::op::v0::Constant::create(
          ov::element::f32, {c_out, c_in, 1, 1}, tap_weights);
      auto conv = std::make_shared<ov::op::v1::Convolution>(
          sub, weights, ov::Strides{1, 1}, ov::CoordinateDiff{0, 0},
          ov::CoordinateDiff{0, 0}, ov::Strides{1, 1});
      if (!sum) {
        sum = conv;
      } else {
        sum = std::make_shared<ov::op::v1::Add>(
            sum, conv, ov::op::AutoBroadcastType::NUMPY);
      }
    }
  }

  auto bias = ov::op::v0::Constant::create(ov::element::f32, {1, c_out, 1, 1},
                                           make_data(c_out, c_out, 0.001f));
  auto add = std::make_shared<ov::op::v1::Add>(
      sum, bias, ov::op::AutoBroadcastType::NUMPY);
  auto result = std::make_shared<ov::op::v0::Result>(add);
  return std::make_shared<ov::Model>(
      ov::ResultVector{result}, inputs,
      c.name + "_stride2_sublattice_conv_candidate");
}

std::shared_ptr<ov::Model>
make_stride2_row_split_conv_model(const ShapeCase &c) {
  if (!is_stride2_sublattice_conv_candidate(c)) {
    throw std::runtime_error(
        "stride2 row-split Conv study expects 3x3 stride-2 dilation-1 Conv");
  }

  const size_t n = c.input[0];
  const size_t c_in = c.input[1];
  const size_t h_out =
      (c.input[2] + static_cast<size_t>(c.pads_begin[0] + c.pads_end[0]) -
       c.weights[2]) /
          c.strides[0] +
      1;
  const size_t w_out =
      (c.input[3] + static_cast<size_t>(c.pads_begin[1] + c.pads_end[1]) -
       c.weights[3]) /
          c.strides[1] +
      1;
  const size_t c_out = c.weights[0];
  const auto conv_weights =
      make_data(ov::shape_size(c.weights), c.input[1] + c.weights[0], 0.01f);

  ov::ParameterVector inputs;
  inputs.reserve(3);

  std::shared_ptr<ov::Node> sum;
  for (size_t kh = 0; kh < 3; ++kh) {
    auto row_input = std::make_shared<ov::op::v0::Parameter>(
        ov::element::f32, ov::Shape{n, c_in, h_out, c.input[3]});
    inputs.push_back(row_input);

    std::vector<float> row_weights(c_out * c_in * 3);
    for (size_t co = 0; co < c_out; ++co) {
      for (size_t ci = 0; ci < c_in; ++ci) {
        for (size_t kw = 0; kw < 3; ++kw) {
          const size_t src = ((co * c_in + ci) * 3 + kh) * 3 + kw;
          const size_t dst = (co * c_in + ci) * 3 + kw;
          row_weights[dst] = conv_weights[src];
        }
      }
    }
    auto weights = ov::op::v0::Constant::create(
        ov::element::f32, {c_out, c_in, 1, 3}, row_weights);
    auto conv = std::make_shared<ov::op::v1::Convolution>(
        row_input, weights, ov::Strides{1, 2}, ov::CoordinateDiff{0, 0},
        ov::CoordinateDiff{0, 0}, ov::Strides{1, 1});
    if (!sum) {
      sum = conv;
    } else {
      sum = std::make_shared<ov::op::v1::Add>(
          sum, conv, ov::op::AutoBroadcastType::NUMPY);
    }
  }

  auto bias = ov::op::v0::Constant::create(ov::element::f32, {1, c_out, 1, 1},
                                           make_data(c_out, c_out, 0.001f));
  auto add = std::make_shared<ov::op::v1::Add>(
      sum, bias, ov::op::AutoBroadcastType::NUMPY);
  auto result = std::make_shared<ov::op::v0::Result>(add);
  (void)w_out;
  return std::make_shared<ov::Model>(
      ov::ResultVector{result}, inputs,
      c.name + "_stride2_row_split_conv_candidate");
}

ov::AnyMap make_config(const Options &options, bool for_gfx) {
  ov::AnyMap config;
  config[ov::hint::inference_precision.name()] = ov::element::f16;
  if (for_gfx &&
      (options.dump_gfx_profile || !options.gfx_profiling_level.empty())) {
    config["GFX_PROFILING_LEVEL"] =
        options.gfx_profiling_level.empty() ? "2" : options.gfx_profiling_level;
    config[ov::enable_profiling.name()] = true;
    config["PERF_COUNT"] = true;
  }
  return config;
}

float input_value(size_t linear_index) {
  const int value = static_cast<int>((linear_index * 97u + 13u) % 251u) - 125;
  return static_cast<float>(value) / 64.0f;
}

void fill_input(ov::Tensor &tensor) {
  auto *data = tensor.data<float>();
  const size_t count = tensor.get_size();
  for (size_t i = 0; i < count; ++i) {
    data[i] = input_value(i);
  }
}

void fill_input_channel_chunk(ov::Tensor &tensor, const ov::Shape &full_shape,
                              size_t channel_offset) {
  auto *data = tensor.data<float>();
  const auto chunk_shape = tensor.get_shape();
  const size_t n_count = chunk_shape[0];
  const size_t chunk_channels = chunk_shape[1];
  const size_t height = chunk_shape[2];
  const size_t width = chunk_shape[3];
  const size_t full_channels = full_shape[1];
  for (size_t n = 0; n < n_count; ++n) {
    for (size_t c = 0; c < chunk_channels; ++c) {
      for (size_t y = 0; y < height; ++y) {
        for (size_t x = 0; x < width; ++x) {
          const size_t chunk_index =
              ((n * chunk_channels + c) * height + y) * width + x;
          const size_t full_index =
              ((n * full_channels + channel_offset + c) * height + y) * width +
              x;
          data[chunk_index] = input_value(full_index);
        }
      }
    }
  }
}

void fill_input_stride2_sublattice(ov::Tensor &tensor,
                                   const ov::Shape &full_shape, size_t kh,
                                   size_t kw) {
  auto *data = tensor.data<float>();
  const auto sub_shape = tensor.get_shape();
  const size_t n_count = sub_shape[0];
  const size_t channels = sub_shape[1];
  const size_t sub_height = sub_shape[2];
  const size_t sub_width = sub_shape[3];
  const size_t full_channels = full_shape[1];
  const size_t full_width = full_shape[3];
  for (size_t n = 0; n < n_count; ++n) {
    for (size_t c = 0; c < channels; ++c) {
      for (size_t y = 0; y < sub_height; ++y) {
        for (size_t x = 0; x < sub_width; ++x) {
          const size_t sub_index =
              ((n * channels + c) * sub_height + y) * sub_width + x;
          const size_t full_y = kh + 2 * y;
          const size_t full_x = kw + 2 * x;
          const size_t full_index =
              ((n * full_channels + c) * full_shape[2] + full_y) *
                  full_width +
              full_x;
          data[sub_index] = input_value(full_index);
        }
      }
    }
  }
}

void fill_input_stride2_row_split(ov::Tensor &tensor,
                                  const ov::Shape &full_shape, size_t kh) {
  auto *data = tensor.data<float>();
  const auto row_shape = tensor.get_shape();
  const size_t n_count = row_shape[0];
  const size_t channels = row_shape[1];
  const size_t row_height = row_shape[2];
  const size_t row_width = row_shape[3];
  const size_t full_channels = full_shape[1];
  const size_t full_width = full_shape[3];
  for (size_t n = 0; n < n_count; ++n) {
    for (size_t c = 0; c < channels; ++c) {
      for (size_t y = 0; y < row_height; ++y) {
        for (size_t x = 0; x < row_width; ++x) {
          const size_t row_index =
              ((n * channels + c) * row_height + y) * row_width + x;
          const size_t full_y = kh + 2 * y;
          const size_t full_index =
              ((n * full_channels + c) * full_shape[2] + full_y) *
                  full_width +
              x;
          data[row_index] = input_value(full_index);
        }
      }
    }
  }
}

double median(std::vector<double> values) {
  std::sort(values.begin(), values.end());
  return values[values.size() / 2];
}

struct RunStats {
  double compile_ms = 0.0;
  double median_infer_ms = 0.0;
  double min_infer_ms = 0.0;
  std::string gfx_profile_json;
};

struct GfxProfileDigest {
  bool available = false;
  uint64_t total_gpu_us = 0;
  uint64_t total_wall_us = 0;
  uint64_t submit_count = 0;
  uint64_t barrier_count = 0;
  uint64_t descriptor_update_count = 0;
  uint64_t pipeline_creation_count = 0;
  uint64_t conv_dispatch_tile_h = 0;
  uint64_t conv_dispatch_tile_w = 0;
  uint64_t conv_dispatch_threads_h = 0;
  uint64_t conv_dispatch_threads_w = 0;
  uint64_t conv_dispatch_channel_block = 0;
  uint64_t conv_output_reuse_lanes = 0;
  uint64_t conv_spatial_input_reuse_lanes = 0;
  uint64_t conv_spatial_input_reuse_saved_width_loads = 0;
  uint64_t conv_workgroup_reduction_lanes = 0;
  uint64_t conv_workgroup_output_lanes = 0;
  uint64_t conv_multi_kernel_coarse_output_tile_elements = 0;
  uint64_t conv_multi_kernel_workgroup_output_tile_deficit = 0;
  uint64_t conv_multi_kernel_workgroup_local_accumulator_bytes = 0;
  uint64_t runtime_dispatch_grid_x = 0;
  uint64_t runtime_dispatch_grid_y = 0;
  uint64_t runtime_dispatch_grid_z = 0;
};

uint64_t extract_profile_uint(std::string_view json, std::string_view field,
                              uint64_t fallback = 0) {
  const std::string needle = "\"" + std::string(field) + "\":";
  const size_t pos = json.find(needle);
  if (pos == std::string_view::npos) {
    return fallback;
  }
  size_t begin = pos + needle.size();
  while (begin < json.size() &&
         (json[begin] == ' ' || json[begin] == '\t' || json[begin] == '"')) {
    ++begin;
  }
  size_t end = begin;
  while (end < json.size() && json[end] >= '0' && json[end] <= '9') {
    ++end;
  }
  if (end == begin) {
    return fallback;
  }
  try {
    return std::stoull(std::string(json.substr(begin, end - begin)));
  } catch (...) {
    return fallback;
  }
}

std::string_view extract_profile_object(std::string_view json,
                                        std::string_view field) {
  const std::string needle = "\"" + std::string(field) + "\":{";
  const size_t pos = json.find(needle);
  if (pos == std::string_view::npos) {
    return {};
  }
  const size_t object_begin = pos + needle.size() - 1;
  size_t depth = 0;
  bool in_string = false;
  bool escaped = false;
  for (size_t i = object_begin; i < json.size(); ++i) {
    const char c = json[i];
    if (in_string) {
      if (escaped) {
        escaped = false;
      } else if (c == '\\') {
        escaped = true;
      } else if (c == '"') {
        in_string = false;
      }
      continue;
    }
    if (c == '"') {
      in_string = true;
      continue;
    }
    if (c == '{') {
      ++depth;
    } else if (c == '}') {
      if (depth == 0) {
        return {};
      }
      --depth;
      if (depth == 0) {
        return json.substr(object_begin, i - object_begin + 1);
      }
    }
  }
  return {};
}

GfxProfileDigest digest_gfx_profile(std::string_view json) {
  GfxProfileDigest digest;
  if (json.empty() || json.find("GFX_PROFILE_ERROR") == 0) {
    return digest;
  }
  digest.available = true;
  const auto compile = extract_profile_object(json, "compile");
  const auto compile_summary = extract_profile_object(compile, "summary");
  const auto compile_counters =
      extract_profile_object(compile_summary, "counter_map");
  const auto extended = extract_profile_object(json, "extended");
  const auto extended_summary = extract_profile_object(extended, "summary");
  const auto runtime_counters =
      extract_profile_object(extended_summary, "counter_map");
  const auto runtime_source = extended.empty() ? json : extended;
  const auto compile_source =
      compile_counters.empty() ? json : compile_counters;
  digest.total_gpu_us = extract_profile_uint(runtime_source, "total_gpu_us");
  digest.total_wall_us = extract_profile_uint(runtime_source, "total_wall_us");
  digest.submit_count =
      std::max(extract_profile_uint(runtime_counters, "submit_count"),
               extract_profile_uint(runtime_counters, "vkQueueSubmit_count"));
  digest.barrier_count =
      extract_profile_uint(runtime_counters, "barrier_count") +
      extract_profile_uint(runtime_counters, "cross_submit_barrier_count");
  digest.descriptor_update_count =
      extract_profile_uint(runtime_counters, "descriptor_update_count");
  digest.pipeline_creation_count =
      extract_profile_uint(runtime_counters, "pipeline_creation_count");
  digest.conv_dispatch_tile_h =
      extract_profile_uint(compile_source, "conv_dispatch_tile_h");
  digest.conv_dispatch_tile_w =
      extract_profile_uint(compile_source, "conv_dispatch_tile_w");
  digest.conv_dispatch_threads_h =
      extract_profile_uint(compile_source, "conv_dispatch_threads_h");
  digest.conv_dispatch_threads_w =
      extract_profile_uint(compile_source, "conv_dispatch_threads_w");
  digest.conv_dispatch_channel_block =
      extract_profile_uint(compile_source, "conv_dispatch_channel_block");
  digest.conv_output_reuse_lanes =
      extract_profile_uint(compile_source, "conv_output_reuse_lanes");
  digest.conv_spatial_input_reuse_lanes =
      extract_profile_uint(compile_source, "conv_spatial_input_reuse_lanes");
  digest.conv_spatial_input_reuse_saved_width_loads = extract_profile_uint(
      compile_source, "conv_spatial_input_reuse_saved_width_loads");
  digest.conv_workgroup_reduction_lanes =
      extract_profile_uint(compile_source, "conv_workgroup_reduction_lanes");
  digest.conv_workgroup_output_lanes =
      extract_profile_uint(compile_source, "conv_workgroup_output_lanes");
  digest.conv_multi_kernel_coarse_output_tile_elements = extract_profile_uint(
      compile_source, "conv_multi_kernel_coarse_output_tile_elements");
  digest.conv_multi_kernel_workgroup_output_tile_deficit = extract_profile_uint(
      compile_source, "conv_multi_kernel_workgroup_output_tile_deficit");
  digest.conv_multi_kernel_workgroup_local_accumulator_bytes =
      extract_profile_uint(
          compile_source,
          "conv_multi_kernel_workgroup_local_accumulator_bytes");
  digest.runtime_dispatch_grid_x =
      extract_profile_uint(runtime_counters, "runtime_dispatch_grid_x");
  digest.runtime_dispatch_grid_y =
      extract_profile_uint(runtime_counters, "runtime_dispatch_grid_y");
  digest.runtime_dispatch_grid_z =
      extract_profile_uint(runtime_counters, "runtime_dispatch_grid_z");
  return digest;
}

void print_gfx_profile_digest_header() {
  std::cout << ",gfx_profile_available,gfx_total_gpu_us,gfx_total_wall_us,"
               "gfx_submit_count,gfx_barrier_count,gfx_descriptor_update_count,"
               "gfx_pipeline_creation_count,gfx_conv_dispatch_tile_h,"
               "gfx_conv_dispatch_tile_w,gfx_conv_dispatch_threads_h,"
               "gfx_conv_dispatch_threads_w,gfx_conv_dispatch_channel_block,"
               "gfx_conv_output_reuse_lanes,gfx_conv_spatial_input_reuse_lanes,"
               "gfx_conv_spatial_input_reuse_saved_width_loads,"
               "gfx_conv_workgroup_reduction_lanes,"
               "gfx_conv_workgroup_output_lanes,"
               "gfx_conv_coarse_output_tile_elements,"
               "gfx_conv_workgroup_output_tile_deficit,"
               "gfx_conv_workgroup_local_accumulator_bytes,"
               "gfx_runtime_dispatch_grid_x,gfx_runtime_dispatch_grid_y,"
               "gfx_runtime_dispatch_grid_z";
}

void print_gfx_profile_digest_csv(const RunStats &stats) {
  const auto digest = digest_gfx_profile(stats.gfx_profile_json);
  std::cout << "," << (digest.available ? 1 : 0) << "," << digest.total_gpu_us
            << "," << digest.total_wall_us << "," << digest.submit_count << ","
            << digest.barrier_count << "," << digest.descriptor_update_count
            << "," << digest.pipeline_creation_count << ","
            << digest.conv_dispatch_tile_h << "," << digest.conv_dispatch_tile_w
            << "," << digest.conv_dispatch_threads_h << ","
            << digest.conv_dispatch_threads_w << ","
            << digest.conv_dispatch_channel_block << ","
            << digest.conv_output_reuse_lanes << ","
            << digest.conv_spatial_input_reuse_lanes << ","
            << digest.conv_spatial_input_reuse_saved_width_loads << ","
            << digest.conv_workgroup_reduction_lanes << ","
            << digest.conv_workgroup_output_lanes << ","
            << digest.conv_multi_kernel_coarse_output_tile_elements << ","
            << digest.conv_multi_kernel_workgroup_output_tile_deficit << ","
            << digest.conv_multi_kernel_workgroup_local_accumulator_bytes << ","
            << digest.runtime_dispatch_grid_x << ","
            << digest.runtime_dispatch_grid_y << ","
            << digest.runtime_dispatch_grid_z;
}

struct CompareTolerance {
  double abs_threshold = 0.0;
  double rel_threshold = 0.0;
};

struct DiffStats {
  size_t elements = 0;
  double max_abs_diff = 0.0;
  double max_rel_diff = 0.0;
  size_t tolerance_violations = 0;
  size_t first_violation_index = 0;
  double first_violation_ref = 0.0;
  double first_violation_gfx = 0.0;
  double first_violation_abs = 0.0;
  double first_violation_rel = 0.0;
};

struct CompareResult {
  DiffStats stats;
  CompareTolerance tolerance;
};

CompareTolerance make_tolerance(const Options &options,
                                const ov::Tensor &expected,
                                const ov::Tensor &actual) {
  constexpr double kToleranceFloor = 1e-4;
  const auto tolerance = ov::test::utils::gfx_accuracy_tolerance(
      expected.get_element_type(), actual.get_element_type(), ov::element::f16,
      kToleranceFloor, kToleranceFloor, options.abs_threshold,
      options.rel_threshold);
  return {tolerance.abs_threshold, tolerance.rel_threshold};
}

bool outside_tolerance(double abs_diff, double rel_diff,
                       const CompareTolerance &tolerance) {
  return abs_diff > tolerance.abs_threshold &&
         rel_diff > tolerance.rel_threshold;
}

template <typename T>
DiffStats compare_typed_tensors(const ov::Tensor &expected,
                                const ov::Tensor &actual,
                                const CompareTolerance &tolerance) {
  const auto *expected_data = expected.data<const T>();
  const auto *actual_data = actual.data<const T>();
  DiffStats stats;
  stats.elements = expected.get_size();
  for (size_t i = 0; i < stats.elements; ++i) {
    const double ref = static_cast<double>(expected_data[i]);
    const double gfx = static_cast<double>(actual_data[i]);
    const double abs_diff = std::abs(ref - gfx);
    const double denom = std::max({std::abs(ref), std::abs(gfx), 1e-12});
    const double rel_diff = abs_diff / denom;
    stats.max_abs_diff = std::max(stats.max_abs_diff, abs_diff);
    stats.max_rel_diff = std::max(stats.max_rel_diff, rel_diff);
    if (outside_tolerance(abs_diff, rel_diff, tolerance)) {
      if (stats.tolerance_violations == 0) {
        stats.first_violation_index = i;
        stats.first_violation_ref = ref;
        stats.first_violation_gfx = gfx;
        stats.first_violation_abs = abs_diff;
        stats.first_violation_rel = rel_diff;
      }
      ++stats.tolerance_violations;
    }
  }
  return stats;
}

DiffStats compare_tensors(const ov::Tensor &expected, const ov::Tensor &actual,
                          const CompareTolerance &tolerance) {
  if (expected.get_element_type() != actual.get_element_type()) {
    throw std::runtime_error("output type mismatch");
  }
  if (expected.get_shape() != actual.get_shape()) {
    throw std::runtime_error("output shape mismatch");
  }
  switch (expected.get_element_type()) {
  case ov::element::f32:
    return compare_typed_tensors<float>(expected, actual, tolerance);
  case ov::element::f16:
    return compare_typed_tensors<ov::float16>(expected, actual, tolerance);
  case ov::element::i32:
    return compare_typed_tensors<int32_t>(expected, actual, tolerance);
  case ov::element::i64:
    return compare_typed_tensors<int64_t>(expected, actual, tolerance);
  case ov::element::u8:
    return compare_typed_tensors<uint8_t>(expected, actual, tolerance);
  case ov::element::boolean:
    return compare_typed_tensors<uint8_t>(expected, actual, tolerance);
  default:
    throw std::runtime_error("unsupported output type: " +
                             expected.get_element_type().to_string());
  }
}

RunStats run_model(ov::Core &core, const std::shared_ptr<ov::Model> &model,
                   const std::string &device, const Options &options) {
  const auto compile_start = std::chrono::steady_clock::now();
  const bool for_gfx = device == "GFX";
  auto compiled =
      core.compile_model(model, device, make_config(options, for_gfx));
  const auto compile_stop = std::chrono::steady_clock::now();

  auto request = compiled.create_infer_request();
  ov::Tensor input_tensor(model->input().get_element_type(),
                          model->input().get_shape());
  fill_input(input_tensor);
  request.set_input_tensor(input_tensor);

  std::vector<double> infer_ms;
  infer_ms.reserve(options.iterations);
  const size_t total_iterations = options.warmup + options.iterations;
  for (size_t i = 0; i < total_iterations; ++i) {
    const auto start = std::chrono::steady_clock::now();
    request.infer();
    const auto stop = std::chrono::steady_clock::now();
    if (i >= options.warmup) {
      infer_ms.push_back(
          std::chrono::duration<double, std::milli>(stop - start).count());
    }
  }

  RunStats stats;
  stats.compile_ms =
      std::chrono::duration<double, std::milli>(compile_stop - compile_start)
          .count();
  stats.median_infer_ms = median(infer_ms);
  stats.min_infer_ms = *std::min_element(infer_ms.begin(), infer_ms.end());
  if (for_gfx && options.dump_gfx_profile) {
    try {
      stats.gfx_profile_json =
          compiled.get_property("GFX_PROFILING_REPORT").as<std::string>();
    } catch (const std::exception &ex) {
      stats.gfx_profile_json = std::string{"GFX_PROFILE_ERROR "} + ex.what();
    }
  }
  return stats;
}

void set_input_split_tensors(ov::InferRequest &request,
                             const std::shared_ptr<ov::Model> &model,
                             const ov::Shape &full_input_shape) {
  const size_t input_count = model->inputs().size();
  if (input_count == 0) {
    throw std::runtime_error("input-split candidate has no inputs");
  }
  size_t channel_offset = 0;
  for (size_t i = 0; i < input_count; ++i) {
    ov::Tensor tensor(model->input(i).get_element_type(),
                      model->input(i).get_shape());
    fill_input_channel_chunk(tensor, full_input_shape, channel_offset);
    channel_offset += model->input(i).get_shape()[1];
    request.set_input_tensor(i, tensor);
  }
}

void set_stride2_sublattice_tensors(ov::InferRequest &request,
                                    const std::shared_ptr<ov::Model> &model,
                                    const ov::Shape &full_input_shape) {
  if (model->inputs().size() != 9) {
    throw std::runtime_error("stride2 sublattice candidate expects 9 inputs");
  }
  for (size_t kh = 0; kh < 3; ++kh) {
    for (size_t kw = 0; kw < 3; ++kw) {
      const size_t input_index = kh * 3 + kw;
      ov::Tensor tensor(model->input(input_index).get_element_type(),
                        model->input(input_index).get_shape());
      fill_input_stride2_sublattice(tensor, full_input_shape, kh, kw);
      request.set_input_tensor(input_index, tensor);
    }
  }
}

void set_stride2_row_split_tensors(ov::InferRequest &request,
                                   const std::shared_ptr<ov::Model> &model,
                                   const ov::Shape &full_input_shape) {
  if (model->inputs().size() != 3) {
    throw std::runtime_error("stride2 row-split candidate expects 3 inputs");
  }
  for (size_t kh = 0; kh < 3; ++kh) {
    ov::Tensor tensor(model->input(kh).get_element_type(),
                      model->input(kh).get_shape());
    fill_input_stride2_row_split(tensor, full_input_shape, kh);
    request.set_input_tensor(kh, tensor);
  }
}

RunStats run_input_split_model(ov::Core &core,
                               const std::shared_ptr<ov::Model> &model,
                               const std::string &device,
                               const Options &options,
                               const ov::Shape &full_input_shape) {
  const auto compile_start = std::chrono::steady_clock::now();
  const bool for_gfx = device == "GFX";
  auto compiled =
      core.compile_model(model, device, make_config(options, for_gfx));
  const auto compile_stop = std::chrono::steady_clock::now();

  auto request = compiled.create_infer_request();
  set_input_split_tensors(request, model, full_input_shape);

  std::vector<double> infer_ms;
  infer_ms.reserve(options.iterations);
  const size_t total_iterations = options.warmup + options.iterations;
  for (size_t i = 0; i < total_iterations; ++i) {
    const auto start = std::chrono::steady_clock::now();
    request.infer();
    const auto stop = std::chrono::steady_clock::now();
    if (i >= options.warmup) {
      infer_ms.push_back(
          std::chrono::duration<double, std::milli>(stop - start).count());
    }
  }

  RunStats stats;
  stats.compile_ms =
      std::chrono::duration<double, std::milli>(compile_stop - compile_start)
          .count();
  stats.median_infer_ms = median(infer_ms);
  stats.min_infer_ms = *std::min_element(infer_ms.begin(), infer_ms.end());
  return stats;
}

RunStats run_stride2_sublattice_model(ov::Core &core,
                                      const std::shared_ptr<ov::Model> &model,
                                      const std::string &device,
                                      const Options &options,
                                      const ov::Shape &full_input_shape) {
  const auto compile_start = std::chrono::steady_clock::now();
  const bool for_gfx = device == "GFX";
  auto compiled =
      core.compile_model(model, device, make_config(options, for_gfx));
  const auto compile_stop = std::chrono::steady_clock::now();

  auto request = compiled.create_infer_request();
  set_stride2_sublattice_tensors(request, model, full_input_shape);

  std::vector<double> infer_ms;
  infer_ms.reserve(options.iterations);
  const size_t total_iterations = options.warmup + options.iterations;
  for (size_t i = 0; i < total_iterations; ++i) {
    const auto start = std::chrono::steady_clock::now();
    request.infer();
    const auto stop = std::chrono::steady_clock::now();
    if (i >= options.warmup) {
      infer_ms.push_back(
          std::chrono::duration<double, std::milli>(stop - start).count());
    }
  }

  RunStats stats;
  stats.compile_ms =
      std::chrono::duration<double, std::milli>(compile_stop - compile_start)
          .count();
  stats.median_infer_ms = median(infer_ms);
  stats.min_infer_ms = *std::min_element(infer_ms.begin(), infer_ms.end());
  return stats;
}

RunStats run_stride2_row_split_model(ov::Core &core,
                                     const std::shared_ptr<ov::Model> &model,
                                     const std::string &device,
                                     const Options &options,
                                     const ov::Shape &full_input_shape) {
  const auto compile_start = std::chrono::steady_clock::now();
  const bool for_gfx = device == "GFX";
  auto compiled =
      core.compile_model(model, device, make_config(options, for_gfx));
  const auto compile_stop = std::chrono::steady_clock::now();

  auto request = compiled.create_infer_request();
  set_stride2_row_split_tensors(request, model, full_input_shape);

  std::vector<double> infer_ms;
  infer_ms.reserve(options.iterations);
  const size_t total_iterations = options.warmup + options.iterations;
  for (size_t i = 0; i < total_iterations; ++i) {
    const auto start = std::chrono::steady_clock::now();
    request.infer();
    const auto stop = std::chrono::steady_clock::now();
    if (i >= options.warmup) {
      infer_ms.push_back(
          std::chrono::duration<double, std::milli>(stop - start).count());
    }
  }

  RunStats stats;
  stats.compile_ms =
      std::chrono::duration<double, std::milli>(compile_stop - compile_start)
          .count();
  stats.median_infer_ms = median(infer_ms);
  stats.min_infer_ms = *std::min_element(infer_ms.begin(), infer_ms.end());
  return stats;
}

RunStats run_case(ov::Core &core, const ShapeCase &c, const std::string &device,
                  const Options &options) {
  return run_model(core, make_conv_model(c), device, options);
}

CompareResult run_compare_case(ov::Core &core, const ShapeCase &c,
                               const Options &options) {
  auto reference_model = make_conv_model(c);
  auto gfx_model = make_conv_model(c);
  auto reference = core.compile_model(reference_model, options.reference_device,
                                      make_config(options, false));
  auto gfx = core.compile_model(gfx_model, "GFX", make_config(options, true));

  ov::Tensor input_tensor(reference_model->input().get_element_type(),
                          reference_model->input().get_shape());
  fill_input(input_tensor);

  auto reference_request = reference.create_infer_request();
  reference_request.set_input_tensor(input_tensor);
  reference_request.infer();

  auto gfx_request = gfx.create_infer_request();
  gfx_request.set_input_tensor(input_tensor);
  gfx_request.infer();

  const auto reference_output = reference_request.get_output_tensor();
  const auto gfx_output = gfx_request.get_output_tensor();
  CompareResult result;
  result.tolerance = make_tolerance(options, reference_output, gfx_output);
  result.stats =
      compare_tensors(reference_output, gfx_output, result.tolerance);
  return result;
}

CompareResult run_model_pair_compare(
    ov::Core &core, const std::shared_ptr<ov::Model> &reference_model,
    const std::string &reference_device,
    const std::shared_ptr<ov::Model> &candidate_model,
    const std::string &candidate_device, const Options &options) {
  auto reference = core.compile_model(reference_model, reference_device,
                                      make_config(options, false));
  auto candidate =
      core.compile_model(candidate_model, candidate_device,
                         make_config(options, candidate_device == "GFX"));

  ov::Tensor input_tensor(reference_model->input().get_element_type(),
                          reference_model->input().get_shape());
  fill_input(input_tensor);

  auto reference_request = reference.create_infer_request();
  reference_request.set_input_tensor(input_tensor);
  reference_request.infer();

  auto candidate_request = candidate.create_infer_request();
  candidate_request.set_input_tensor(input_tensor);
  candidate_request.infer();

  const auto reference_output = reference_request.get_output_tensor();
  const auto candidate_output = candidate_request.get_output_tensor();
  CompareResult result;
  result.tolerance =
      make_tolerance(options, reference_output, candidate_output);
  result.stats =
      compare_tensors(reference_output, candidate_output, result.tolerance);
  return result;
}

CompareResult run_model_compare(
    ov::Core &core, const std::shared_ptr<ov::Model> &reference_model,
    const std::shared_ptr<ov::Model> &gfx_model, const Options &options) {
  return run_model_pair_compare(core, reference_model, options.reference_device,
                                gfx_model, "GFX", options);
}

CompareResult run_input_split_model_compare(
    ov::Core &core, const std::shared_ptr<ov::Model> &reference_model,
    const std::shared_ptr<ov::Model> &candidate_model,
    const std::string &candidate_device, const Options &options,
    const ov::Shape &full_input_shape) {
  auto reference = core.compile_model(reference_model, options.reference_device,
                                      make_config(options, false));
  auto candidate =
      core.compile_model(candidate_model, candidate_device,
                         make_config(options, candidate_device == "GFX"));

  ov::Tensor input_tensor(reference_model->input().get_element_type(),
                          reference_model->input().get_shape());
  fill_input(input_tensor);

  auto reference_request = reference.create_infer_request();
  reference_request.set_input_tensor(input_tensor);
  reference_request.infer();

  auto candidate_request = candidate.create_infer_request();
  set_input_split_tensors(candidate_request, candidate_model, full_input_shape);
  candidate_request.infer();

  const auto reference_output = reference_request.get_output_tensor();
  const auto candidate_output = candidate_request.get_output_tensor();
  CompareResult result;
  result.tolerance =
      make_tolerance(options, reference_output, candidate_output);
  result.stats =
      compare_tensors(reference_output, candidate_output, result.tolerance);
  return result;
}

CompareResult run_stride2_sublattice_model_compare(
    ov::Core &core, const std::shared_ptr<ov::Model> &reference_model,
    const std::shared_ptr<ov::Model> &candidate_model,
    const std::string &candidate_device, const Options &options,
    const ov::Shape &full_input_shape) {
  auto reference = core.compile_model(reference_model, options.reference_device,
                                      make_config(options, false));
  auto candidate =
      core.compile_model(candidate_model, candidate_device,
                         make_config(options, candidate_device == "GFX"));

  ov::Tensor input_tensor(reference_model->input().get_element_type(),
                          reference_model->input().get_shape());
  fill_input(input_tensor);

  auto reference_request = reference.create_infer_request();
  reference_request.set_input_tensor(input_tensor);
  reference_request.infer();

  auto candidate_request = candidate.create_infer_request();
  set_stride2_sublattice_tensors(candidate_request, candidate_model,
                                 full_input_shape);
  candidate_request.infer();

  const auto reference_output = reference_request.get_output_tensor();
  const auto candidate_output = candidate_request.get_output_tensor();
  CompareResult result;
  result.tolerance =
      make_tolerance(options, reference_output, candidate_output);
  result.stats =
      compare_tensors(reference_output, candidate_output, result.tolerance);
  return result;
}

CompareResult run_stride2_row_split_model_compare(
    ov::Core &core, const std::shared_ptr<ov::Model> &reference_model,
    const std::shared_ptr<ov::Model> &candidate_model,
    const std::string &candidate_device, const Options &options,
    const ov::Shape &full_input_shape) {
  auto reference = core.compile_model(reference_model, options.reference_device,
                                      make_config(options, false));
  auto candidate =
      core.compile_model(candidate_model, candidate_device,
                         make_config(options, candidate_device == "GFX"));

  ov::Tensor input_tensor(reference_model->input().get_element_type(),
                          reference_model->input().get_shape());
  fill_input(input_tensor);

  auto reference_request = reference.create_infer_request();
  reference_request.set_input_tensor(input_tensor);
  reference_request.infer();

  auto candidate_request = candidate.create_infer_request();
  set_stride2_row_split_tensors(candidate_request, candidate_model,
                                full_input_shape);
  candidate_request.infer();

  const auto reference_output = reference_request.get_output_tensor();
  const auto candidate_output = candidate_request.get_output_tensor();
  CompareResult result;
  result.tolerance =
      make_tolerance(options, reference_output, candidate_output);
  result.stats =
      compare_tensors(reference_output, candidate_output, result.tolerance);
  return result;
}

std::vector<ShapeCase> yolo26x_cases() {
  return {
      {"yolo26x_model_1_conv_s2",
       {1, 96, 320, 320},
       {192, 96, 3, 3},
       {2, 2},
       {1, 1},
       {1, 1},
       {1, 1}},
      {"yolo26x_model_3_conv_s2",
       {1, 384, 160, 160},
       {384, 384, 3, 3},
       {2, 2},
       {1, 1},
       {1, 1},
       {1, 1}},
      {"yolo26x_model_5_conv_s2",
       {1, 768, 80, 80},
       {768, 768, 3, 3},
       {2, 2},
       {1, 1},
       {1, 1},
       {1, 1}},
      {"yolo26x_c2_48_48_k3_160",
       {1, 48, 160, 160},
       {48, 48, 3, 3},
       {1, 1},
       {1, 1},
       {1, 1},
       {1, 1}},
      {"yolo26x_c4_96_96_k3_80",
       {1, 96, 80, 80},
       {96, 96, 3, 3},
       {1, 1},
       {1, 1},
       {1, 1},
       {1, 1}},
      {"yolo26x_c6_192_192_k3_40",
       {1, 192, 40, 40},
       {192, 192, 3, 3},
       {1, 1},
       {1, 1},
       {1, 1},
       {1, 1}},
      {"yolo26x_head_384_96_k3_80",
       {1, 384, 80, 80},
       {96, 384, 3, 3},
       {1, 1},
       {1, 1},
       {1, 1},
       {1, 1}},
      {"yolo26x_pw_48_48_160",
       {1, 48, 160, 160},
       {48, 48, 1, 1},
       {1, 1},
       {0, 0},
       {0, 0},
       {1, 1}},
      {"yolo26x_pw_96_96_80",
       {1, 96, 80, 80},
       {96, 96, 1, 1},
       {1, 1},
       {0, 0},
       {0, 0},
       {1, 1}},
      {"yolo26x_pw_192_192_160",
       {1, 192, 160, 160},
       {192, 192, 1, 1},
       {1, 1},
       {0, 0},
       {0, 0},
       {1, 1}},
      {"yolo26x_pw_384_384_80",
       {1, 384, 80, 80},
       {384, 384, 1, 1},
       {1, 1},
       {0, 0},
       {0, 0},
       {1, 1}},
      {"yolo26x_pw_768_768_40",
       {1, 768, 40, 40},
       {768, 768, 1, 1},
       {1, 1},
       {0, 0},
       {0, 0},
       {1, 1}},
      {"yolo26x_pw_384_384_160",
       {1, 384, 160, 160},
       {384, 384, 1, 1},
       {1, 1},
       {0, 0},
       {0, 0},
       {1, 1}},
      {"yolo26x_pw_1536_384_80",
       {1, 1536, 80, 80},
       {384, 1536, 1, 1},
       {1, 1},
       {0, 0},
       {0, 0},
       {1, 1}},
  };
}

} // namespace

int main(int argc, char **argv) {
  try {
    const Options options = parse_options(argc, argv);
    ov::Core core;
    register_gfx_plugin(core);
    if (options.compare_template || options.compare_pointwise_matmul ||
        options.compare_input_split_conv || options.compare_output_split_conv ||
        options.compare_stride2_sublattice_conv ||
        options.compare_stride2_row_split_conv) {
      register_reference_plugin(core, options.reference_device,
                                options.reference_plugin_path);
    }
    const auto cases = yolo26x_cases();

    if (options.list_cases) {
      for (const auto &c : cases) {
        std::cout << c.name << "\n";
      }
      return 0;
    }

    std::cout << std::fixed;
    if (options.compare_template) {
      std::cout << std::setprecision(10);
      std::cout << "case,reference_device,gfx_device,elements,max_abs_diff,max_"
                   "rel_diff,"
                   "abs_threshold,rel_threshold,tolerance_violations,first_"
                   "violation_index,"
                   "first_violation_ref,first_violation_gfx,first_violation_"
                   "abs,first_violation_rel\n";
      bool ran_any_case = false;
      size_t total_violations = 0;
      for (const auto &c : cases) {
        if (!case_matches_filters(c, options.case_filters)) {
          continue;
        }
        ran_any_case = true;
        const auto result = run_compare_case(core, c, options);
        total_violations += result.stats.tolerance_violations;
        std::cout << c.name << "," << options.reference_device << ",GFX,"
                  << result.stats.elements << "," << result.stats.max_abs_diff
                  << "," << result.stats.max_rel_diff << ","
                  << result.tolerance.abs_threshold << ","
                  << result.tolerance.rel_threshold << ","
                  << result.stats.tolerance_violations << ","
                  << result.stats.first_violation_index << ","
                  << result.stats.first_violation_ref << ","
                  << result.stats.first_violation_gfx << ","
                  << result.stats.first_violation_abs << ","
                  << result.stats.first_violation_rel << "\n";
      }
      if (!ran_any_case) {
        std::cerr << "fatal: no shape cases matched --case filters\n";
        return 1;
      }
      return total_violations == 0 ? 0 : 3;
    }

    if (options.compare_pointwise_matmul) {
      std::cout << std::setprecision(10);
      std::cout << "case,direct_gfx_ms,matmul_gfx_ms,direct_over_matmul,"
                   "matmul_compile_ms,elements,max_abs_diff,max_rel_diff,"
                   "abs_threshold,rel_threshold,tolerance_violations\n";
      bool ran_any_case = false;
      size_t total_violations = 0;
      for (const auto &c : cases) {
        if (!case_matches_filters(c, options.case_filters) ||
            !is_pointwise_matmul_candidate(c)) {
          continue;
        }
        ran_any_case = true;
        auto direct_model = make_conv_model(c);
        auto matmul_model = make_pointwise_matmul_model(c);
        const auto direct_stats = run_model(core, direct_model, "GFX", options);
        const auto matmul_stats = run_model(core, matmul_model, "GFX", options);
        const auto compare = run_model_compare(
            core, make_conv_model(c), make_pointwise_matmul_model(c), options);
        total_violations += compare.stats.tolerance_violations;
        const double ratio =
            matmul_stats.median_infer_ms > 0.0
                ? direct_stats.median_infer_ms / matmul_stats.median_infer_ms
                : 0.0;
        std::cout << c.name << "," << direct_stats.median_infer_ms << ","
                  << matmul_stats.median_infer_ms << "," << ratio << ","
                  << matmul_stats.compile_ms << "," << compare.stats.elements
                  << "," << compare.stats.max_abs_diff << ","
                  << compare.stats.max_rel_diff << ","
                  << compare.tolerance.abs_threshold << ","
                  << compare.tolerance.rel_threshold << ","
                  << compare.stats.tolerance_violations << "\n";
      }
      if (!ran_any_case) {
        std::cerr
            << "fatal: no pointwise 1x1 shape cases matched --case filters\n";
        return 1;
      }
      return total_violations == 0 ? 0 : 3;
    }

    if (options.compare_input_split_conv) {
      std::cout << std::setprecision(10);
      std::cout << "case,split_parts,direct_gfx_ms,split_gfx_ms,"
                   "direct_over_split,split_compile_ms,elements,max_abs_diff,"
                   "max_rel_diff,abs_threshold,rel_threshold,"
                   "tolerance_violations,template_max_abs_diff,"
                   "template_max_rel_diff,template_tolerance_violations\n";
      bool ran_any_case = false;
      size_t total_violations = 0;
      for (const auto &c : cases) {
        if (!case_matches_filters(c, options.case_filters) ||
            !is_input_split_conv_candidate(c, options.input_split_parts)) {
          continue;
        }
        ran_any_case = true;
        auto direct_model = make_conv_model(c);
        auto split_model =
            make_input_split_conv_model(c, options.input_split_parts);
        const auto direct_stats = run_model(core, direct_model, "GFX", options);
        const auto split_stats =
            run_input_split_model(core, split_model, "GFX", options, c.input);
        const auto compare = run_input_split_model_compare(
            core, make_conv_model(c),
            make_input_split_conv_model(c, options.input_split_parts), "GFX",
            options, c.input);
        const auto template_check = run_input_split_model_compare(
            core, make_conv_model(c),
            make_input_split_conv_model(c, options.input_split_parts),
            options.reference_device, options, c.input);
        total_violations += compare.stats.tolerance_violations;
        const double ratio =
            split_stats.median_infer_ms > 0.0
                ? direct_stats.median_infer_ms / split_stats.median_infer_ms
                : 0.0;
        std::cout << c.name << "," << options.input_split_parts << ","
                  << direct_stats.median_infer_ms << ","
                  << split_stats.median_infer_ms << "," << ratio << ","
                  << split_stats.compile_ms << "," << compare.stats.elements
                  << "," << compare.stats.max_abs_diff << ","
                  << compare.stats.max_rel_diff << ","
                  << compare.tolerance.abs_threshold << ","
                  << compare.tolerance.rel_threshold << ","
                  << compare.stats.tolerance_violations << ","
                  << template_check.stats.max_abs_diff << ","
                  << template_check.stats.max_rel_diff << ","
                  << template_check.stats.tolerance_violations << "\n";
      }
      if (!ran_any_case) {
        std::cerr << "fatal: no input-split Conv shape cases matched --case "
                     "filters and split-parts\n";
        return 1;
      }
      return total_violations == 0 ? 0 : 3;
    }

    if (options.compare_output_split_conv) {
      std::cout << std::setprecision(10);
      std::cout << "case,split_parts,direct_gfx_ms,split_gfx_ms,"
                   "direct_over_split,split_compile_ms,elements,max_abs_diff,"
                   "max_rel_diff,abs_threshold,rel_threshold,"
                   "tolerance_violations,template_max_abs_diff,"
                   "template_max_rel_diff,template_tolerance_violations\n";
      bool ran_any_case = false;
      size_t total_violations = 0;
      for (const auto &c : cases) {
        if (!case_matches_filters(c, options.case_filters) ||
            !is_output_split_conv_candidate(c, options.output_split_parts)) {
          continue;
        }
        ran_any_case = true;
        auto direct_model = make_conv_model(c);
        auto split_model =
            make_output_split_conv_model(c, options.output_split_parts);
        const auto direct_stats = run_model(core, direct_model, "GFX", options);
        const auto split_stats = run_model(core, split_model, "GFX", options);
        const auto compare = run_model_compare(
            core, make_conv_model(c),
            make_output_split_conv_model(c, options.output_split_parts),
            options);
        const auto template_check = run_model_pair_compare(
            core, make_conv_model(c), options.reference_device,
            make_output_split_conv_model(c, options.output_split_parts),
            options.reference_device, options);
        total_violations += compare.stats.tolerance_violations;
        const double ratio =
            split_stats.median_infer_ms > 0.0
                ? direct_stats.median_infer_ms / split_stats.median_infer_ms
                : 0.0;
        std::cout << c.name << "," << options.output_split_parts << ","
                  << direct_stats.median_infer_ms << ","
                  << split_stats.median_infer_ms << "," << ratio << ","
                  << split_stats.compile_ms << "," << compare.stats.elements
                  << "," << compare.stats.max_abs_diff << ","
                  << compare.stats.max_rel_diff << ","
                  << compare.tolerance.abs_threshold << ","
                  << compare.tolerance.rel_threshold << ","
                  << compare.stats.tolerance_violations << ","
                  << template_check.stats.max_abs_diff << ","
                  << template_check.stats.max_rel_diff << ","
                  << template_check.stats.tolerance_violations << "\n";
      }
      if (!ran_any_case) {
        std::cerr << "fatal: no output-split Conv shape cases matched --case "
                     "filters and split-parts\n";
        return 1;
      }
      return total_violations == 0 ? 0 : 3;
    }

    if (options.compare_stride2_sublattice_conv) {
      std::cout << std::setprecision(10);
      std::cout << "case,direct_gfx_ms,sublattice_gfx_ms,"
                   "direct_over_sublattice,sublattice_compile_ms,elements,"
                   "max_abs_diff,max_rel_diff,abs_threshold,rel_threshold,"
                   "tolerance_violations,template_max_abs_diff,"
                   "template_max_rel_diff,template_tolerance_violations\n";
      bool ran_any_case = false;
      size_t total_violations = 0;
      for (const auto &c : cases) {
        if (!case_matches_filters(c, options.case_filters) ||
            !is_stride2_sublattice_conv_candidate(c)) {
          continue;
        }
        ran_any_case = true;
        const auto valid_case = make_stride2_valid_core_case(c);
        auto direct_model = make_conv_model(valid_case);
        auto sublattice_model = make_stride2_sublattice_conv_model(valid_case);
        const auto direct_stats = run_model(core, direct_model, "GFX", options);
        const auto sublattice_stats =
            run_stride2_sublattice_model(core, sublattice_model, "GFX",
                                         options, valid_case.input);
        const auto compare = run_stride2_sublattice_model_compare(
            core, make_conv_model(valid_case),
            make_stride2_sublattice_conv_model(valid_case), "GFX", options,
            valid_case.input);
        const auto template_check = run_stride2_sublattice_model_compare(
            core, make_conv_model(valid_case),
            make_stride2_sublattice_conv_model(valid_case),
            options.reference_device, options, valid_case.input);
        total_violations += compare.stats.tolerance_violations;
        const double ratio = sublattice_stats.median_infer_ms > 0.0
                                 ? direct_stats.median_infer_ms /
                                       sublattice_stats.median_infer_ms
                                 : 0.0;
        std::cout << valid_case.name << "," << direct_stats.median_infer_ms
                  << "," << sublattice_stats.median_infer_ms << "," << ratio
                  << "," << sublattice_stats.compile_ms << ","
                  << compare.stats.elements << "," << compare.stats.max_abs_diff
                  << "," << compare.stats.max_rel_diff << ","
                  << compare.tolerance.abs_threshold << ","
                  << compare.tolerance.rel_threshold << ","
                  << compare.stats.tolerance_violations << ","
                  << template_check.stats.max_abs_diff << ","
                  << template_check.stats.max_rel_diff << ","
                  << template_check.stats.tolerance_violations << "\n";
      }
      if (!ran_any_case) {
        std::cerr << "fatal: no stride-2 sublattice Conv shape cases matched "
                     "--case filters\n";
        return 1;
      }
      return total_violations == 0 ? 0 : 3;
    }

    if (options.compare_stride2_row_split_conv) {
      std::cout << std::setprecision(10);
      std::cout << "case,direct_gfx_ms,row_split_gfx_ms,"
                   "direct_over_row_split,row_split_compile_ms,elements,"
                   "max_abs_diff,max_rel_diff,abs_threshold,rel_threshold,"
                   "tolerance_violations,template_max_abs_diff,"
                   "template_max_rel_diff,template_tolerance_violations\n";
      bool ran_any_case = false;
      size_t total_violations = 0;
      for (const auto &c : cases) {
        if (!case_matches_filters(c, options.case_filters) ||
            !is_stride2_sublattice_conv_candidate(c)) {
          continue;
        }
        ran_any_case = true;
        const auto valid_case = make_stride2_valid_core_case(c);
        auto direct_model = make_conv_model(valid_case);
        auto row_split_model =
            make_stride2_row_split_conv_model(valid_case);
        const auto direct_stats = run_model(core, direct_model, "GFX", options);
        const auto row_split_stats =
            run_stride2_row_split_model(core, row_split_model, "GFX", options,
                                        valid_case.input);
        const auto compare = run_stride2_row_split_model_compare(
            core, make_conv_model(valid_case),
            make_stride2_row_split_conv_model(valid_case), "GFX", options,
            valid_case.input);
        const auto template_check = run_stride2_row_split_model_compare(
            core, make_conv_model(valid_case),
            make_stride2_row_split_conv_model(valid_case),
            options.reference_device, options, valid_case.input);
        total_violations += compare.stats.tolerance_violations;
        const double ratio =
            row_split_stats.median_infer_ms > 0.0
                ? direct_stats.median_infer_ms /
                      row_split_stats.median_infer_ms
                : 0.0;
        std::cout << valid_case.name << "," << direct_stats.median_infer_ms
                  << "," << row_split_stats.median_infer_ms << "," << ratio
                  << "," << row_split_stats.compile_ms << ","
                  << compare.stats.elements << "," << compare.stats.max_abs_diff
                  << "," << compare.stats.max_rel_diff << ","
                  << compare.tolerance.abs_threshold << ","
                  << compare.tolerance.rel_threshold << ","
                  << compare.stats.tolerance_violations << ","
                  << template_check.stats.max_abs_diff << ","
                  << template_check.stats.max_rel_diff << ","
                  << template_check.stats.tolerance_violations << "\n";
      }
      if (!ran_any_case) {
        std::cerr << "fatal: no stride-2 row-split Conv shape cases matched "
                     "--case filters\n";
        return 1;
      }
      return total_violations == 0 ? 0 : 3;
    }

    std::cout << std::setprecision(3);
    std::cout << "case,device,compile_ms,median_infer_ms,min_infer_ms,fps";
    if (options.dump_gfx_profile) {
      print_gfx_profile_digest_header();
    }
    std::cout << "\n";
    bool ran_any_case = false;
    for (const auto &c : cases) {
      if (!case_matches_filters(c, options.case_filters)) {
        continue;
      }
      ran_any_case = true;
      for (const auto &device : options.devices) {
        const RunStats stats = run_case(core, c, device, options);
        const double fps =
            stats.median_infer_ms > 0.0 ? 1000.0 / stats.median_infer_ms : 0.0;
        std::cout << c.name << "," << device << "," << stats.compile_ms << ","
                  << stats.median_infer_ms << "," << stats.min_infer_ms << ","
                  << fps;
        if (options.dump_gfx_profile) {
          print_gfx_profile_digest_csv(stats);
        }
        std::cout << "\n";
        if (!stats.gfx_profile_json.empty()) {
          std::cout << "GFX_PROFILE " << c.name << "," << device << ","
                    << stats.gfx_profile_json << "\n";
        }
      }
    }
    if (!ran_any_case) {
      std::cerr << "fatal: no shape cases matched --case filters\n";
      return 1;
    }
    return 0;
  } catch (const std::exception &ex) {
    std::cerr << "fatal: " << ex.what() << "\n";
    return 1;
  }
}
