#include <openvino/op/constant.hpp>
#include <openvino/op/convolution.hpp>
#include <openvino/op/gather.hpp>
#include <openvino/op/parameter.hpp>
#include <openvino/op/result.hpp>
#include <openvino/op/shape_of.hpp>
#include <openvino/op/split.hpp>
#include <openvino/openvino.hpp>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <exception>
#include <iomanip>
#include <iostream>
#include <limits>
#include <numeric>
#include <optional>
#include <sstream>
#include <string>
#include <type_traits>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include "common_test_utils/ov_plugin_cache.hpp"

namespace {

constexpr const char *kGfxDiagnosticF32MpsImageProperty =
    "GFX_DIAGNOSTIC_F32_MPS_IMAGE";

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
    const auto devices = core.get_available_devices();
    if (std::find(devices.begin(), devices.end(), reference_device_name) ==
        devices.end()) {
      core.register_plugin(reference_plugin_path, reference_device_name);
    }
  }
  if (reference_device_name == "TEMPLATE") {
    try {
      ov::test::utils::register_template_plugin(core);
    } catch (...) {
    }
  }
}

std::string reference_device(const ov::Core &core,
                             const std::string &requested_device) {
  if (requested_device == "CPU") {
    throw std::runtime_error(
        "CPU reference device is not supported; use TEMPLATE");
  }
  const auto devices = core.get_available_devices();
  if (std::find(devices.begin(), devices.end(), requested_device) !=
      devices.end()) {
    return requested_device;
  }
  throw std::runtime_error(requested_device +
                           " reference device not available");
}

uint64_t mix_input_value(uint64_t value) {
  value += 0x9e3779b97f4a7c15ULL;
  value = (value ^ (value >> 30)) * 0xbf58476d1ce4e5b9ULL;
  value = (value ^ (value >> 27)) * 0x94d049bb133111ebULL;
  return value ^ (value >> 31);
}

uint64_t deterministic_validation_seed(uint64_t base, size_t index) {
  return mix_input_value(base ^ (0x9e3779b97f4a7c15ULL * (index + 1)));
}

template <typename T>
void fill_tensor_data(ov::Tensor &tensor, uint64_t input_seed) {
  T *data = tensor.data<T>();
  const size_t count = tensor.get_size();
  for (size_t i = 0; i < count; ++i) {
    if (input_seed == 0) {
      data[i] = static_cast<T>(((static_cast<int64_t>(i) % 251) - 125) / 32.0f);
      continue;
    }
    const uint64_t mixed =
        mix_input_value(input_seed ^ static_cast<uint64_t>(i));
    if constexpr (std::is_same_v<T, uint8_t>) {
      data[i] = static_cast<uint8_t>(mixed % 251);
    } else if constexpr (std::is_integral_v<T>) {
      data[i] = static_cast<T>(static_cast<int64_t>(mixed % 251) - 125);
    } else {
      data[i] =
          static_cast<T>((static_cast<int64_t>(mixed % 2001) - 1000) / 257.0f);
    }
  }
}

void fill_tensor(ov::Tensor &tensor, uint64_t input_seed = 0) {
  switch (tensor.get_element_type()) {
  case ov::element::f32:
    fill_tensor_data<float>(tensor, input_seed);
    return;
  case ov::element::f16:
    fill_tensor_data<ov::float16>(tensor, input_seed);
    return;
  case ov::element::i32:
    fill_tensor_data<int32_t>(tensor, input_seed);
    return;
  case ov::element::i64:
    fill_tensor_data<int64_t>(tensor, input_seed);
    return;
  case ov::element::u8:
    fill_tensor_data<uint8_t>(tensor, input_seed);
    return;
  default:
    throw std::runtime_error("unsupported input type: " +
                             tensor.get_element_type().to_string());
  }
}

ov::Tensor clone_tensor_data(const ov::Tensor &tensor) {
  ov::Tensor clone(tensor.get_element_type(), tensor.get_shape());
  const size_t byte_size = tensor.get_byte_size();
  if (byte_size > 0) {
    std::memcpy(clone.data(), tensor.data(), byte_size);
  }
  return clone;
}

struct DiffStats {
  double max_abs_diff = 0.0;
  double max_rel_diff = 0.0;
  size_t elements = 0;
  size_t max_index = 0;
  double lhs_at_max = 0.0;
  double rhs_at_max = 0.0;
};

template <typename T>
DiffStats compare_typed(const ov::Tensor &a, const ov::Tensor &b) {
  const T *lhs = a.data<const T>();
  const T *rhs = b.data<const T>();
  DiffStats stats;
  stats.elements = a.get_size();
  for (size_t i = 0; i < stats.elements; ++i) {
    const double da = static_cast<double>(lhs[i]);
    const double db = static_cast<double>(rhs[i]);
    const double abs_diff = std::abs(da - db);
    const double denom = std::max({std::abs(da), std::abs(db), 1e-12});
    const double rel_diff = abs_diff / denom;
    if (abs_diff > stats.max_abs_diff) {
      stats.max_abs_diff = abs_diff;
      stats.max_index = i;
      stats.lhs_at_max = da;
      stats.rhs_at_max = db;
    }
    stats.max_rel_diff = std::max(stats.max_rel_diff, rel_diff);
  }
  return stats;
}

DiffStats compare_tensors(const ov::Tensor &a, const ov::Tensor &b) {
  if (a.get_element_type() != b.get_element_type()) {
    throw std::runtime_error("output type mismatch");
  }
  if (a.get_shape() != b.get_shape()) {
    throw std::runtime_error("output shape mismatch");
  }
  switch (a.get_element_type()) {
  case ov::element::f32:
    return compare_typed<float>(a, b);
  case ov::element::f16:
    return compare_typed<ov::float16>(a, b);
  case ov::element::i32:
    return compare_typed<int32_t>(a, b);
  case ov::element::i64:
    return compare_typed<int64_t>(a, b);
  case ov::element::u8:
    return compare_typed<uint8_t>(a, b);
  case ov::element::boolean:
    return compare_typed<uint8_t>(a, b);
  default:
    throw std::runtime_error("unsupported output type: " +
                             a.get_element_type().to_string());
  }
}

ov::Shape make_static_shape(const ov::PartialShape &ps, int64_t fallback = 1) {
  if (ps.is_static()) {
    return ps.to_shape();
  }
  ov::Shape shape;
  if (ps.rank().is_static()) {
    shape.reserve(ps.rank().get_length());
  }
  for (const auto &dim : ps) {
    shape.push_back(
        static_cast<size_t>(dim.is_static() ? dim.get_length() : fallback));
  }
  if (shape.empty()) {
    shape.push_back(1);
  }
  return shape;
}

bool is_debug_skippable_node(const std::shared_ptr<ov::Node> &node) {
  return ov::is_type<ov::op::v0::Parameter>(node) ||
         ov::is_type<ov::op::v0::Constant>(node) ||
         ov::is_type<ov::op::v0::Result>(node);
}

size_t count_upstream_ops_limited(const ov::Output<ov::Node> &source,
                                  size_t limit) {
  std::vector<std::shared_ptr<ov::Node>> stack;
  std::unordered_set<const ov::Node *> visited;
  if (auto node = source.get_node_shared_ptr()) {
    stack.push_back(node);
  }
  size_t count = 0;
  while (!stack.empty()) {
    auto node = stack.back();
    stack.pop_back();
    if (!node || !visited.insert(node.get()).second) {
      continue;
    }
    if (!is_debug_skippable_node(node)) {
      ++count;
      if (count > limit) {
        return count;
      }
    }
    for (const auto &input : node->input_values()) {
      if (auto input_node = input.get_node_shared_ptr()) {
        stack.push_back(input_node);
      }
    }
  }
  return count;
}

enum class PerOpInputMode {
  Reference,
  Generated,
  GfxRecursive,
};

PerOpInputMode parse_per_op_input_mode(const std::string &value) {
  if (value == "reference") {
    return PerOpInputMode::Reference;
  }
  if (value == "generated") {
    return PerOpInputMode::Generated;
  }
  if (value == "gfx-recursive") {
    return PerOpInputMode::GfxRecursive;
  }
  if (value == "gfx-upstream") {
    throw std::runtime_error("legacy --per-op-input-mode gfx-upstream was "
                             "removed; use gfx-recursive");
  }
  throw std::runtime_error("unsupported --per-op-input-mode: " + value);
}

struct CompareOptions {
  bool per_op = false;
  bool per_op_all = false;
  bool print_ops = false;
  bool gfx_only = false;
  std::string reference_device = "TEMPLATE";
  std::string reference_plugin_path;
  size_t start_op = 0;
  size_t window_size = 1;
  std::optional<size_t> stop_after_op;
  std::optional<size_t> single_output_op;
  size_t single_output_port = 0;
  double abs_threshold = 1e-4;
  double rel_threshold = 1e-4;
  uint64_t input_seed = 0;
  size_t random_seed_count = 0;
  uint64_t random_seed_base = 0;
  bool diagnostic_f32_mps_image = false;
  bool tinyllama_prompt_inputs = false;
  PerOpInputMode per_op_input_mode = PerOpInputMode::Reference;
  size_t per_op_recursive_limit = 0;
  size_t per_op_recursive_trace_every = 0;
};

struct RecursiveMaterializationState {
  size_t limit = 0;
  size_t trace_every = 0;
  size_t materialized = 0;
  std::string last_node;
  std::string failure;
};

struct OutputKey {
  const ov::Node *node = nullptr;
  size_t port = 0;
  bool operator==(const OutputKey &other) const {
    return node == other.node && port == other.port;
  }
};

struct OutputKeyHash {
  size_t operator()(const OutputKey &key) const {
    size_t h1 = std::hash<const ov::Node *>()(key.node);
    size_t h2 = std::hash<size_t>()(key.port);
    return h1 ^ (h2 + 0x9e3779b97f4a7c15ULL + (h1 << 6) + (h1 >> 2));
  }
};

ov::AnyMap make_compile_config(bool for_gfx,
                               const CompareOptions *options = nullptr);
ov::InferRequest make_request(
    ov::CompiledModel &compiled_model,
    const std::vector<std::pair<ov::Output<const ov::Node>, ov::Tensor>>
        &inputs);
void maybe_print_gfx_profile(const ov::CompiledModel &compiled_model);
std::optional<ov::Tensor> evaluate_source_tensor(
    const ov::Output<ov::Node> &source,
    const std::vector<std::pair<ov::Output<const ov::Node>, ov::Tensor>>
        &inputs);
std::optional<ov::Tensor> evaluate_source_tensor_with_compiled_submodel(
    ov::Core &core, const std::string &device, bool for_gfx,
    const ov::Output<ov::Node> &source,
    const std::vector<std::pair<ov::Output<const ov::Node>, ov::Tensor>>
        &inputs,
    const CompareOptions *options = nullptr);
std::optional<ov::Tensor> evaluate_source_tensor_with_reference(
    ov::Core &core, const std::string &ref_device,
    const ov::Output<ov::Node> &source,
    const std::vector<std::pair<ov::Output<const ov::Node>, ov::Tensor>>
        &inputs);

struct TensorSummary {
  size_t elements = 0;
  size_t finite_count = 0;
  size_t nan_count = 0;
  size_t inf_count = 0;
  double min = 0.0;
  double max = 0.0;
  double mean = 0.0;
  double l2 = 0.0;
};

ov::AnyMap make_compile_config(bool for_gfx, const CompareOptions *options) {
  ov::AnyMap config;
  config[ov::hint::inference_precision.name()] = ov::element::f16;
  if (for_gfx) {
    if (options && options->diagnostic_f32_mps_image) {
      config[kGfxDiagnosticF32MpsImageProperty] = true;
    }
    if (options &&
        (options->per_op || options->per_op_all ||
         options->single_output_op.has_value())) {
      config["GFX_ENABLE_FUSION"] = false;
    }
    if (const char *profiling_level = std::getenv("OV_GFX_PROFILING_LEVEL")) {
      if (*profiling_level) {
        config["GFX_PROFILING_LEVEL"] = std::string(profiling_level);
        config[ov::enable_profiling.name()] = true;
        config["PERF_COUNT"] = true;
      }
    }
    if (const char *disable_fusion = std::getenv("OV_GFX_DISABLE_FUSION")) {
      if (std::string(disable_fusion) != "0" &&
          !std::string(disable_fusion).empty()) {
        config["GFX_ENABLE_FUSION"] = false;
      }
    }
  }
  return config;
}

ov::InferRequest make_request(
    ov::CompiledModel &compiled_model,
    const std::vector<std::pair<ov::Output<const ov::Node>, ov::Tensor>>
        &inputs) {
  auto request = compiled_model.create_infer_request();
  for (const auto &[port, tensor] : inputs) {
    request.set_tensor(port, tensor);
  }
  return request;
}

void maybe_print_gfx_profile(const ov::CompiledModel &compiled_model) {
  const char *dump_profile = std::getenv("OV_GFX_DUMP_PROFILE");
  if (!dump_profile || !*dump_profile) {
    return;
  }
  try {
    const auto profile_json =
        compiled_model.get_property("GFX_PROFILING_REPORT").as<std::string>();
    std::cout << "GFX_PROFILE " << profile_json << '\n';
  } catch (const std::exception &ex) {
    std::cout << "GFX_PROFILE_ERROR " << ex.what() << '\n';
  }
}

template <typename T>
TensorSummary summarize_typed_tensor(const ov::Tensor &tensor) {
  const T *data = tensor.data<const T>();
  TensorSummary summary;
  summary.elements = tensor.get_size();
  if (summary.elements == 0) {
    return summary;
  }

  double min_value = 0.0;
  double max_value = 0.0;
  double sum = 0.0;
  double sum_sq = 0.0;
  bool initialized = false;
  for (size_t i = 0; i < summary.elements; ++i) {
    const double value = static_cast<double>(data[i]);
    if (std::isnan(value)) {
      ++summary.nan_count;
      continue;
    }
    if (!std::isfinite(value)) {
      ++summary.inf_count;
      continue;
    }
    ++summary.finite_count;
    if (!initialized) {
      min_value = value;
      max_value = value;
      initialized = true;
    } else {
      min_value = std::min(min_value, value);
      max_value = std::max(max_value, value);
    }
    sum += value;
    sum_sq += value * value;
  }

  if (initialized) {
    summary.min = min_value;
    summary.max = max_value;
  }
  if (summary.finite_count > 0) {
    summary.mean = sum / static_cast<double>(summary.finite_count);
    summary.l2 = std::sqrt(sum_sq);
  }
  return summary;
}

TensorSummary summarize_tensor(const ov::Tensor &tensor) {
  switch (tensor.get_element_type()) {
  case ov::element::f32:
    return summarize_typed_tensor<float>(tensor);
  case ov::element::f16:
    return summarize_typed_tensor<ov::float16>(tensor);
  case ov::element::i32:
    return summarize_typed_tensor<int32_t>(tensor);
  case ov::element::i64:
    return summarize_typed_tensor<int64_t>(tensor);
  case ov::element::u8:
    return summarize_typed_tensor<uint8_t>(tensor);
  case ov::element::boolean:
    return summarize_typed_tensor<uint8_t>(tensor);
  default:
    throw std::runtime_error("unsupported output type: " +
                             tensor.get_element_type().to_string());
  }
}

std::string shape_to_string(const ov::Shape &shape) {
  std::ostringstream os;
  os << '[';
  for (size_t i = 0; i < shape.size(); ++i) {
    if (i) {
      os << ',';
    }
    os << shape[i];
  }
  os << ']';
  return os.str();
}

size_t broadcast_offset_for_flat_index(size_t flat_index,
                                       const ov::Shape &input_shape,
                                       const ov::Shape &output_shape) {
  const size_t rank = output_shape.size();
  if (input_shape.empty()) {
    return 0;
  }
  std::vector<size_t> input_strides(input_shape.size(), 1);
  for (size_t i = input_shape.size(); i-- > 1;) {
    input_strides[i - 1] = input_strides[i] * input_shape[i];
  }
  size_t input_offset = 0;
  size_t remaining = flat_index;
  for (size_t rev = 0; rev < rank; ++rev) {
    const size_t out_dim = output_shape[rank - 1 - rev];
    const size_t coord = out_dim == 0 ? 0 : remaining % out_dim;
    remaining = out_dim == 0 ? 0 : remaining / out_dim;
    if (rev < input_shape.size()) {
      const size_t input_dim_index = input_shape.size() - 1 - rev;
      if (input_shape[input_dim_index] != 1) {
        input_offset += coord * input_strides[input_dim_index];
      }
    }
  }
  return input_offset;
}

double tensor_value_as_double(const ov::Tensor &tensor, size_t index) {
  switch (tensor.get_element_type()) {
  case ov::element::boolean:
  case ov::element::u8:
    return static_cast<double>(tensor.data<const uint8_t>()[index]);
  case ov::element::f32:
    return static_cast<double>(tensor.data<const float>()[index]);
  case ov::element::f16:
    return static_cast<double>(tensor.data<const ov::float16>()[index]);
  case ov::element::i32:
    return static_cast<double>(tensor.data<const int32_t>()[index]);
  case ov::element::i64:
    return static_cast<double>(tensor.data<const int64_t>()[index]);
  default:
    return 0.0;
  }
}

void print_select_mismatch_probe(
    ov::Core &core, const std::string &ref_device,
    const std::shared_ptr<ov::Node> &node,
    const std::vector<std::pair<ov::Output<const ov::Node>, ov::Tensor>>
        &inputs,
    const DiffStats &stats, const ov::Shape &runtime_output_shape) {
  if (std::string(node->get_type_name()) != "Select" ||
      node->get_input_size() != 3) {
    return;
  }
  const auto &out_shape = runtime_output_shape;
  std::cout << "SELECT_PROBE output_shape=" << shape_to_string(out_shape)
            << " max_index=" << stats.max_index << '\n';
  for (size_t i = 0; i < 3; ++i) {
    auto tensor = evaluate_source_tensor(node->input_value(i), inputs);
    if (!tensor.has_value()) {
      tensor = evaluate_source_tensor_with_reference(
          core, ref_device, node->input_value(i), inputs);
    }
    if (!tensor.has_value()) {
      std::cout << "SELECT_PROBE input[" << i << "] unavailable\n";
      continue;
    }
    const size_t offset = broadcast_offset_for_flat_index(
        stats.max_index, tensor->get_shape(), out_shape);
    std::cout << "SELECT_PROBE input[" << i << "]"
              << " type=" << tensor->get_element_type()
              << " shape=" << shape_to_string(tensor->get_shape())
              << " offset=" << offset << " value=" << std::setprecision(10)
              << tensor_value_as_double(*tensor, offset) << '\n';
    if (tensor->get_element_type() == ov::element::boolean) {
      const auto *data = tensor->data<const uint8_t>();
      const size_t limit = std::min<size_t>(tensor->get_size(), 16);
      std::cout << "SELECT_PROBE input[" << i << "] first_bytes=";
      for (size_t j = 0; j < limit; ++j) {
        if (j) {
          std::cout << ',';
        }
        std::cout << static_cast<unsigned>(data[j]);
      }
      std::cout << '\n';
    }
  }
}

std::vector<int64_t> tinyllama_prompt_ids() {
  return {1,     1724, 338, 4673, 29963, 1177,  29949,
          29973, 673,  297, 697,  3273,  10541, 29889};
}

void fill_tinyllama_prompt_tensor(const std::string &name, ov::Tensor &tensor) {
  if (name == "input_ids") {
    const auto ids = tinyllama_prompt_ids();
    std::copy(ids.begin(), ids.end(), tensor.data<int64_t>());
    return;
  }
  if (name == "attention_mask") {
    std::fill_n(tensor.data<int64_t>(), tensor.get_size(), int64_t{1});
    return;
  }
  if (name == "position_ids") {
    auto *data = tensor.data<int64_t>();
    for (size_t i = 0; i < tensor.get_size(); ++i) {
      data[i] = static_cast<int64_t>(i);
    }
    return;
  }
  if (name == "beam_idx") {
    std::fill_n(tensor.data<int32_t>(), tensor.get_size(), int32_t{0});
    return;
  }
  fill_tensor(tensor, 0);
}

ov::Shape make_input_shape(const ov::Output<const ov::Node> &input,
                           const CompareOptions &options) {
  const std::string name = input.get_any_name();
  if (options.tinyllama_prompt_inputs) {
    if (name == "input_ids" || name == "attention_mask" ||
        name == "position_ids") {
      return {1, tinyllama_prompt_ids().size()};
    }
    if (name == "beam_idx") {
      return {1};
    }
  }
  return make_static_shape(input.get_partial_shape());
}

std::vector<std::pair<ov::Output<const ov::Node>, ov::Tensor>>
make_inputs(const std::shared_ptr<ov::Model> &model,
            const CompareOptions &options) {
  std::vector<std::pair<ov::Output<const ov::Node>, ov::Tensor>> inputs;
  inputs.reserve(model->inputs().size());
  for (const auto &input : model->inputs()) {
    ov::Tensor tensor(input.get_element_type(),
                      make_input_shape(input, options));
    if (options.tinyllama_prompt_inputs) {
      fill_tinyllama_prompt_tensor(input.get_any_name(), tensor);
    } else {
      fill_tensor(tensor, options.input_seed);
    }
    inputs.emplace_back(input, std::move(tensor));
  }
  return inputs;
}

ov::Tensor make_generated_external_tensor(const ov::Output<ov::Node> &source,
                                          uint64_t seed) {
  ov::Tensor tensor(source.get_element_type(),
                    make_static_shape(source.get_partial_shape()));
  fill_tensor(tensor, seed);
  return tensor;
}

uint64_t stable_external_input_seed(const ov::Output<ov::Node> &source,
                                    uint64_t base_seed) {
  const auto node = source.get_node_shared_ptr();
  const std::string name =
      (node ? node->get_friendly_name() : std::string("node")) + ":" +
      std::to_string(source.get_index());
  uint64_t hash = 1469598103934665603ULL;
  for (const unsigned char ch : name) {
    hash ^= static_cast<uint64_t>(ch);
    hash *= 1099511628211ULL;
  }
  return base_seed ^ hash;
}

DiffStats compare_model_outputs(
    ov::CompiledModel &ref_model, ov::CompiledModel &gfx_model,
    const std::vector<std::pair<ov::Output<const ov::Node>, ov::Tensor>>
        &inputs,
    bool print_outputs, std::vector<ov::Tensor> *ref_outputs = nullptr) {
  auto ref_req = make_request(ref_model, inputs);
  auto gfx_req = make_request(gfx_model, inputs);

  ref_req.infer();
  gfx_req.infer();

  DiffStats total;
  if (ref_outputs) {
    ref_outputs->clear();
    ref_outputs->reserve(ref_model.outputs().size());
  }
  for (size_t i = 0; i < ref_model.outputs().size(); ++i) {
    const auto ref_tensor = ref_req.get_tensor(ref_model.output(i));
    const auto gfx_tensor = gfx_req.get_tensor(gfx_model.output(i));
    if (ref_outputs) {
      ref_outputs->push_back(ref_tensor);
    }
    const auto stats = compare_tensors(ref_tensor, gfx_tensor);
    if (stats.max_abs_diff > total.max_abs_diff) {
      total.max_abs_diff = stats.max_abs_diff;
      total.max_index = stats.max_index;
      total.lhs_at_max = stats.lhs_at_max;
      total.rhs_at_max = stats.rhs_at_max;
    }
    total.max_rel_diff = std::max(total.max_rel_diff, stats.max_rel_diff);
    total.elements += stats.elements;
    if (print_outputs) {
      std::cout << "output[" << i << "]"
                << " elements=" << stats.elements
                << " max_abs_diff=" << std::setprecision(10)
                << stats.max_abs_diff << " max_rel_diff=" << stats.max_rel_diff
                << " max_index=" << stats.max_index
                << " ref=" << stats.lhs_at_max << " gfx=" << stats.rhs_at_max
                << '\n';
    }
  }
  return total;
}

std::optional<ov::Tensor>
evaluate_constant_source_tensor(const ov::Output<ov::Node> &source) {
  auto node = source.get_node_shared_ptr();
  if (!node) {
    return std::nullopt;
  }
  if (auto constant =
          std::dynamic_pointer_cast<const ov::op::v0::Constant>(node)) {
    return constant->get_tensor_view();
  }
  if (!node->has_evaluate()) {
    return std::nullopt;
  }

  ov::TensorVector inputs;
  inputs.reserve(node->get_input_size());
  for (const auto &input_value : node->input_values()) {
    auto input_tensor = evaluate_constant_source_tensor(input_value);
    if (!input_tensor.has_value()) {
      return std::nullopt;
    }
    inputs.push_back(*input_tensor);
  }

  ov::TensorVector outputs;
  outputs.reserve(node->get_output_size());
  for (size_t i = 0; i < node->get_output_size(); ++i) {
    if (node->get_output_partial_shape(i).is_dynamic()) {
      return std::nullopt;
    }
    outputs.emplace_back(node->get_output_element_type(i),
                         node->get_output_shape(i));
  }
  if (!node->evaluate(outputs, inputs)) {
    return std::nullopt;
  }
  return outputs.at(source.get_index());
}

std::optional<ov::Tensor> evaluate_source_tensor(
    const ov::Output<ov::Node> &source,
    const std::vector<std::pair<ov::Output<const ov::Node>, ov::Tensor>>
        &inputs) {
  auto node = source.get_node_shared_ptr();
  if (!node) {
    return std::nullopt;
  }

  if (ov::is_type<ov::op::v0::Parameter>(node)) {
    for (const auto &[port, tensor] : inputs) {
      if (port.get_node_shared_ptr() == node &&
          port.get_index() == source.get_index()) {
        return tensor;
      }
    }
    return std::nullopt;
  }

  if (auto constant =
          std::dynamic_pointer_cast<const ov::op::v0::Constant>(node)) {
    return constant->get_tensor_view();
  }

  if (!node->has_evaluate()) {
    return std::nullopt;
  }

  ov::TensorVector eval_inputs;
  eval_inputs.reserve(node->get_input_size());
  for (const auto &input_value : node->input_values()) {
    auto input_tensor = evaluate_source_tensor(input_value, inputs);
    if (!input_tensor.has_value()) {
      return std::nullopt;
    }
    eval_inputs.push_back(*input_tensor);
  }

  ov::TensorVector outputs;
  outputs.reserve(node->get_output_size());
  for (size_t i = 0; i < node->get_output_size(); ++i) {
    if (node->get_output_partial_shape(i).is_dynamic()) {
      return std::nullopt;
    }
    outputs.emplace_back(node->get_output_element_type(i),
                         node->get_output_shape(i));
  }
  if (!node->evaluate(outputs, eval_inputs)) {
    return std::nullopt;
  }
  return outputs.at(source.get_index());
}

std::optional<ov::Tensor> evaluate_source_tensor_with_compiled_submodel(
    ov::Core &core, const std::string &device, bool for_gfx,
    const ov::Output<ov::Node> &source,
    const std::vector<std::pair<ov::Output<const ov::Node>, ov::Tensor>>
        &inputs,
    const CompareOptions *options) {
  ov::ParameterVector params;
  params.reserve(inputs.size());
  for (const auto &[port, _] : inputs) {
    auto param_const = std::dynamic_pointer_cast<const ov::op::v0::Parameter>(
        port.get_node_shared_ptr());
    if (!param_const) {
      return std::nullopt;
    }
    params.push_back(
        std::const_pointer_cast<ov::op::v0::Parameter>(param_const));
  }

  ov::OutputVector outputs;
  outputs.push_back(std::make_shared<ov::op::v0::Result>(source));
  auto source_node = source.get_node_shared_ptr();
  const std::string submodel_name =
      source_node ? (source_node->get_friendly_name() + "/out_" +
                     std::to_string(source.get_index()))
                  : std::string("ref_source");
  auto submodel = std::make_shared<ov::Model>(outputs, params, submodel_name);
  auto compiled = core.compile_model(submodel, device,
                                     make_compile_config(for_gfx, options));
  auto request = make_request(compiled, inputs);
  request.infer();
  return clone_tensor_data(request.get_tensor(compiled.output(0)));
}

std::optional<ov::Tensor> evaluate_source_tensor_with_reference(
    ov::Core &core, const std::string &ref_device,
    const ov::Output<ov::Node> &source,
    const std::vector<std::pair<ov::Output<const ov::Node>, ov::Tensor>>
        &inputs) {
  return evaluate_source_tensor_with_compiled_submodel(core, ref_device, false,
                                                       source, inputs);
}

std::vector<int64_t> tensor_to_i64_vector(const ov::Tensor &tensor) {
  std::vector<int64_t> values;
  values.reserve(tensor.get_size());
  if (tensor.get_element_type() == ov::element::i64) {
    const auto *data = tensor.data<const int64_t>();
    values.assign(data, data + tensor.get_size());
    return values;
  }
  if (tensor.get_element_type() == ov::element::i32) {
    const auto *data = tensor.data<const int32_t>();
    for (size_t i = 0; i < tensor.get_size(); ++i) {
      values.push_back(static_cast<int64_t>(data[i]));
    }
    return values;
  }
  throw std::runtime_error("expected i32/i64 tensor, got " +
                           tensor.get_element_type().to_string());
}

size_t shape_product_range(const ov::Shape &shape, size_t begin, size_t end) {
  size_t product = 1;
  for (size_t i = begin; i < end; ++i) {
    product *= shape[i];
  }
  return product;
}

bool materialize_split_outputs_on_host(
    const std::shared_ptr<ov::Node> &node, const ov::Tensor &data,
    const ov::Tensor &axis_tensor, const ov::Tensor *split_lengths_tensor,
    std::unordered_map<OutputKey, ov::Tensor, OutputKeyHash> &cache) {
  if (!node || (node->get_type_name() != std::string("Split") &&
                node->get_type_name() != std::string("VariadicSplit"))) {
    return false;
  }
  ov::Shape data_shape = data.get_shape();
  if (node->get_input_partial_shape(0).is_static()) {
    data_shape = node->get_input_shape(0);
  }
  if (data_shape.empty()) {
    return false;
  }
  const auto axis_values = tensor_to_i64_vector(axis_tensor);
  if (axis_values.size() != 1) {
    return false;
  }
  int64_t axis = axis_values.front();
  const int64_t rank = static_cast<int64_t>(data_shape.size());
  if (axis < 0) {
    axis += rank;
  }
  if (axis < 0 || axis >= rank) {
    return false;
  }
  const auto axis_idx = static_cast<size_t>(axis);
  std::vector<size_t> split_sizes;
  if (auto split = ov::as_type_ptr<ov::op::v1::Split>(node)) {
    const size_t parts = split->get_num_splits();
    if (parts == 0 || data_shape[axis_idx] % parts != 0) {
      return false;
    }
    split_sizes.assign(parts, data_shape[axis_idx] / parts);
  } else {
    bool static_outputs = node->get_output_size() > 0;
    std::vector<size_t> static_output_sizes;
    static_output_sizes.reserve(node->get_output_size());
    size_t static_sum = 0;
    for (size_t out_idx = 0; out_idx < node->get_output_size(); ++out_idx) {
      if (!node->get_output_partial_shape(out_idx).is_static()) {
        static_outputs = false;
        break;
      }
      const auto out_shape = node->get_output_shape(out_idx);
      if (out_shape.size() != data_shape.size()) {
        static_outputs = false;
        break;
      }
      static_output_sizes.push_back(out_shape[axis_idx]);
      static_sum += out_shape[axis_idx];
    }
    if (static_outputs && static_sum == data_shape[axis_idx]) {
      split_sizes = std::move(static_output_sizes);
    } else {
      if (!split_lengths_tensor) {
        return false;
      }
      const auto lengths = tensor_to_i64_vector(*split_lengths_tensor);
      split_sizes.reserve(lengths.size());
      int64_t infer_index = -1;
      size_t known_sum = 0;
      for (size_t i = 0; i < lengths.size(); ++i) {
        const int64_t length = lengths[i];
        if (length < 0) {
          if (length != -1 || infer_index >= 0) {
            return false;
          }
          infer_index = static_cast<int64_t>(i);
          split_sizes.push_back(0);
        } else {
          known_sum += static_cast<size_t>(length);
          split_sizes.push_back(static_cast<size_t>(length));
        }
      }
      if (infer_index >= 0) {
        if (known_sum > data_shape[axis_idx]) {
          return false;
        }
        split_sizes[static_cast<size_t>(infer_index)] =
            data_shape[axis_idx] - known_sum;
      }
    }
  }
  if (split_sizes.size() != node->get_output_size()) {
    return false;
  }
  const size_t outer = shape_product_range(data_shape, 0, axis_idx);
  const size_t inner =
      shape_product_range(data_shape, axis_idx + 1, data_shape.size());
  const size_t elem_size = data.get_element_type().size();
  const auto *src = static_cast<const uint8_t *>(data.data());
  size_t axis_offset = 0;
  for (size_t out_idx = 0; out_idx < split_sizes.size(); ++out_idx) {
    ov::Shape out_shape = data_shape;
    out_shape[axis_idx] = split_sizes[out_idx];
    ov::Tensor out(data.get_element_type(), out_shape);
    auto *dst = static_cast<uint8_t *>(out.data());
    const size_t region_bytes = split_sizes[out_idx] * inner * elem_size;
    for (size_t outer_idx = 0; outer_idx < outer; ++outer_idx) {
      const size_t src_offset =
          (outer_idx * data_shape[axis_idx] + axis_offset) * inner * elem_size;
      const size_t dst_offset = outer_idx * region_bytes;
      if (region_bytes > 0) {
        std::memcpy(dst + dst_offset, src + src_offset, region_bytes);
      }
    }
    cache.insert_or_assign({node.get(), out_idx}, out);
    axis_offset += split_sizes[out_idx];
  }
  return true;
}

std::optional<ov::Tensor>
materialize_shapeof_output_on_host(const std::shared_ptr<ov::Node> &node,
                                   const ov::Output<ov::Node> &source) {
  // Compare-runner only: prepare isolated per-op inputs from static shape
  // metadata. This must not become a production GFX inference fallback path.
  if (!ov::is_type<ov::op::v0::ShapeOf>(node) &&
      !ov::is_type<ov::op::v3::ShapeOf>(node)) {
    return std::nullopt;
  }
  if (source.get_index() != 0 || node->get_input_size() == 0) {
    return std::nullopt;
  }
  const auto input_pshape = node->get_input_partial_shape(0);
  if (!input_pshape.is_static()) {
    return std::nullopt;
  }
  const auto input_shape = input_pshape.to_shape();
  const auto output_type = node->get_output_element_type(0);
  ov::Tensor tensor(output_type, ov::Shape{input_shape.size()});
  if (output_type == ov::element::i64) {
    auto *data = tensor.data<int64_t>();
    for (size_t i = 0; i < input_shape.size(); ++i) {
      data[i] = static_cast<int64_t>(input_shape[i]);
    }
    return tensor;
  }
  if (output_type == ov::element::i32) {
    auto *data = tensor.data<int32_t>();
    for (size_t i = 0; i < input_shape.size(); ++i) {
      data[i] = static_cast<int32_t>(input_shape[i]);
    }
    return tensor;
  }
  return std::nullopt;
}

bool materialize_1d_gather_output_on_host(
    const std::shared_ptr<ov::Node> &node, const ov::Tensor &data,
    const ov::Tensor &indices, const ov::Tensor &axis,
    std::unordered_map<OutputKey, ov::Tensor, OutputKeyHash> &cache) {
  // Compare-runner only: resolve 1D shape-list Gather while recursively
  // preparing the inputs for one isolated GFX op. Production inference must
  // execute on GPU.
  auto gather_v1 = ov::as_type_ptr<ov::op::v1::Gather>(node);
  auto gather_v7 = ov::as_type_ptr<ov::op::v7::Gather>(node);
  auto gather_v8 = ov::as_type_ptr<ov::op::v8::Gather>(node);
  if (!gather_v1 && !gather_v7 && !gather_v8) {
    return false;
  }
  if ((gather_v7 && gather_v7->get_batch_dims() != 0) ||
      (gather_v8 && gather_v8->get_batch_dims() != 0)) {
    return false;
  }
  const auto data_shape = data.get_shape();
  if (data_shape.size() != 1 || node->get_output_size() != 1) {
    return false;
  }
  const auto axis_values = tensor_to_i64_vector(axis);
  if (axis_values.size() != 1) {
    return false;
  }
  int64_t axis_value = axis_values.front();
  if (axis_value < 0) {
    axis_value += static_cast<int64_t>(data_shape.size());
  }
  if (axis_value != 0) {
    return false;
  }

  const auto index_values = tensor_to_i64_vector(indices);
  ov::Shape out_shape = indices.get_shape();
  if (node->get_output_partial_shape(0).is_static()) {
    out_shape = node->get_output_shape(0);
  }
  if (ov::shape_size(out_shape) != index_values.size()) {
    return false;
  }

  ov::Tensor out(data.get_element_type(), out_shape);
  const size_t elem_size = data.get_element_type().size();
  const auto *src = static_cast<const uint8_t *>(data.data());
  auto *dst = static_cast<uint8_t *>(out.data());
  for (size_t i = 0; i < index_values.size(); ++i) {
    int64_t idx = index_values[i];
    const int64_t axis_dim = static_cast<int64_t>(data_shape[0]);
    if (idx < 0) {
      idx += axis_dim;
    }
    idx = std::clamp<int64_t>(idx, 0, axis_dim - 1);
    std::memcpy(dst + i * elem_size, src + static_cast<size_t>(idx) * elem_size,
                elem_size);
  }
  cache.insert_or_assign({node.get(), 0}, out);
  return true;
}

std::optional<ov::Tensor> evaluate_source_tensor_with_gfx_recursive(
    ov::Core &core, const ov::Output<ov::Node> &source,
    const std::vector<std::pair<ov::Output<const ov::Node>, ov::Tensor>>
        &inputs,
    std::unordered_map<OutputKey, ov::Tensor, OutputKeyHash> &cache,
    RecursiveMaterializationState &state, const CompareOptions *options,
    size_t depth = 0) {
  if (depth > 4096) {
    return std::nullopt;
  }
  auto node = source.get_node_shared_ptr();
  if (!node) {
    return std::nullopt;
  }

  OutputKey key{source.get_node(), source.get_index()};
  if (auto it = cache.find(key); it != cache.end()) {
    return it->second;
  }

  if (auto tensor = evaluate_source_tensor(source, inputs)) {
    cache.insert_or_assign(key, *tensor);
    return tensor;
  }

  const std::string node_desc =
      node->get_friendly_name() + " (" + node->get_type_name() + ")";
  if (auto shapeof_tensor = materialize_shapeof_output_on_host(node, source)) {
    cache.insert_or_assign(key, *shapeof_tensor);
    if (state.trace_every > 0) {
      std::cout << "GFX_RECURSIVE host_shapeof node=" << node_desc
                << " output=" << source.get_index()
                << " shape=" << shape_to_string(shapeof_tensor->get_shape())
                << '\n';
    }
    return shapeof_tensor;
  }
  constexpr size_t kGraphSliceProducerLimit = 96;
  if (depth == 0 &&
      count_upstream_ops_limited(source, kGraphSliceProducerLimit) <=
          kGraphSliceProducerLimit) {
    try {
      auto tensor = evaluate_source_tensor_with_compiled_submodel(
          core, "GFX", true, source, inputs, options);
      if (tensor.has_value()) {
        cache.insert_or_assign(key, *tensor);
        if (state.trace_every > 0) {
          std::cout << "GFX_RECURSIVE graph_slice node=" << node_desc
                    << " output=" << source.get_index()
                    << " shape=" << shape_to_string(tensor->get_shape())
                    << '\n';
        }
        return tensor;
      }
    } catch (const std::exception &ex) {
      if (state.trace_every > 0) {
        std::cout << "GFX_RECURSIVE graph_slice_fallback node=" << node_desc
                  << " reason=" << ex.what() << '\n';
      }
    }
  } else if (depth == 0 && state.trace_every > 0) {
    std::cout << "GFX_RECURSIVE graph_slice_skipped node=" << node_desc
              << " reason=upstream producer count exceeds "
              << kGraphSliceProducerLimit << '\n';
  }
  if (state.limit > 0 && state.materialized >= state.limit) {
    state.failure = "gfx-recursive materialization limit reached before " +
                    node_desc + " after " + std::to_string(state.materialized) +
                    " producers";
    return std::nullopt;
  }
  ++state.materialized;
  state.last_node = node_desc;
  if (state.trace_every > 0 && (state.materialized == 1 ||
                                state.materialized % state.trace_every == 0)) {
    std::cout << "GFX_RECURSIVE materialized=" << state.materialized
              << " depth=" << depth << " cache=" << cache.size()
              << " node=" << node_desc << '\n';
  }

  ov::ParameterVector params;
  ov::OutputVector cloned_inputs;
  ov::TensorVector eval_inputs;
  std::vector<std::pair<ov::Output<const ov::Node>, ov::Tensor>>
      isolated_inputs;
  params.reserve(node->get_input_size());
  cloned_inputs.reserve(node->get_input_size());
  eval_inputs.reserve(node->get_input_size());
  isolated_inputs.reserve(node->get_input_size());
  for (size_t i = 0; i < node->get_input_size(); ++i) {
    if (auto const_tensor =
            evaluate_constant_source_tensor(node->input_value(i))) {
      auto constant = std::make_shared<ov::op::v0::Constant>(
          const_tensor->get_element_type(), const_tensor->get_shape(),
          const_tensor->data());
      constant->set_friendly_name(node->get_friendly_name() +
                                  "/gfx_upstream_const_" + std::to_string(i));
      cloned_inputs.push_back(constant->output(0));
      eval_inputs.push_back(*const_tensor);
      cache.insert_or_assign(
          {node->input_value(i).get_node(), node->input_value(i).get_index()},
          *const_tensor);
      continue;
    }
    auto input_tensor = evaluate_source_tensor_with_gfx_recursive(
        core, node->input_value(i), inputs, cache, state, options, depth + 1);
    if (!input_tensor.has_value()) {
      if (state.failure.empty()) {
        state.failure = "gfx-recursive failed to materialize input " +
                        std::to_string(i) + " for " + node_desc;
      }
      return std::nullopt;
    }
    auto param = std::make_shared<ov::op::v0::Parameter>(
        input_tensor->get_element_type(), input_tensor->get_shape());
    param->set_friendly_name(node->get_friendly_name() +
                             "/gfx_upstream_input_" + std::to_string(i));
    cloned_inputs.push_back(param->output(0));
    eval_inputs.push_back(*input_tensor);
    isolated_inputs.emplace_back(param->output(0), *input_tensor);
    params.push_back(param);
  }

  if (node->get_type_name() == std::string("Split") ||
      node->get_type_name() == std::string("VariadicSplit")) {
    const ov::Tensor *split_lengths =
        eval_inputs.size() > 2 ? &eval_inputs[2] : nullptr;
    if (eval_inputs.size() >= 2 &&
        materialize_split_outputs_on_host(node, eval_inputs[0], eval_inputs[1],
                                          split_lengths, cache)) {
      if (state.trace_every > 0) {
        std::cout << "GFX_RECURSIVE host_split node=" << node_desc
                  << " output=" << source.get_index()
                  << " shape=" << cache.at(key).get_shape() << '\n';
      }
      if (auto it = cache.find(key); it != cache.end()) {
        return it->second;
      }
    }
    state.failure = "gfx-recursive failed to host-materialize " + node_desc;
    return std::nullopt;
  }

  if (node->get_type_name() == std::string("Gather")) {
    if (eval_inputs.size() >= 3 &&
        materialize_1d_gather_output_on_host(
            node, eval_inputs[0], eval_inputs[1], eval_inputs[2], cache)) {
      if (state.trace_every > 0) {
        std::cout << "GFX_RECURSIVE host_gather node=" << node_desc
                  << " output=" << source.get_index()
                  << " shape=" << cache.at(key).get_shape() << '\n';
      }
      if (auto it = cache.find(key); it != cache.end()) {
        return it->second;
      }
    }
  }

  auto cloned = node->clone_with_new_inputs(cloned_inputs);
  ov::OutputVector results;
  results.reserve(cloned->get_output_size());
  for (const auto &output : cloned->outputs()) {
    results.push_back(std::make_shared<ov::op::v0::Result>(output));
  }
  if (results.empty()) {
    return std::nullopt;
  }
  auto submodel = std::make_shared<ov::Model>(
      results, params, node->get_friendly_name() + "_gfx_upstream");
  ov::CompiledModel compiled;
  try {
    compiled =
        core.compile_model(submodel, "GFX", make_compile_config(true, options));
  } catch (const std::exception &ex) {
    state.failure =
        "gfx-recursive compile failed at " + node_desc + ": " + ex.what();
    return std::nullopt;
  }
  auto request = make_request(compiled, isolated_inputs);
  try {
    request.infer();
  } catch (const std::exception &ex) {
    state.failure =
        "gfx-recursive infer failed at " + node_desc + ": " + ex.what();
    return std::nullopt;
  }

  for (size_t out_idx = 0; out_idx < cloned->get_output_size(); ++out_idx) {
    cache.insert_or_assign(
        {node.get(), out_idx},
        clone_tensor_data(request.get_tensor(compiled.output(out_idx))));
  }
  if (auto it = cache.find(key); it != cache.end()) {
    return it->second;
  }
  return std::nullopt;
}

void print_conv_probe(
    ov::Core &core, const std::string &ref_device,
    const std::shared_ptr<ov::Node> &node,
    const std::vector<std::pair<ov::Output<const ov::Node>, ov::Tensor>>
        &inputs,
    const DiffStats &stats) {
  auto conv = std::dynamic_pointer_cast<ov::op::v1::Convolution>(node);
  if (!conv || stats.elements == 0) {
    return;
  }
  if (conv->get_input_size() != 2 || conv->get_output_shape(0).size() != 4 ||
      inputs.empty()) {
    return;
  }
  auto input_tensor = evaluate_source_tensor(conv->input_value(0), inputs);
  auto weights_tensor = evaluate_source_tensor(conv->input_value(1), inputs);
  if (!input_tensor.has_value()) {
    input_tensor = evaluate_source_tensor_with_reference(
        core, ref_device, conv->input_value(0), inputs);
  }
  if (!weights_tensor.has_value()) {
    weights_tensor = evaluate_source_tensor_with_reference(
        core, ref_device, conv->input_value(1), inputs);
  }
  if (!input_tensor.has_value() || !weights_tensor.has_value()) {
    return;
  }
  if (input_tensor->get_element_type() != ov::element::f32 ||
      weights_tensor->get_element_type() != ov::element::f32) {
    return;
  }
  if (input_tensor->get_shape() != conv->get_input_shape(0) ||
      weights_tensor->get_shape() != conv->get_input_shape(1)) {
    return;
  }

  const auto *input_ptr = input_tensor->data<const float>();
  const auto *weight_ptr = weights_tensor->data<const float>();
  const auto &out_shape = conv->get_output_shape(0);
  const size_t spatial = out_shape[2] * out_shape[3];
  const size_t channel_stride = spatial;
  const size_t batch_stride = out_shape[1] * channel_stride;
  const size_t n = stats.max_index / batch_stride;
  const size_t rem0 = stats.max_index % batch_stride;
  const size_t oc = rem0 / channel_stride;
  const size_t rem1 = rem0 % channel_stride;
  const size_t oh = rem1 / out_shape[3];
  const size_t ow = rem1 % out_shape[3];

  const auto &in_shape = conv->get_input_shape(0);
  const auto &w_shape = conv->get_input_shape(1);
  const auto &strides = conv->get_strides();
  const auto &dilations = conv->get_dilations();
  const auto &pads_begin = conv->get_pads_begin();

  auto f16_round = [](float v) -> float {
    return static_cast<float>(ov::float16(v));
  };

  float acc_f16 = 0.0f;
  float acc_f16_accum = 0.0f;
  float acc_f32 = 0.0f;
  float acc_hwio = 0.0f;
  for (size_t ic = 0; ic < in_shape[1]; ++ic) {
    for (size_t kh = 0; kh < w_shape[2]; ++kh) {
      const int64_t ih = static_cast<int64_t>(oh * strides[0]) -
                         static_cast<int64_t>(pads_begin[0]) +
                         static_cast<int64_t>(kh * dilations[0]);
      if (ih < 0 || ih >= static_cast<int64_t>(in_shape[2])) {
        continue;
      }
      for (size_t kw = 0; kw < w_shape[3]; ++kw) {
        const int64_t iw = static_cast<int64_t>(ow * strides[1]) -
                           static_cast<int64_t>(pads_begin[1]) +
                           static_cast<int64_t>(kw * dilations[1]);
        if (iw < 0 || iw >= static_cast<int64_t>(in_shape[3])) {
          continue;
        }
        const size_t in_idx =
            ((n * in_shape[1] + ic) * in_shape[2] + static_cast<size_t>(ih)) *
                in_shape[3] +
            static_cast<size_t>(iw);
        const size_t w_idx =
            ((oc * w_shape[1] + ic) * w_shape[2] + kh) * w_shape[3] + kw;
        const size_t w_idx_hwio =
            ((kh * w_shape[3] + kw) * w_shape[1] + ic) * w_shape[0] + oc;
        const float in_v = input_ptr[in_idx];
        const float w_v = weight_ptr[w_idx];
        const float prod_f16 = f16_round(in_v) * f16_round(w_v);
        acc_f16 += prod_f16;
        acc_f16_accum = f16_round(acc_f16_accum + prod_f16);
        acc_f32 += in_v * w_v;
        acc_hwio += in_v * weight_ptr[w_idx_hwio];
      }
    }
  }

  std::cout << "CONV_PROBE n=" << n << " oc=" << oc << " oh=" << oh
            << " ow=" << ow << " naive_f16=" << std::setprecision(10) << acc_f16
            << " naive_f16_accum=" << acc_f16_accum << " naive_f32=" << acc_f32
            << " naive_hwio=" << acc_hwio << '\n';
}

struct FullGraphOutputDesc {
  size_t op_index = 0;
  std::string name;
  std::string type;
};

std::shared_ptr<ov::Model>
make_full_graph_debug_model(const std::shared_ptr<ov::Model> &model,
                            std::vector<FullGraphOutputDesc> &output_descs) {
  auto cloned_model = model->clone();
  const auto cloned_ops = cloned_model->get_ordered_ops();
  ov::ResultVector results;
  output_descs.clear();
  for (size_t idx = 0; idx < cloned_ops.size(); ++idx) {
    const auto &node = cloned_ops[idx];
    if (is_debug_skippable_node(node)) {
      continue;
    }
    for (size_t port = 0; port < node->get_output_size(); ++port) {
      results.push_back(
          std::make_shared<ov::op::v0::Result>(node->output(port)));
      FullGraphOutputDesc desc;
      desc.op_index = idx;
      desc.type = node->get_type_name();
      desc.name = node->get_friendly_name();
      if (node->get_output_size() > 1) {
        desc.name += "/out_" + std::to_string(port);
      }
      output_descs.push_back(std::move(desc));
    }
  }
  return std::make_shared<ov::Model>(results, cloned_model->get_parameters(),
                                     cloned_model->get_friendly_name() +
                                         "_per_op");
}

int run_full_graph_per_op_compare(ov::Core &core,
                                  const std::shared_ptr<ov::Model> &model,
                                  const CompareOptions &options) {
  const auto ref_device = reference_device(core, options.reference_device);
  std::vector<FullGraphOutputDesc> output_descs;
  auto debug_model = make_full_graph_debug_model(model, output_descs);
  const auto inputs = make_inputs(debug_model, options);

  auto ref_model =
      core.compile_model(debug_model, ref_device, make_compile_config(false));
  auto gfx_model = core.compile_model(debug_model, "GFX",
                                      make_compile_config(true, &options));
  auto ref_req = make_request(ref_model, inputs);
  auto gfx_req = make_request(gfx_model, inputs);

  ref_req.infer();
  gfx_req.infer();

  size_t mismatch_count = 0;
  double global_max_abs = 0.0;
  double global_max_rel = 0.0;
  std::optional<size_t> first_mismatch_output;
  DiffStats first_mismatch_stats;
  for (size_t i = 0; i < output_descs.size(); ++i) {
    const auto ref_tensor = ref_req.get_tensor(ref_model.output(i));
    const auto gfx_tensor = gfx_req.get_tensor(gfx_model.output(i));
    const auto stats = compare_tensors(ref_tensor, gfx_tensor);
    const auto &desc = output_descs[i];
    global_max_abs = std::max(global_max_abs, stats.max_abs_diff);
    global_max_rel = std::max(global_max_rel, stats.max_rel_diff);
    const bool mismatch = stats.max_abs_diff > options.abs_threshold &&
                          stats.max_rel_diff > options.rel_threshold;
    if (mismatch) {
      ++mismatch_count;
      if (!first_mismatch_output.has_value()) {
        first_mismatch_output = i;
        first_mismatch_stats = stats;
      }
    }
    std::cout << "[op " << desc.op_index << "] " << desc.name << " ("
              << desc.type << ")"
              << " max_abs_diff=" << std::setprecision(10) << stats.max_abs_diff
              << " max_rel_diff=" << stats.max_rel_diff << '\n';
  }
  if (first_mismatch_output.has_value()) {
    const auto &desc = output_descs[*first_mismatch_output];
    std::cout << "PER_OP_FIRST_MISMATCH"
              << " op_index=" << desc.op_index
              << " output_index=" << *first_mismatch_output
              << " name=" << desc.name << " type=" << desc.type
              << " max_index=" << first_mismatch_stats.max_index
              << " ref=" << std::setprecision(10)
              << first_mismatch_stats.lhs_at_max
              << " gfx=" << first_mismatch_stats.rhs_at_max << '\n';
    std::cout << "PER_OP_MISMATCH count=" << mismatch_count
              << " max_abs=" << std::setprecision(10) << global_max_abs
              << " max_rel=" << global_max_rel << '\n';
    return 3;
  }
  std::cout << "PER_OP_MATCH max_abs=" << std::setprecision(10)
            << global_max_abs << " max_rel=" << global_max_rel << '\n';
  return 0;
}

int run_per_op_compare(
    ov::Core &core, const std::shared_ptr<ov::Model> &model,
    const std::vector<std::pair<ov::Output<const ov::Node>, ov::Tensor>>
        &inputs,
    const CompareOptions &options) {
  const auto ordered_ops = model->get_ordered_ops();
  const auto ref_device = reference_device(core, options.reference_device);
  std::vector<size_t> relevant_indices;
  relevant_indices.reserve(ordered_ops.size());
  std::unordered_map<const ov::Node *, size_t> relevant_pos;
  for (size_t idx = 0; idx < ordered_ops.size(); ++idx) {
    const auto &node = ordered_ops[idx];
    if (is_debug_skippable_node(node)) {
      continue;
    }
    relevant_pos.emplace(node.get(), relevant_indices.size());
    relevant_indices.push_back(idx);
  }
  size_t checked = 0;
  size_t skipped = 0;
  std::unordered_map<OutputKey, ov::Tensor, OutputKeyHash>
      cached_reference_outputs;
  for (size_t idx = 0; idx < ordered_ops.size(); ++idx) {
    const auto &node = ordered_ops[idx];
    if (idx < options.start_op) {
      continue;
    }
    if (is_debug_skippable_node(node)) {
      continue;
    }
    if (options.stop_after_op.has_value() &&
        checked >= *options.stop_after_op) {
      break;
    }

    const auto pos_it = relevant_pos.find(node.get());
    OPENVINO_ASSERT(pos_it != relevant_pos.end(),
                    "per-op debug: node position missing");
    const size_t pos = pos_it->second;
    const size_t window_begin_pos =
        (options.window_size > 0 && pos + 1 > options.window_size)
            ? (pos + 1 - options.window_size)
            : 0;

    ov::ParameterVector isolated_params;
    std::vector<std::pair<ov::Output<const ov::Node>, ov::Tensor>>
        isolated_input_tensors;
    std::unordered_map<OutputKey, ov::Output<ov::Node>, OutputKeyHash>
        cloned_outputs;
    std::unordered_map<OutputKey, ov::Output<ov::Node>, OutputKeyHash>
        external_values;
    RecursiveMaterializationState recursive_state{
        options.per_op_recursive_limit,
        options.per_op_recursive_trace_every,
        0,
        {},
        {}};
    std::shared_ptr<ov::Node> cloned_target;

    auto materialize_external = [&](const ov::Output<ov::Node> &source)
        -> std::optional<ov::Output<ov::Node>> {
      OutputKey key{source.get_node(), source.get_index()};
      if (auto it = external_values.find(key); it != external_values.end()) {
        return it->second;
      }
      if (auto const_tensor = evaluate_constant_source_tensor(source)) {
        auto constant = std::make_shared<ov::op::v0::Constant>(
            const_tensor->get_element_type(), const_tensor->get_shape(),
            const_tensor->data());
        constant->set_friendly_name(
            (source.get_node_shared_ptr()
                 ? source.get_node_shared_ptr()->get_friendly_name()
                 : std::string("const")) +
            "/isolated_const_" + std::to_string(source.get_index()));
        auto out = constant->output(0);
        external_values.emplace(key, out);
        return out;
      }
      auto cached_it = cached_reference_outputs.find(key);
      std::optional<ov::Tensor> tensor;
      if (cached_it != cached_reference_outputs.end()) {
        tensor = cached_it->second;
      } else if (options.per_op_input_mode == PerOpInputMode::Generated) {
        tensor = make_generated_external_tensor(
            source, stable_external_input_seed(source, options.input_seed));
      } else {
        tensor = evaluate_source_tensor(source, inputs);
      }
      if (!tensor.has_value() &&
          options.per_op_input_mode == PerOpInputMode::Reference) {
        tensor = evaluate_source_tensor_with_reference(core, ref_device, source,
                                                       inputs);
      }
      if (!tensor.has_value() &&
          options.per_op_input_mode == PerOpInputMode::GfxRecursive) {
        tensor = evaluate_source_tensor_with_gfx_recursive(
            core, source, inputs, cached_reference_outputs, recursive_state,
            &options);
      }
      if (!tensor.has_value()) {
        return std::nullopt;
      }
      cached_reference_outputs.insert_or_assign(key, *tensor);
      auto param = std::make_shared<ov::op::v0::Parameter>(
          tensor->get_element_type(), tensor->get_shape());
      param->set_friendly_name(
          (source.get_node_shared_ptr()
               ? source.get_node_shared_ptr()->get_friendly_name()
               : std::string("param")) +
          "/isolated_input_" + std::to_string(source.get_index()));
      isolated_params.push_back(param);
      isolated_input_tensors.emplace_back(param->output(0), *tensor);
      auto out = param->output(0);
      external_values.emplace(key, out);
      return out;
    };

    bool missing_input = false;
    for (size_t p = window_begin_pos; p <= pos; ++p) {
      const auto &stage_node = ordered_ops[relevant_indices[p]];
      ov::OutputVector cloned_inputs;
      cloned_inputs.reserve(stage_node->get_input_size());
      for (const auto &source : stage_node->input_values()) {
        OutputKey key{source.get_node(), source.get_index()};
        if (auto it = cloned_outputs.find(key); it != cloned_outputs.end()) {
          cloned_inputs.push_back(it->second);
          continue;
        }
        auto ext = materialize_external(source);
        if (!ext.has_value()) {
          std::cout << "[op " << idx << "] " << node->get_friendly_name()
                    << " (" << node->get_type_name()
                    << ") infer_skip=failed to materialize isolated input from "
                    << stage_node->get_friendly_name();
          if (!recursive_state.failure.empty()) {
            std::cout << " reason=" << recursive_state.failure;
          } else if (!recursive_state.last_node.empty()) {
            std::cout << " last_recursive_node=" << recursive_state.last_node;
          }
          std::cout << '\n';
          missing_input = true;
          break;
        }
        cloned_inputs.push_back(*ext);
      }
      if (missing_input) {
        break;
      }
      auto cloned = stage_node->clone_with_new_inputs(cloned_inputs);
      for (size_t out_idx = 0; out_idx < cloned->get_output_size(); ++out_idx) {
        cloned_outputs[{stage_node.get(), out_idx}] = cloned->output(out_idx);
      }
      cloned_target = cloned;
    }
    if (missing_input) {
      ++skipped;
      ++checked;
      continue;
    }
    OPENVINO_ASSERT(cloned_target, "per-op debug: isolated target is null");
    ov::OutputVector outputs;
    outputs.reserve(cloned_target->get_output_size());
    for (const auto &output : cloned_target->outputs()) {
      outputs.push_back(std::make_shared<ov::op::v0::Result>(output));
    }
    auto submodel = std::make_shared<ov::Model>(outputs, isolated_params,
                                                node->get_friendly_name());

    ov::CompiledModel ref_submodel;
    ov::CompiledModel gfx_submodel;
    try {
      ref_submodel =
          core.compile_model(submodel, ref_device, make_compile_config(false));
      gfx_submodel = core.compile_model(submodel, "GFX",
                                        make_compile_config(true, &options));
    } catch (const std::exception &ex) {
      std::cout << "[op " << idx << "] " << node->get_friendly_name() << " ("
                << node->get_type_name() << ") compile_skip=" << ex.what()
                << '\n';
      ++skipped;
      ++checked;
      continue;
    }

    DiffStats stats;
    std::vector<ov::Tensor> ref_outputs;
    try {
      stats =
          compare_model_outputs(ref_submodel, gfx_submodel,
                                isolated_input_tensors, false, &ref_outputs);
    } catch (const std::exception &ex) {
      std::cout << "[op " << idx << "] " << node->get_friendly_name() << " ("
                << node->get_type_name() << ") infer_skip=" << ex.what()
                << '\n';
      ++skipped;
      ++checked;
      continue;
    }
    for (size_t out_idx = 0; out_idx < ref_outputs.size(); ++out_idx) {
      cached_reference_outputs.insert_or_assign({node.get(), out_idx},
                                                ref_outputs[out_idx]);
    }

    std::cout << "[op " << idx << "] " << node->get_friendly_name() << " ("
              << node->get_type_name()
              << ") max_abs_diff=" << std::setprecision(10)
              << stats.max_abs_diff << " max_rel_diff=" << stats.max_rel_diff;
    std::cout << '\n';
    if (!options.per_op_all && stats.max_abs_diff > options.abs_threshold &&
        stats.max_rel_diff > options.rel_threshold) {
      print_conv_probe(core, ref_device, node, inputs, stats);
      if (!ref_outputs.empty()) {
        print_select_mismatch_probe(core, ref_device, node, inputs, stats,
                                    ref_outputs.front().get_shape());
      }
      std::cout << "FIRST_MISMATCH op_index=" << idx
                << " name=" << node->get_friendly_name()
                << " type=" << node->get_type_name()
                << " max_index=" << stats.max_index
                << " ref=" << std::setprecision(10) << stats.lhs_at_max
                << " gfx=" << stats.rhs_at_max << '\n';
      return 3;
    }
    ++checked;
  }
  if (skipped > 0) {
    std::cout << "PER_OP_SKIPPED count=" << skipped << '\n';
    return 4;
  }
  std::cout << "PER_OP_MATCH\n";
  return 0;
}

} // namespace

int main(int argc, char **argv) try {
  if (argc < 2) {
    std::cerr
        << "usage: ov_gfx_compare_runner <model.xml> [--per-op] [--per-op-all] "
           "[--print-ops] [--gfx-only] "
           "[--reference-device NAME] [--reference-plugin PATH] "
           "[--start-op N] "
           "[--window-size N] "
           "[--stop-after-op N] [--single-op-output N] "
           "[--single-op-output-index N] "
           "[--abs-threshold V] [--rel-threshold V] "
           "[--input-seed S] [--random-seed-count N] [--random-seed-base S] "
           "[--diagnostic-f32-mps-image] "
           "[--tinyllama-prompt-inputs] "
           "[--per-op-input-mode reference|generated|gfx-recursive] "
           "[--per-op-recursive-limit N] [--per-op-recursive-trace N] "
           "[--per-op-generated-inputs]\n";
    return 2;
  }

  const std::string model_path = argv[1];
  CompareOptions options;
  for (int i = 2; i < argc; ++i) {
    const std::string arg = argv[i];
    if (arg == "--per-op") {
      options.per_op = true;
      continue;
    }
    if (arg == "--per-op-all") {
      options.per_op = true;
      options.per_op_all = true;
      continue;
    }
    if (arg == "--print-ops") {
      options.print_ops = true;
      continue;
    }
    if (arg == "--gfx-only") {
      options.gfx_only = true;
      continue;
    }
    if (arg == "--reference-device") {
      if (i + 1 >= argc) {
        throw std::runtime_error("--reference-device requires a value");
      }
      options.reference_device = argv[++i];
      continue;
    }
    if (arg == "--reference-plugin") {
      if (i + 1 >= argc) {
        throw std::runtime_error("--reference-plugin requires a value");
      }
      options.reference_plugin_path = argv[++i];
      continue;
    }
    if (arg == "--start-op") {
      if (i + 1 >= argc) {
        throw std::runtime_error("--start-op requires a value");
      }
      options.start_op = static_cast<size_t>(std::stoul(argv[++i]));
      continue;
    }
    if (arg == "--window-size") {
      if (i + 1 >= argc) {
        throw std::runtime_error("--window-size requires a value");
      }
      options.window_size =
          std::max<size_t>(1, static_cast<size_t>(std::stoul(argv[++i])));
      continue;
    }
    if (arg == "--stop-after-op") {
      if (i + 1 >= argc) {
        throw std::runtime_error("--stop-after-op requires a value");
      }
      options.stop_after_op = static_cast<size_t>(std::stoul(argv[++i]));
      continue;
    }
    if (arg == "--single-op-output") {
      if (i + 1 >= argc) {
        throw std::runtime_error("--single-op-output requires a value");
      }
      options.single_output_op = static_cast<size_t>(std::stoul(argv[++i]));
      continue;
    }
    if (arg == "--single-op-output-index") {
      if (i + 1 >= argc) {
        throw std::runtime_error("--single-op-output-index requires a value");
      }
      options.single_output_port = static_cast<size_t>(std::stoul(argv[++i]));
      continue;
    }
    if (arg == "--abs-threshold") {
      if (i + 1 >= argc) {
        throw std::runtime_error("--abs-threshold requires a value");
      }
      options.abs_threshold = std::stod(argv[++i]);
      continue;
    }
    if (arg == "--rel-threshold") {
      if (i + 1 >= argc) {
        throw std::runtime_error("--rel-threshold requires a value");
      }
      options.rel_threshold = std::stod(argv[++i]);
      continue;
    }
    if (arg == "--input-seed") {
      if (i + 1 >= argc) {
        throw std::runtime_error("--input-seed requires a value");
      }
      options.input_seed = static_cast<uint64_t>(std::stoull(argv[++i]));
      continue;
    }
    if (arg == "--random-seed-count") {
      if (i + 1 >= argc) {
        throw std::runtime_error("--random-seed-count requires a value");
      }
      options.random_seed_count = static_cast<size_t>(std::stoull(argv[++i]));
      continue;
    }
    if (arg == "--random-seed-base") {
      if (i + 1 >= argc) {
        throw std::runtime_error("--random-seed-base requires a value");
      }
      options.random_seed_base = static_cast<uint64_t>(std::stoull(argv[++i]));
      continue;
    }
    if (arg == "--diagnostic-f32-mps-image") {
      options.diagnostic_f32_mps_image = true;
      continue;
    }
    if (arg == "--tinyllama-prompt-inputs") {
      options.tinyllama_prompt_inputs = true;
      continue;
    }
    if (arg == "--per-op-input-mode") {
      if (i + 1 >= argc) {
        throw std::runtime_error("--per-op-input-mode requires a value");
      }
      options.per_op_input_mode = parse_per_op_input_mode(argv[++i]);
      continue;
    }
    if (arg == "--per-op-recursive-limit") {
      if (i + 1 >= argc) {
        throw std::runtime_error("--per-op-recursive-limit requires a value");
      }
      options.per_op_recursive_limit =
          static_cast<size_t>(std::stoull(argv[++i]));
      continue;
    }
    if (arg == "--per-op-recursive-trace") {
      if (i + 1 >= argc) {
        throw std::runtime_error("--per-op-recursive-trace requires a value");
      }
      options.per_op_recursive_trace_every =
          static_cast<size_t>(std::stoull(argv[++i]));
      continue;
    }
    if (arg == "--per-op-generated-inputs") {
      options.per_op_input_mode = PerOpInputMode::Generated;
      continue;
    }
    throw std::runtime_error("unknown argument: " + arg);
  }

  ov::Core core;
  register_gfx_plugin(core);
  if (options.reference_device == "CPU") {
    throw std::runtime_error(
        "CPU reference device is not supported; use TEMPLATE");
  }
  register_reference_plugin(core, options.reference_device,
                            options.reference_plugin_path);

  auto model = core.read_model(model_path);
  const auto inputs = make_inputs(model, options);

  if (options.print_ops) {
    const auto ordered_ops = model->get_ordered_ops();
    for (size_t idx = 0; idx < ordered_ops.size(); ++idx) {
      const auto &node = ordered_ops[idx];
      if (is_debug_skippable_node(node)) {
        continue;
      }
      std::cout << "[op " << idx << "] " << node->get_friendly_name() << " ("
                << node->get_type_name() << ")\n";
    }
    return 0;
  }

  if (options.random_seed_count > 0) {
    if (options.gfx_only || options.per_op || options.per_op_all ||
        options.single_output_op.has_value()) {
      throw std::runtime_error("--random-seed-count is supported only for full "
                               "graph accuracy compare");
    }
    if (options.random_seed_base == 0) {
      options.random_seed_base = static_cast<uint64_t>(
          std::chrono::high_resolution_clock::now().time_since_epoch().count());
    }
    const auto ref_device = reference_device(core, options.reference_device);
    auto ref_model =
        core.compile_model(model, ref_device, make_compile_config(false));
    auto gfx_model =
        core.compile_model(model, "GFX", make_compile_config(true, &options));

    size_t failures = 0;
    double batch_max_abs = 0.0;
    double batch_max_rel = 0.0;
    std::cout << "RANDOM_SEED_BATCH count=" << options.random_seed_count
              << " base=" << options.random_seed_base
              << " reference_device=" << ref_device << '\n';
    for (size_t i = 0; i < options.random_seed_count; ++i) {
      CompareOptions seed_options = options;
      seed_options.input_seed =
          deterministic_validation_seed(options.random_seed_base, i);
      const auto seed_inputs = make_inputs(model, seed_options);
      auto ref_req = make_request(ref_model, seed_inputs);
      auto gfx_req = make_request(gfx_model, seed_inputs);
      ref_req.infer();
      gfx_req.infer();

      double seed_max_abs = 0.0;
      double seed_max_rel = 0.0;
      for (const auto &output : model->outputs()) {
        const auto stats = compare_tensors(ref_req.get_tensor(output),
                                           gfx_req.get_tensor(output));
        seed_max_abs = std::max(seed_max_abs, stats.max_abs_diff);
        seed_max_rel = std::max(seed_max_rel, stats.max_rel_diff);
      }
      batch_max_abs = std::max(batch_max_abs, seed_max_abs);
      batch_max_rel = std::max(batch_max_rel, seed_max_rel);
      const bool failed = seed_max_abs > options.abs_threshold &&
                          seed_max_rel > options.rel_threshold;
      failures += failed ? 1u : 0u;
      std::cout << "RANDOM_SEED_RESULT index=" << i
                << " seed=" << seed_options.input_seed
                << " max_abs_diff=" << std::setprecision(10) << seed_max_abs
                << " max_rel_diff=" << seed_max_rel
                << " status=" << (failed ? "FAIL" : "PASS") << '\n';
    }
    std::cout << "RANDOM_SEED_GLOBAL failures=" << failures
              << " max_abs_diff=" << std::setprecision(10) << batch_max_abs
              << " max_rel_diff=" << batch_max_rel << '\n';
    return failures == 0 ? 0 : 3;
  }

  if (options.single_output_op.has_value()) {
    const auto ordered_ops = model->get_ordered_ops();
    if (*options.single_output_op >= ordered_ops.size()) {
      throw std::runtime_error("--single-op-output is out of range");
    }
    const auto &node = ordered_ops[*options.single_output_op];
    if (node->get_output_size() == 0) {
      throw std::runtime_error(
          "--single-op-output selected node has no outputs");
    }
    if (options.single_output_port >= node->get_output_size()) {
      throw std::runtime_error(
          "--single-op-output-index is out of range for selected node");
    }
    auto debug_model = std::make_shared<ov::Model>(
        ov::OutputVector{std::make_shared<ov::op::v0::Result>(
            node->output(options.single_output_port))},
        model->get_parameters(), node->get_friendly_name() + "_debug_output");
    const auto debug_inputs = make_inputs(debug_model, options);
    const auto ref_device = reference_device(core, options.reference_device);
    auto ref_model =
        core.compile_model(debug_model, ref_device, make_compile_config(false));
    auto gfx_model = core.compile_model(debug_model, "GFX",
                                        make_compile_config(true, &options));
    std::cout << "SINGLE_OP_OUTPUT op_index=" << *options.single_output_op
              << " output_index=" << options.single_output_port
              << " name=" << node->get_friendly_name()
              << " type=" << node->get_type_name() << '\n';
    const auto stats =
        compare_model_outputs(ref_model, gfx_model, debug_inputs, true);
    std::cout << "GLOBAL max_abs_diff=" << std::setprecision(10)
              << stats.max_abs_diff << " max_rel_diff=" << stats.max_rel_diff
              << '\n';
    return (stats.max_abs_diff > options.abs_threshold &&
            stats.max_rel_diff > options.rel_threshold)
               ? 3
               : 0;
  }

  if (options.per_op_all) {
    return run_full_graph_per_op_compare(core, model, options);
  }

  if (options.per_op) {
    return run_per_op_compare(core, model, inputs, options);
  }

  if (options.gfx_only) {
    auto gfx_model =
        core.compile_model(model, "GFX", make_compile_config(true, &options));
    auto gfx_req = make_request(gfx_model, inputs);
    gfx_req.infer();
    maybe_print_gfx_profile(gfx_model);

    std::cout << "GFX_ONLY\n";
    for (const auto &output : model->outputs()) {
      const auto tensor = gfx_req.get_tensor(output);
      const auto summary = summarize_tensor(tensor);
      std::cout << output.get_node()->get_friendly_name() << ':'
                << output.get_index() << " elements=" << summary.elements
                << " finite=" << summary.finite_count
                << " nan=" << summary.nan_count << " inf=" << summary.inf_count
                << " min=" << std::setprecision(10) << summary.min
                << " max=" << summary.max << " mean=" << summary.mean
                << " l2=" << summary.l2 << '\n';
    }
    return 0;
  }

  const auto ref_device = reference_device(core, options.reference_device);
  std::cout << "REFERENCE device=" << ref_device << '\n';
  auto ref_model =
      core.compile_model(model, ref_device, make_compile_config(false));
  auto gfx_model =
      core.compile_model(model, "GFX", make_compile_config(true, &options));
  auto ref_req = make_request(ref_model, inputs);
  auto gfx_req = make_request(gfx_model, inputs);

  ref_req.infer();
  gfx_req.infer();
  maybe_print_gfx_profile(gfx_model);

  double global_max_abs = 0.0;
  double global_max_rel = 0.0;
  for (const auto &output : model->outputs()) {
    const auto ref_tensor = ref_req.get_tensor(output);
    const auto gfx_tensor = gfx_req.get_tensor(output);
    const auto stats = compare_tensors(ref_tensor, gfx_tensor);
    global_max_abs = std::max(global_max_abs, stats.max_abs_diff);
    global_max_rel = std::max(global_max_rel, stats.max_rel_diff);
    std::cout << output.get_node()->get_friendly_name() << ':'
              << output.get_index() << " elements=" << stats.elements
              << " max_abs_diff=" << std::setprecision(10) << stats.max_abs_diff
              << " max_rel_diff=" << stats.max_rel_diff << '\n';
  }

  std::cout << "GLOBAL max_abs_diff=" << std::setprecision(10) << global_max_abs
            << " max_rel_diff=" << global_max_rel << '\n';
  return 0;
} catch (const std::exception &ex) {
  std::cerr << "ERROR: " << ex.what() << '\n';
  return 1;
}
