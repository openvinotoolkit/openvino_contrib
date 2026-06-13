// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//
#pragma once

#include <iomanip>
#include <limits>
#include <sstream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

namespace ov {
namespace gfx_plugin {
namespace compiler {
namespace detail {

inline void append_materialization_field(std::ostringstream &os,
                                         std::string_view value) {
  os << value.size() << ":" << value << ";";
}

inline void append_materialization_bool(std::ostringstream &os, bool value) {
  append_materialization_field(os, value ? "1" : "0");
}

template <typename T>
void append_materialization_number(std::ostringstream &os, T value) {
  append_materialization_field(os, std::to_string(value));
}

inline void append_materialization_number(std::ostringstream &os, float value) {
  std::ostringstream value_os;
  value_os << std::setprecision(std::numeric_limits<float>::max_digits10)
           << value;
  append_materialization_field(os, value_os.str());
}

template <typename T>
void append_materialization_integral_vector(std::ostringstream &os,
                                            const std::vector<T> &values) {
  append_materialization_field(os, std::to_string(values.size()));
  for (const auto value : values) {
    append_materialization_field(os, std::to_string(value));
  }
}

inline void
append_materialization_float_vector(std::ostringstream &os,
                                    const std::vector<float> &values) {
  append_materialization_field(os, std::to_string(values.size()));
  for (const auto value : values) {
    append_materialization_number(os, value);
  }
}

inline void
append_materialization_string_vector(std::ostringstream &os,
                                     const std::vector<std::string> &values) {
  append_materialization_field(os, std::to_string(values.size()));
  for (const auto &value : values) {
    append_materialization_field(os, value);
  }
}

class MaterializationWireReader final {
public:
  explicit MaterializationWireReader(std::string_view wire) : m_wire(wire) {}

  std::vector<std::string> take_diagnostics() {
    return std::move(m_diagnostics);
  }

  void diagnostic(std::string message) {
    m_diagnostics.push_back(std::move(message));
  }

  std::string string_field(std::string_view name) {
    if (m_pos >= m_wire.size()) {
      m_diagnostics.push_back(
          std::string("cache materialization wire ended before ") +
          std::string(name));
      return {};
    }
    const size_t colon = m_wire.find(':', m_pos);
    if (colon == std::string_view::npos) {
      m_diagnostics.push_back(std::string("cache materialization field ") +
                              std::string(name) +
                              " has no length separator");
      m_pos = m_wire.size();
      return {};
    }
    const auto length_text = m_wire.substr(m_pos, colon - m_pos);
    size_t length = 0;
    try {
      length = static_cast<size_t>(std::stoull(std::string(length_text)));
    } catch (const std::exception &) {
      m_diagnostics.push_back(std::string("cache materialization field ") +
                              std::string(name) + " has invalid length");
      m_pos = m_wire.size();
      return {};
    }
    const size_t value_begin = colon + 1;
    const size_t value_end = value_begin + length;
    if (value_end >= m_wire.size() || m_wire[value_end] != ';') {
      m_diagnostics.push_back(std::string("cache materialization field ") +
                              std::string(name) + " is truncated");
      m_pos = m_wire.size();
      return {};
    }
    m_pos = value_end + 1;
    return std::string(m_wire.substr(value_begin, length));
  }

  bool bool_field(std::string_view name) {
    const auto value = string_field(name);
    if (value == "1") {
      return true;
    }
    if (value == "0") {
      return false;
    }
    m_diagnostics.push_back(std::string("cache materialization field ") +
                            std::string(name) + " is not bool");
    return false;
  }

  uint32_t u32_field(std::string_view name) {
    return static_cast<uint32_t>(u64_field(name));
  }

  uint64_t u64_field(std::string_view name) {
    const auto value = string_field(name);
    try {
      return static_cast<uint64_t>(std::stoull(value));
    } catch (const std::exception &) {
      m_diagnostics.push_back(std::string("cache materialization field ") +
                              std::string(name) + " is not uint64");
      return 0;
    }
  }

  size_t size_field(std::string_view name) {
    return static_cast<size_t>(u64_field(name));
  }

  int64_t i64_field(std::string_view name) {
    const auto value = string_field(name);
    try {
      return static_cast<int64_t>(std::stoll(value));
    } catch (const std::exception &) {
      m_diagnostics.push_back(std::string("cache materialization field ") +
                              std::string(name) + " is not int64");
      return 0;
    }
  }

  int32_t i32_field(std::string_view name) {
    return static_cast<int32_t>(i64_field(name));
  }

  float float_field(std::string_view name) {
    const auto value = string_field(name);
    try {
      return std::stof(value);
    } catch (const std::exception &) {
      m_diagnostics.push_back(std::string("cache materialization field ") +
                              std::string(name) + " is not float");
      return 0.0f;
    }
  }

  std::vector<std::string> string_vector(std::string_view name) {
    const auto count = size_field(name);
    std::vector<std::string> values;
    values.reserve(count);
    for (size_t i = 0; i < count; ++i) {
      values.push_back(string_field(name));
    }
    return values;
  }

  std::vector<size_t> size_vector(std::string_view name) {
    const auto count = size_field(name);
    std::vector<size_t> values;
    values.reserve(count);
    for (size_t i = 0; i < count; ++i) {
      values.push_back(size_field(name));
    }
    return values;
  }

  std::vector<int64_t> i64_vector(std::string_view name) {
    const auto count = size_field(name);
    std::vector<int64_t> values;
    values.reserve(count);
    for (size_t i = 0; i < count; ++i) {
      values.push_back(i64_field(name));
    }
    return values;
  }

  std::vector<int32_t> i32_vector(std::string_view name) {
    const auto count = size_field(name);
    std::vector<int32_t> values;
    values.reserve(count);
    for (size_t i = 0; i < count; ++i) {
      values.push_back(i32_field(name));
    }
    return values;
  }

  std::vector<uint32_t> u32_vector(std::string_view name) {
    const auto count = size_field(name);
    std::vector<uint32_t> values;
    values.reserve(count);
    for (size_t i = 0; i < count; ++i) {
      values.push_back(u32_field(name));
    }
    return values;
  }

  std::vector<float> float_vector(std::string_view name) {
    const auto count = size_field(name);
    std::vector<float> values;
    values.reserve(count);
    for (size_t i = 0; i < count; ++i) {
      values.push_back(float_field(name));
    }
    return values;
  }

private:
  std::string_view m_wire;
  size_t m_pos = 0;
  std::vector<std::string> m_diagnostics;
};

} // namespace detail
} // namespace compiler
} // namespace gfx_plugin
} // namespace ov
