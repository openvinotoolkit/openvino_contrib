#pragma once

#include <cctype>
#include <cstdint>
#include <cstdlib>
#include <fstream>
#include <sstream>
#include <string>
#include <string_view>
#include <vector>

namespace ov {
namespace gfx_plugin {
namespace microbench {

struct CalibrationProbe {
    std::string name;
    std::string actual_backend;
    double arithmetic_intensity = 0.0;
    double overhead_subtracted_ms = 0.0;
    double adjusted_gbps = 0.0;
    double adjusted_tflops = 0.0;
    double gpu_gbps = 0.0;
    double gpu_tflops = 0.0;
    double first_to_steady_ratio = 0.0;
    double wait_share_of_wall = 0.0;
    double transfer_share_of_wall = 0.0;
    uint64_t submit_count = 0;
    uint64_t barrier_count = 0;
    std::vector<std::string> hints;
};

struct CalibrationArtifact {
    uint32_t schema_version = 1;
    uint32_t microbench_schema_version = 2;
    std::string tool = "ov_gfx_microbench";
    std::string device_key;
    std::string backend;
    std::string device_name;
    std::string platform;
    std::string vendor_id;
    std::string device_id;
    std::string driver_version;
    double fixed_overhead_us = 0.0;
    double bandwidth_estimate_gbps = 0.0;
    double compute_estimate_tflops = 0.0;
    double gpu_bandwidth_estimate_gbps = 0.0;
    double gpu_compute_estimate_tflops = 0.0;
    std::vector<std::string> triage_hints;
    std::vector<std::string> assumptions;
    std::vector<CalibrationProbe> probes;
};

inline std::string escape_json(std::string_view value) {
    std::ostringstream oss;
    for (const char ch : value) {
        switch (ch) {
        case '\\':
            oss << "\\\\";
            break;
        case '"':
            oss << "\\\"";
            break;
        case '\n':
            oss << "\\n";
            break;
        case '\r':
            oss << "\\r";
            break;
        case '\t':
            oss << "\\t";
            break;
        default:
            oss << ch;
            break;
        }
    }
    return oss.str();
}

inline std::string_view extract_json_object_field(std::string_view json, std::string_view field_name) {
    const std::string needle = "\"" + std::string(field_name) + "\":{";
    const size_t key_pos = json.find(needle);
    if (key_pos == std::string_view::npos) {
        return {};
    }
    const size_t object_start = key_pos + needle.size() - 1;
    size_t depth = 0;
    bool in_string = false;
    bool escaped = false;
    for (size_t i = object_start; i < json.size(); ++i) {
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
            depth += 1;
            continue;
        }
        if (c == '}') {
            depth -= 1;
            if (depth == 0) {
                return json.substr(object_start, i - object_start + 1);
            }
        }
    }
    return {};
}

inline std::string_view extract_json_array_field(std::string_view json, std::string_view field_name) {
    const std::string needle = "\"" + std::string(field_name) + "\":[";
    const size_t key_pos = json.find(needle);
    if (key_pos == std::string_view::npos) {
        return {};
    }
    const size_t array_start = key_pos + needle.size() - 1;
    size_t depth = 0;
    bool in_string = false;
    bool escaped = false;
    for (size_t i = array_start; i < json.size(); ++i) {
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
        if (c == '[') {
            depth += 1;
            continue;
        }
        if (c == ']') {
            depth -= 1;
            if (depth == 0) {
                return json.substr(array_start, i - array_start + 1);
            }
        }
    }
    return {};
}

inline std::vector<std::string_view> split_top_level_json_array_objects(std::string_view array_json) {
    std::vector<std::string_view> objects;
    if (array_json.size() < 2 || array_json.front() != '[' || array_json.back() != ']') {
        return objects;
    }
    bool in_string = false;
    bool escaped = false;
    size_t object_start = std::string_view::npos;
    size_t depth = 0;
    for (size_t i = 1; i + 1 < array_json.size(); ++i) {
        const char c = array_json[i];
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
            if (depth == 0) {
                object_start = i;
            }
            depth += 1;
            continue;
        }
        if (c == '}') {
            if (depth == 0) {
                continue;
            }
            depth -= 1;
            if (depth == 0 && object_start != std::string_view::npos) {
                objects.push_back(array_json.substr(object_start, i - object_start + 1));
                object_start = std::string_view::npos;
            }
        }
    }
    return objects;
}

inline std::vector<std::string> split_top_level_json_string_array(std::string_view array_json) {
    std::vector<std::string> values;
    if (array_json.size() < 2 || array_json.front() != '[' || array_json.back() != ']') {
        return values;
    }
    std::string current;
    bool in_string = false;
    bool escaped = false;
    for (size_t i = 1; i + 1 < array_json.size(); ++i) {
        const char c = array_json[i];
        if (!in_string) {
            if (c == '"') {
                in_string = true;
                current.clear();
            }
            continue;
        }
        if (escaped) {
            current.push_back(c);
            escaped = false;
            continue;
        }
        if (c == '\\') {
            escaped = true;
            continue;
        }
        if (c == '"') {
            values.push_back(current);
            in_string = false;
            continue;
        }
        current.push_back(c);
    }
    return values;
}

inline uint64_t extract_json_uint_field(std::string_view object_json, std::string_view field_name, uint64_t fallback = 0) {
    const std::string needle = "\"" + std::string(field_name) + "\":";
    const size_t pos = object_json.find(needle);
    if (pos == std::string_view::npos) {
        return fallback;
    }
    size_t begin = pos + needle.size();
    while (begin < object_json.size() && std::isspace(static_cast<unsigned char>(object_json[begin]))) {
        ++begin;
    }
    size_t end = begin;
    while (end < object_json.size() && std::isdigit(static_cast<unsigned char>(object_json[end]))) {
        ++end;
    }
    if (end == begin) {
        return fallback;
    }
    return static_cast<uint64_t>(std::strtoull(std::string(object_json.substr(begin, end - begin)).c_str(), nullptr, 10));
}

inline double extract_json_double_field(std::string_view object_json, std::string_view field_name, double fallback = 0.0) {
    const std::string needle = "\"" + std::string(field_name) + "\":";
    const size_t pos = object_json.find(needle);
    if (pos == std::string_view::npos) {
        return fallback;
    }
    size_t begin = pos + needle.size();
    while (begin < object_json.size() && std::isspace(static_cast<unsigned char>(object_json[begin]))) {
        ++begin;
    }
    size_t end = begin;
    while (end < object_json.size()) {
        const char c = object_json[end];
        if (!(std::isdigit(static_cast<unsigned char>(c)) || c == '.' || c == '-' || c == '+' || c == 'e' || c == 'E')) {
            break;
        }
        ++end;
    }
    if (end == begin) {
        return fallback;
    }
    return std::strtod(std::string(object_json.substr(begin, end - begin)).c_str(), nullptr);
}

inline std::string extract_json_string_field(std::string_view object_json, std::string_view field_name) {
    const std::string needle = "\"" + std::string(field_name) + "\":\"";
    const size_t pos = object_json.find(needle);
    if (pos == std::string_view::npos) {
        return {};
    }
    size_t begin = pos + needle.size();
    std::string out;
    bool escaped = false;
    for (size_t i = begin; i < object_json.size(); ++i) {
        const char c = object_json[i];
        if (escaped) {
            out.push_back(c);
            escaped = false;
            continue;
        }
        if (c == '\\') {
            escaped = true;
            continue;
        }
        if (c == '"') {
            return out;
        }
        out.push_back(c);
    }
    return {};
}

inline std::string calibration_probe_to_json(const CalibrationProbe& probe) {
    std::ostringstream json;
    json << "{";
    json << "\"name\":\"" << escape_json(probe.name) << "\",";
    json << "\"actual_backend\":\"" << escape_json(probe.actual_backend) << "\",";
    json << "\"arithmetic_intensity\":" << probe.arithmetic_intensity << ",";
    json << "\"overhead_subtracted_ms\":" << probe.overhead_subtracted_ms << ",";
    json << "\"adjusted_gbps\":" << probe.adjusted_gbps << ",";
    json << "\"adjusted_tflops\":" << probe.adjusted_tflops << ",";
    json << "\"gpu_gbps\":" << probe.gpu_gbps << ",";
    json << "\"gpu_tflops\":" << probe.gpu_tflops << ",";
    json << "\"first_to_steady_ratio\":" << probe.first_to_steady_ratio << ",";
    json << "\"wait_share_of_wall\":" << probe.wait_share_of_wall << ",";
    json << "\"transfer_share_of_wall\":" << probe.transfer_share_of_wall << ",";
    json << "\"submit_count\":" << probe.submit_count << ",";
    json << "\"barrier_count\":" << probe.barrier_count << ",";
    json << "\"hints\":[";
    for (size_t i = 0; i < probe.hints.size(); ++i) {
        if (i > 0) {
            json << ",";
        }
        json << "\"" << escape_json(probe.hints[i]) << "\"";
    }
    json << "]";
    json << "}";
    return json.str();
}

inline std::string calibration_artifact_to_json(const CalibrationArtifact& artifact) {
    std::ostringstream json;
    json << "{";
    json << "\"schema_version\":" << artifact.schema_version << ",";
    json << "\"microbench_schema_version\":" << artifact.microbench_schema_version << ",";
    json << "\"tool\":\"" << escape_json(artifact.tool) << "\",";
    json << "\"device_key\":\"" << escape_json(artifact.device_key) << "\",";
    json << "\"backend\":\"" << escape_json(artifact.backend) << "\",";
    json << "\"device_name\":\"" << escape_json(artifact.device_name) << "\",";
    json << "\"platform\":\"" << escape_json(artifact.platform) << "\",";
    json << "\"vendor_id\":\"" << escape_json(artifact.vendor_id) << "\",";
    json << "\"device_id\":\"" << escape_json(artifact.device_id) << "\",";
    json << "\"driver_version\":\"" << escape_json(artifact.driver_version) << "\",";
    json << "\"fixed_overhead_us\":" << artifact.fixed_overhead_us << ",";
    json << "\"bandwidth_estimate_gbps\":" << artifact.bandwidth_estimate_gbps << ",";
    json << "\"compute_estimate_tflops\":" << artifact.compute_estimate_tflops << ",";
    json << "\"gpu_bandwidth_estimate_gbps\":" << artifact.gpu_bandwidth_estimate_gbps << ",";
    json << "\"gpu_compute_estimate_tflops\":" << artifact.gpu_compute_estimate_tflops << ",";
    json << "\"triage_hints\":[";
    for (size_t i = 0; i < artifact.triage_hints.size(); ++i) {
        if (i > 0) {
            json << ",";
        }
        json << "\"" << escape_json(artifact.triage_hints[i]) << "\"";
    }
    json << "],";
    json << "\"assumptions\":[";
    for (size_t i = 0; i < artifact.assumptions.size(); ++i) {
        if (i > 0) {
            json << ",";
        }
        json << "\"" << escape_json(artifact.assumptions[i]) << "\"";
    }
    json << "],";
    json << "\"probes\":[";
    for (size_t i = 0; i < artifact.probes.size(); ++i) {
        if (i > 0) {
            json << ",";
        }
        json << calibration_probe_to_json(artifact.probes[i]);
    }
    json << "]";
    json << "}";
    return json.str();
}

inline CalibrationProbe parse_calibration_probe(std::string_view json) {
    CalibrationProbe probe;
    probe.name = extract_json_string_field(json, "name");
    probe.actual_backend = extract_json_string_field(json, "actual_backend");
    probe.arithmetic_intensity = extract_json_double_field(json, "arithmetic_intensity");
    probe.overhead_subtracted_ms = extract_json_double_field(json, "overhead_subtracted_ms");
    probe.adjusted_gbps = extract_json_double_field(json, "adjusted_gbps");
    probe.adjusted_tflops = extract_json_double_field(json, "adjusted_tflops");
    probe.gpu_gbps = extract_json_double_field(json, "gpu_gbps");
    probe.gpu_tflops = extract_json_double_field(json, "gpu_tflops");
    probe.first_to_steady_ratio = extract_json_double_field(json, "first_to_steady_ratio");
    probe.wait_share_of_wall = extract_json_double_field(json, "wait_share_of_wall");
    probe.transfer_share_of_wall = extract_json_double_field(json, "transfer_share_of_wall");
    probe.submit_count = extract_json_uint_field(json, "submit_count");
    probe.barrier_count = extract_json_uint_field(json, "barrier_count");
    probe.hints = split_top_level_json_string_array(extract_json_array_field(json, "hints"));
    return probe;
}

inline CalibrationArtifact parse_calibration_artifact(std::string_view json) {
    CalibrationArtifact artifact;
    artifact.schema_version = static_cast<uint32_t>(extract_json_uint_field(json, "schema_version", artifact.schema_version));
    artifact.microbench_schema_version =
        static_cast<uint32_t>(extract_json_uint_field(json, "microbench_schema_version", artifact.microbench_schema_version));
    artifact.tool = extract_json_string_field(json, "tool");
    artifact.device_key = extract_json_string_field(json, "device_key");
    artifact.backend = extract_json_string_field(json, "backend");
    artifact.device_name = extract_json_string_field(json, "device_name");
    artifact.platform = extract_json_string_field(json, "platform");
    artifact.vendor_id = extract_json_string_field(json, "vendor_id");
    artifact.device_id = extract_json_string_field(json, "device_id");
    artifact.driver_version = extract_json_string_field(json, "driver_version");
    artifact.fixed_overhead_us = extract_json_double_field(json, "fixed_overhead_us");
    artifact.bandwidth_estimate_gbps = extract_json_double_field(json, "bandwidth_estimate_gbps");
    artifact.compute_estimate_tflops = extract_json_double_field(json, "compute_estimate_tflops");
    artifact.gpu_bandwidth_estimate_gbps = extract_json_double_field(json, "gpu_bandwidth_estimate_gbps");
    artifact.gpu_compute_estimate_tflops = extract_json_double_field(json, "gpu_compute_estimate_tflops");
    artifact.triage_hints = split_top_level_json_string_array(extract_json_array_field(json, "triage_hints"));
    artifact.assumptions = split_top_level_json_string_array(extract_json_array_field(json, "assumptions"));
    const auto probe_array = extract_json_array_field(json, "probes");
    for (const auto& probe_json : split_top_level_json_array_objects(probe_array)) {
        artifact.probes.push_back(parse_calibration_probe(probe_json));
    }
    return artifact;
}

inline bool write_calibration_artifact_file(const CalibrationArtifact& artifact, const std::string& path) {
    std::ofstream out(path, std::ios::out | std::ios::trunc);
    if (!out.is_open()) {
        return false;
    }
    out << calibration_artifact_to_json(artifact) << '\n';
    return true;
}

inline bool read_calibration_artifact_file(const std::string& path, CalibrationArtifact& artifact) {
    std::ifstream in(path);
    if (!in.is_open()) {
        return false;
    }
    const std::string json((std::istreambuf_iterator<char>(in)), std::istreambuf_iterator<char>());
    artifact = parse_calibration_artifact(json);
    return true;
}

}  // namespace microbench
}  // namespace gfx_plugin
}  // namespace ov
