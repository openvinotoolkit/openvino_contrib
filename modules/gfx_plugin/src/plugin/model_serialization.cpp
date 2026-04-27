// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "plugin/model_serialization.hpp"

#include <sstream>
#include <string>

#include "openvino/core/except.hpp"
#include "openvino/pass/serialize.hpp"
#include "openvino/runtime/icore.hpp"
#include "openvino/runtime/shared_buffer.hpp"

namespace ov {
namespace gfx_plugin {
namespace {

uint64_t read_u64(std::istream& stream, const char* error_prefix) {
    uint64_t value = 0;
    stream.read(reinterpret_cast<char*>(&value), sizeof(value));
    OPENVINO_ASSERT(stream.good(), error_prefix, ": failed to read size");
    return value;
}

std::string read_string(std::istream& stream, uint64_t size, const char* error_prefix) {
    std::string data;
    data.resize(static_cast<size_t>(size));
    if (size > 0) {
        stream.read(data.data(), static_cast<std::streamsize>(size));
        OPENVINO_ASSERT(stream.good(), error_prefix, ": failed to read payload");
    }
    return data;
}

void write_u64(std::ostream& stream, uint64_t value) {
    stream.write(reinterpret_cast<const char*>(&value), sizeof(value));
}

}  // namespace

void write_model_to_stream(const std::shared_ptr<const ov::Model>& model, std::ostream& stream) {
    OPENVINO_ASSERT(model, "GFX: cannot export null model");

    std::stringstream xml_stream;
    std::stringstream bin_stream;
    ov::pass::Serialize serializer(xml_stream, bin_stream);
    auto clone = model->clone();
    serializer.run_on_model(clone);

    const std::string xml = xml_stream.str();
    const std::string bin = bin_stream.str();
    write_u64(stream, static_cast<uint64_t>(xml.size()));
    if (!xml.empty()) {
        stream.write(xml.data(), static_cast<std::streamsize>(xml.size()));
    }
    write_u64(stream, static_cast<uint64_t>(bin.size()));
    if (!bin.empty()) {
        stream.write(bin.data(), static_cast<std::streamsize>(bin.size()));
    }
}

std::shared_ptr<ov::Model> read_model_from_stream(const std::shared_ptr<ov::ICore>& core, std::istream& stream) {
    OPENVINO_ASSERT(core, "GFX: core is null");
    const uint64_t xml_size = read_u64(stream, "GFX: model import");
    const std::string xml = read_string(stream, xml_size, "GFX: model import");
    const uint64_t weights_size = read_u64(stream, "GFX: model import");

    ov::Tensor weights;
    if (weights_size > 0) {
        weights = ov::Tensor{ov::element::u8, ov::Shape{static_cast<size_t>(weights_size)}};
        stream.read(static_cast<char*>(weights.data()), static_cast<std::streamsize>(weights_size));
        OPENVINO_ASSERT(stream.good(), "GFX: model import: failed to read weights");
    }
    return core->read_model(xml, weights);
}

std::shared_ptr<ov::Model> read_model_from_buffer(const std::shared_ptr<ov::ICore>& core, const ov::Tensor& model) {
    ov::SharedStreamBuffer buffer{model.data(), model.get_byte_size()};
    std::istream stream{&buffer};
    return read_model_from_stream(core, stream);
}

}  // namespace gfx_plugin
}  // namespace ov
