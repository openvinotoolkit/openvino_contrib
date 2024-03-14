#include "compiled_model.hpp"

#include <fstream>
#include <memory>
#include <openvino/op/constant.hpp>
#include <openvino/opsets/opset13.hpp>
#include <openvino/runtime/properties.hpp>
#include <openvino/util/log.hpp>

#include "infer_request.hpp"
#include "plugin.hpp"

namespace ov {
namespace llama_cpp_plugin {

LlamaCppModel::LlamaCppModel(const std::shared_ptr<ov::Model>& model,
                             const std::shared_ptr<const ov::IPlugin>& plugin,
                             const ov::SoPtr<ov::IRemoteContext>& context,
                             const std::shared_ptr<ov::threading::ITaskExecutor>& task_executor)
    : ICompiledModel(model, plugin, context, task_executor) {
    OPENVINO_THROW_NOT_IMPLEMENTED("Currently only direct GGUF file loading is "
                                   "supported for the LLAMA_CPP* plugins");
}

LlamaCppModel::LlamaCppModel(const std::shared_ptr<ov::Model>& ov_model,
                             std::istream& input_stream,
                             const std::shared_ptr<const IPlugin>& plugin)
    : ICompiledModel(ov_model, plugin) {
    OPENVINO_THROW_NOT_IMPLEMENTED("Currently only direct GGUF file loading is "
                                   "supported for the LLAMA_CPP* plugins");
}

LlamaCppModel::~LlamaCppModel() {
    llama_free(m_llama_ctx);
    llama_free_model(m_llama_model_ptr);
    llama_backend_free();
    delete num_tokens_processed_ptr;
}

LlamaCppModel::LlamaCppModel(const std::string& gguf_fname, const std::shared_ptr<const IPlugin>& plugin)
    : ICompiledModel(nullptr, plugin),
      m_gguf_fname(gguf_fname) {
    num_tokens_processed_ptr = new size_t;  // TODO (vshampor): hack, remove
    *num_tokens_processed_ptr = 0;
    OPENVINO_DEBUG << "llama_cpp_plugin: loading llama model directly from GGUF... " << std::endl;
    llama_model_params mparams = llama_model_default_params();
    mparams.n_gpu_layers = 99;
    m_llama_model_ptr = llama_load_model_from_file(gguf_fname.c_str(), mparams);
    llama_context_params cparams = llama_context_default_params();
    m_llama_ctx = llama_new_context_with_model(m_llama_model_ptr, cparams);
    OPENVINO_DEBUG << "llama_cpp_plugin: llama model loaded successfully from GGUF..." << std::endl;

    auto input_ids = std::make_shared<ov::opset13::Parameter>(ov::element::Type_t::i64, ov::PartialShape({-1, -1}));
    auto fake_convert = std::make_shared<ov::opset13::Convert>(input_ids->output(0), ov::element::Type_t::f32);
    auto logits = std::make_shared<ov::opset13::Result>(fake_convert->output(0));

    ov::ParameterVector inputs{input_ids};

    std::vector<std::tuple<std::string, ov::element::Type_t, ov::PartialShape>> additional_inputs_in_order = {
        {"attention_mask", ov::element::Type_t::i64, {-1, -1}},
        {"position_ids", ov::element::Type_t::i64, {-1, -1}},
        {"beam_idx", ov::element::Type_t::i32, {-1, -1}}};

    for (const auto& descr : additional_inputs_in_order) {
        auto unused_inp = std::make_shared<ov::opset13::Parameter>(std::get<1>(descr), std::get<2>(descr));
        inputs.push_back(unused_inp);
    }

    m_fake_model = std::make_shared<ov::Model>(logits, inputs, "fake_ov_model_for_io_specification");

    m_fake_model->inputs()[0].set_names({"input_ids"});
    for (size_t i = 0; i < additional_inputs_in_order.size(); i++) {
        m_fake_model->inputs()[i + 1].set_names({std::get<0>(additional_inputs_in_order[i])});
    }

    m_fake_model->outputs()[0].set_names({"logits"});

    for (auto input : m_fake_model->inputs()) {
        m_fake_inputs.emplace_back(input);
    }
    for (auto output : m_fake_model->outputs()) {
        m_fake_outputs.emplace_back(output);
    }
}

std::shared_ptr<const ov::Model> LlamaCppModel::get_runtime_model() const {
    OPENVINO_THROW_NOT_IMPLEMENTED("llama_cpp_plugin: Not Implemented");
}

void LlamaCppModel::set_property(const ov::AnyMap& properties) {
    OPENVINO_DEBUG << "llama_cpp_plugin: attempted to set_property (did nothing)";
}

ov::Any LlamaCppModel::get_property(const std::string& name) const {
    if (ov::supported_properties == name) {
        return decltype(ov::supported_properties)::value_type(std::vector<PropertyName>());
    }
    OPENVINO_THROW_NOT_IMPLEMENTED("llama_cpp_plugin: Not Implemented");
}

std::shared_ptr<ov::ISyncInferRequest> LlamaCppModel::create_sync_infer_request() const {
    return std::make_shared<LlamaCppSyncInferRequest>(
        std::static_pointer_cast<const LlamaCppModel>(shared_from_this()));
}

const std::vector<ov::Output<const ov::Node>>& LlamaCppModel::inputs() const {
    return m_fake_inputs;
};
const std::vector<ov::Output<const ov::Node>>& LlamaCppModel::outputs() const {
    return m_fake_outputs;
};

void LlamaCppModel::export_model(std::ostream& output_stream) const {
    std::ifstream in(m_gguf_fname, std::ios::binary);
    output_stream << in.rdbuf();
}

}  // namespace llama_cpp_plugin
}  // namespace ov
