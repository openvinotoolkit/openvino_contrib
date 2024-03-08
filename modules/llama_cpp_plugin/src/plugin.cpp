#include "plugin.hpp"
#include "compiled_model.hpp"
#include "openvino/op/constant.hpp"
#include <openvino/runtime/properties.hpp>
#include "openvino/runtime/internal_properties.hpp"


namespace {
static constexpr const char* wait_executor_name = "LlamaCppWaitExecutor";
static constexpr const char* stream_executor_name = "LlamaCppStreamsExecutor";
static constexpr const char* template_exclusive_executor = "LlamaCppExecutor";
}  // namespace


namespace ov {
    namespace llama_cpp_plugin {
        LlamaCppPlugin::LlamaCppPlugin() : IPlugin() {
            set_device_name("LLAMA_CPP");
        }
        std::shared_ptr<ov::ICompiledModel> LlamaCppPlugin::compile_model(const std::shared_ptr<const ov::Model>& model,
            const ov::AnyMap& properties) const {
            std::cout << "VSHAMPOR: LlamaCppPlugin::compile_model" << std::endl;

            //std::string gpt2_node_name = "transformer.h.9.attn.c_proj.weight";
            //std::cout << "VSHAMPOR: sanity check - looking for node containing " << gpt2_node_name << std::endl;
            //auto ops = model->get_ops();
            //auto iter = std::find_if(ops.begin(), ops.end(), [gpt2_node_name](const std::shared_ptr<ov::Node>& val) {
            //        return val->get_friendly_name().find(gpt2_node_name) != std::string::npos; });
            //if (iter == ops.end()) {
            //    std::cout << "VSHAMPOR: did not find the node\n";
            //} else {
            //    std::shared_ptr<ov::Node> node_with_tensor = *iter;
            //    std::cout << "VSHAMPOR: node type is " << node_with_tensor->get_type_name() << std::endl;
            //    std::shared_ptr<ov::op::v0::Constant> const_node_ptr = ov::as_type_ptr<ov::op::v0::Constant>(node_with_tensor);
            //    const float* data_ptr = const_node_ptr->get_data_ptr<element::Type_t::f32>();
            //    // ov::descriptor::Tensor& tensor_descr = node_with_tensor->get_output_tensor(0);
            //    // std::cout << "VSHAMPOR: node output tensor shape is " << tensor_descr.get_shape().to_string() << std::endl;
            //    // ov::TensorVector in, out;
            //    // node_with_tensor->evaluate(out, in);
            //    // std::cout << "VSHAMPOR: evaluated " << out.size() << " output tensors\n";
            //    // if (!out.empty()) {
            //    //     const ov::Tensor& tensor = out[0];
            //    //     const float* vals = tensor.data<float>();
            //    //     std::cout << "VSHAMPOR: first elements of the weight tensor are ";
            //    //     for (size_t i = 0; i < 10; i++) {
            //    //         std::cout << vals[i] << " ";
            //    //     }
            //    //     std::cout << std::endl;
            //    // }
            //    std::cout << "VSHAMPOR: first elements of the weight tensor are ";
            //    for (size_t i = 0; i < 10; i++) {
            //        std::cout << data_ptr[i] << " ";
            //    }
            //    std::cout << std::endl;
            //}
            return compile_model(model, properties, {});
        }

        std::shared_ptr<ov::ICompiledModel> LlamaCppPlugin::compile_model(const std::string& fname, const ov::AnyMap& properties) const {
            return std::make_shared<LlamaCppModel>(fname, shared_from_this());
        }
        std::shared_ptr<ov::ICompiledModel> LlamaCppPlugin::compile_model(const std::shared_ptr<const ov::Model>& model,
            const ov::AnyMap& properties,
            const ov::SoPtr<ov::IRemoteContext>& context) const {
            std::cout << "VSHAMPOR: compile_model called in C++" << std::endl;
            return std::make_shared<LlamaCppModel>(model->clone(), shared_from_this(), context, get_executor_manager()->get_executor(template_exclusive_executor));
        }

        void LlamaCppPlugin::set_property(const ov::AnyMap& properties) {
            for (const auto& map_entry : properties) {
                if (map_entry.first == ov::cache_dir.name()) {
                    m_cache_dir = map_entry.second.as<std::string>();
                }
                else {
                    OPENVINO_THROW_NOT_IMPLEMENTED("VSHAMPOR: setting property ", map_entry.first, "not implemented");
                }
            }
        }

        ov::Any LlamaCppPlugin::get_property(const std::string& name, const ov::AnyMap& arguments) const {
            if (ov::supported_properties == name) {
                return decltype(ov::supported_properties)::value_type(std::vector<PropertyName>({ov::cache_dir, ov::device::capabilities, ov::device::full_name}));
            }
            if (ov::device::capabilities == name) {
                return decltype(ov::device::capabilities)::value_type(std::vector<std::string>({ov::device::capability::EXPORT_IMPORT}));
            }
            if (ov::internal::supported_properties == name) {
                return decltype(ov::internal::supported_properties)::value_type(std::vector<PropertyName>({ov::internal::caching_properties}));
            }

            if (ov::cache_dir == name) {
                return m_cache_dir;
            }
            if (ov::internal::caching_properties == name) {
                return std::vector<ov::PropertyName>{ov::device::full_name};
            }

            if (ov::device::full_name == name) {
                return std::string("LLAMA_CPP");
            }

            OPENVINO_THROW_NOT_IMPLEMENTED("VSHAMPOR: Not Implemented");
        }

        ov::SoPtr<ov::IRemoteContext> LlamaCppPlugin::create_context(const ov::AnyMap& remote_properties) const {
            OPENVINO_THROW_NOT_IMPLEMENTED("VSHAMPOR: Not Implemented");
        }
        ov::SoPtr<ov::IRemoteContext> LlamaCppPlugin::get_default_context(const ov::AnyMap& remote_properties) const {
            OPENVINO_THROW_NOT_IMPLEMENTED("VSHAMPOR: Not Implemented");
        }
        std::shared_ptr<ov::ICompiledModel> LlamaCppPlugin::import_model(std::istream& model_file_stream,
            const ov::AnyMap& properties) const {
            std::cout << "VSHAMPOR: importing model" << '\n';
            std::cout << "VSHAMPOR: deserializing ov::Model first" << '\n';
             // read XML content
             std::string xmlString;
             std::uint64_t dataSize = 0;
             model_file_stream.read(reinterpret_cast<char*>(&dataSize), sizeof(dataSize));
             xmlString.resize(dataSize);
             model_file_stream.read(const_cast<char*>(xmlString.c_str()), dataSize);

             // read blob content
             ov::Tensor weights;
             model_file_stream.read(reinterpret_cast<char*>(&dataSize), sizeof(dataSize));
             if (0 != dataSize) {
                 weights = ov::Tensor(ov::element::from<char>(), ov::Shape{static_cast<ov::Shape::size_type>(dataSize)});
                 model_file_stream.read(weights.data<char>(), dataSize);
             }

             auto ov_model = get_core()->read_model(xmlString, weights);
            std::cout << "VSHAMPOR: ov::Model deserialized, passing the rest of the stream to LlamaCppModel ctor" << '\n';
            return std::make_shared<LlamaCppModel>(ov_model, model_file_stream, shared_from_this());
        }

        const std::string CURRENT_GGUF_FILE_NAME = "current.gguf";
        std::string LlamaCppPlugin::get_current_gguf_file_path() const { return m_cache_dir + "/" + CURRENT_GGUF_FILE_NAME; }

        std::shared_ptr<ov::ICompiledModel> LlamaCppPlugin::import_model(std::istream& model,
            const ov::SoPtr<ov::IRemoteContext>& context,
            const ov::AnyMap& properties) const {
            OPENVINO_THROW_NOT_IMPLEMENTED("VSHAMPOR: Not Implemented");
        }

        ov::SupportedOpsMap LlamaCppPlugin::query_model(const std::shared_ptr<const ov::Model>& model,
            const ov::AnyMap& properties) const {
            OPENVINO_THROW_NOT_IMPLEMENTED("VSHAMPOR: Not Implemented");
        }
    }
}  // namespace ov

static const ov::Version version = {CI_BUILD_NUMBER, "llama_cpp_plugin"};
OV_DEFINE_PLUGIN_CREATE_FUNCTION(ov::llama_cpp_plugin::LlamaCppPlugin, version)
