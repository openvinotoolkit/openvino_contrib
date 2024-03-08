#include "compiled_model.hpp"
#include "plugin.hpp"
#include "infer_request.hpp"
#include <memory>
#include <openvino/op/constant.hpp>
#include <openvino/opsets/opset13.hpp>
#include <fstream>
#include <openvino/runtime/properties.hpp>

namespace ov {
    namespace llama_cpp_plugin {
        class TensorWeightMatcher {
        public:
            // TODO (vshampor) implement this for faster weight node matching.
            // Use std::list, two passes - first for full name match, second for prefix-match; remove entries from list on match
            using RTInfoTensorName = std::string;
            using OvNodeName = std::string;
            using LlamaTensorName = std::string;

            TensorWeightMatcher(const std::shared_ptr<ov::Model>& model, std::map<RTInfoTensorName, ov::Shape> tensor_names_with_shapes_to_match) {
                std::multimap<RTInfoTensorName, std::shared_ptr<ov::op::v0::Constant>> intermediate_matches_map;

                const auto node_vector = model->get_ops();
                std::list<std::shared_ptr<ov::op::v0::Constant>> const_nodes_in_model;
                for (const auto& node_ptr : node_vector) {
                    if (ov::is_type<ov::op::v0::Constant>(node_ptr)) const_nodes_in_model.push_back(ov::as_type_ptr<ov::op::v0::Constant>(node_ptr));
                }

                // full substring match pass
                std::map<RTInfoTensorName, ov::Shape> unmatched_rt_info_names_on_first_pass = extract_matches(intermediate_matches_map, tensor_names_with_shapes_to_match, const_nodes_in_model,
                        [](const std::string& substring, const std::string& source) { return source.find(substring) != std::string::npos; });

                // prefix substring match pass
                std::map<RTInfoTensorName, ov::Shape> unmatched_rt_info_names_on_second_pass = extract_matches(intermediate_matches_map, unmatched_rt_info_names_on_first_pass, const_nodes_in_model,
                        [](const std::string& substring, const std::string& source) {
                        return source.find(get_weight_name_without_torch_postfix(substring)) != std::string::npos; });

                for (auto it = intermediate_matches_map.begin(); it != intermediate_matches_map.end(); it = intermediate_matches_map.upper_bound(it->first)) {
                    // TODO: perf improvement by iterating with ++;
                    RTInfoTensorName rt_info_name = it->first;
                    if (intermediate_matches_map.count(rt_info_name) != 1) {
                        std::cout << "VSHAMPOR: multiple matches for weight name " << rt_info_name << " and shape " << it->second->get_shape().to_string() << ", found ";
                        auto range_it_pair = intermediate_matches_map.equal_range(rt_info_name);
                        for (auto multimatch_it = range_it_pair.first; multimatch_it != range_it_pair.second; multimatch_it++) {
                            auto node_ptr = multimatch_it->second;
                            std::cout << node_ptr->get_friendly_name() << "(shape " << node_ptr->get_shape().to_string() << "),";
                        }
                        std::cout << "will take the first match" << std::endl;
                    }
                    const auto& match = intermediate_matches_map.find(rt_info_name)->second;
                    m_rtinfo_name_to_weight_node_map[rt_info_name] = match;
                }
                if (!unmatched_rt_info_names_on_second_pass.empty()) {
                    std::cout << "VSHAMPOR: did not find the weight node for " << unmatched_rt_info_names_on_second_pass.size() << " weights:" << std::endl;
                }
                for (const auto& unmatched_entry: unmatched_rt_info_names_on_second_pass) {
                    std::cout << '\t' << unmatched_entry.first << std::endl;
                }
            }

        std::unordered_map<RTInfoTensorName, std::shared_ptr<ov::op::v0::Constant>> get_matches() { return m_rtinfo_name_to_weight_node_map; }

        private:
            std::map<RTInfoTensorName, ov::Shape> extract_matches(std::multimap<RTInfoTensorName, std::shared_ptr<ov::op::v0::Constant>>& output_matches_map,
                                                                  const std::map<RTInfoTensorName, ov::Shape>& names_with_shapes_to_match,
                                                                  const std::list<std::shared_ptr<ov::op::v0::Constant>>& search_list,
                                                                  std::function<bool(const std::string& substring, const std::string& source)> name_match_predicate) {
                std::map<RTInfoTensorName, ov::Shape> unmatched_rt_info_names;
                for (const auto& pair: names_with_shapes_to_match) {
                    RTInfoTensorName rt_info_name = pair.first;
                    const ov::Shape& wanted_shape = pair.second;
                    bool matched = false;
                    for (auto it = search_list.begin(); it != search_list.end(); it++) {
                        auto node_ptr = *it;
                        const std::string& friendly_name = node_ptr->get_friendly_name();
                        if (name_match_predicate(rt_info_name, friendly_name) &&
                            node_ptr->get_shape() == wanted_shape) {
                            output_matches_map.insert(std::make_pair(rt_info_name, node_ptr));
                            matched = true;
                            break;
                        }
                    }
                    if (!matched) unmatched_rt_info_names.insert(pair);
                }
                return unmatched_rt_info_names;
            }

            static std::string get_weight_name_without_torch_postfix(std::string torch_weight_name) {
                size_t idx = torch_weight_name.rfind(".");
                if (idx == std::string::npos) return torch_weight_name;
                return std::string(torch_weight_name, 0, idx);
            }

            size_t num_exact_matches = 0;
            size_t num_partial_matches = 0;
            std::unordered_map<RTInfoTensorName, std::shared_ptr<ov::op::v0::Constant>> m_rtinfo_name_to_weight_node_map;
        };


        std::vector<std::shared_ptr<ov::Node>> get_nodes_containing_name_with_shape(const std::shared_ptr<ov::Model>& model, const std::string& weight_name, const ov::Shape& shape) {
            auto ops = model->get_ops();
            std::vector<std::shared_ptr<ov::Node>> found_weight_nodes;
            std::copy_if(ops.begin(), ops.end(), std::back_inserter(found_weight_nodes),
                    [&weight_name, &shape](const std::shared_ptr<ov::Node>& val) {
                        if (!ov::is_type<ov::op::v0::Constant>(val)) return false;
                        std::shared_ptr<ov::op::v0::Constant> node_ptr = ov::as_type_ptr<ov::op::v0::Constant>(val);
                        return val->get_friendly_name().find(weight_name) != std::string::npos &&
                               val->get_shape() == shape;
                    });
            return found_weight_nodes;
        }

        bool has_weight_matches(const std::shared_ptr<ov::Model>& model, const std::string& weight_name, const ov::Shape& shape) {
            std::vector<std::shared_ptr<ov::Node>> found_weight_nodes;
            found_weight_nodes = get_nodes_containing_name_with_shape(model, weight_name, shape);
            return !found_weight_nodes.empty();
        }

        std::string get_weight_name_without_torch_postfix(std::string torch_weight_name) {
            size_t idx = torch_weight_name.rfind(".");
            if (idx == std::string::npos) return torch_weight_name;
            return std::string(torch_weight_name, 0, idx);
        }

        bool has_partial_weight_matches(const std::shared_ptr<ov::Model>& model, const std::string& weight_name, const ov::Shape& shape) {
            std::vector<std::shared_ptr<ov::Node>> found_weight_nodes;
            found_weight_nodes = get_nodes_containing_name_with_shape(model, get_weight_name_without_torch_postfix(weight_name), shape);
            return !found_weight_nodes.empty();
        }

        std::shared_ptr<ov::op::v0::Constant> get_weight_by_name_and_shape(const std::shared_ptr<ov::Model>& model, const std::string& weight_name, const ov::Shape& shape) {
            OPENVINO_ASSERT(has_weight_matches(model, weight_name, shape));
            std::vector<std::shared_ptr<ov::Node>> found_weight_nodes;
            found_weight_nodes = get_nodes_containing_name_with_shape(model, weight_name, shape);

            if (found_weight_nodes.size() > 1) {
                std::cout << "VSHAMPOR: multiple matches for weight name " << weight_name << " and shape " << shape.to_string() << ", found ";
                for (const auto& node_ptr : found_weight_nodes) {
                    std::cout << node_ptr->get_friendly_name() << "(shape " << shape.to_string() << "),";
                }
                std::cout << "will take the first match" << std::endl;
            }
            std::shared_ptr<ov::Node> node_with_tensor = found_weight_nodes.front();
            OPENVINO_ASSERT(ov::is_type<ov::op::v0::Constant>(node_with_tensor));
            std::shared_ptr<ov::op::v0::Constant> const_node_ptr = ov::as_type_ptr<ov::op::v0::Constant>(node_with_tensor);
            return const_node_ptr;
        }

        using TransposePermutation = std::pair<size_t, size_t>;

        std::vector<size_t> expand_front(const std::vector<size_t>& vec, size_t val) {
            OPENVINO_ASSERT(vec.size() < GGML_MAX_DIMS);
            std::vector<size_t> retval(GGML_MAX_DIMS, val);
            std::copy(vec.rbegin(), vec.rend(), retval.rbegin());
            return retval;
        }

        void write_float_plus_one(std::ofstream& out, const float* src) {
            float elt = *src;
            elt += 1;
            out.write((const char*) &elt, sizeof(float));
        }

        void append_tensor_data_with_transpositions(const std::string& fname, const std::vector<gguf_tensor_info>& tensor_infos, const std::vector<void*>& tensor_data_ptrs,
                const std::map<std::string, TransposePermutation>& transpositions, const std::set<std::string> increment_by_one_tensor_names) {
             // assuming contiguous data underneath each pointer from tensor_data_ptrs
             OPENVINO_ASSERT(tensor_infos.size() == tensor_data_ptrs.size());
             std::ofstream out(fname, std::ios::app | std::ios::out);
             for (size_t i = 0; i < tensor_infos.size(); i++) {
                const auto& tensor_info = tensor_infos[i];
                OPENVINO_ASSERT(tensor_info.type == GGML_TYPE_F32); // TODO (vshampor): writing transposed tensor data for other data types, especially lower-bitwidth; maybe use OV inference for that

                const char* ir_tensor_data = reinterpret_cast<char*>(tensor_data_ptrs[i]);

                std::string tensor_llama_name = std::string(tensor_info.name.data);
                auto it = transpositions.find(tensor_llama_name);
                if (it == transpositions.end()) {
                    // original IR tensor should not be transposed to conform to GGUF expectations, can write as-is
                    if (increment_by_one_tensor_names.count(tensor_llama_name) != 0) { // gemma case
                        size_t elt_size = sizeof(float); // FP32 only for now
                        OPENVINO_ASSERT(!(tensor_info.size % elt_size));
                        size_t num_elts = tensor_info.size / elt_size;
                        for (size_t elt_idx = 0; elt_idx < num_elts; elt_idx++) {
                            write_float_plus_one(out, ((float*) ir_tensor_data) + elt_idx);
                        }
                    }
                    else {
                        out.write(ir_tensor_data, tensor_info.size);
                    }
                    continue;
                }

                if (it != transpositions.end()) {
                    std::vector<size_t> gguf_layout_shape;

                    // the shape in .ne is inverted w.r.t original export (~= IR) weight layout
                    for (size_t dim_idx = 0; dim_idx < tensor_info.n_dims; dim_idx++) {
                        gguf_layout_shape.push_back(tensor_info.ne[GGML_MAX_DIMS - 1 - tensor_info.n_dims - dim_idx]);
                    }

                    TransposePermutation permutation = it->second;
                    std::vector<size_t> ir_layout_shape(gguf_layout_shape);
                    std::swap(ir_layout_shape[permutation.first], ir_layout_shape[permutation.second]);

                    std::vector<size_t> ir_layout_strides(tensor_info.n_dims, 1);

                    for (size_t idx = 0; idx < tensor_info.n_dims - 1 ; idx++) {
                        auto previous_stride_it = ir_layout_strides.rbegin() + idx;
                        auto stride_it = ir_layout_strides.rbegin() + idx + 1;
                        auto shape_it = ir_layout_shape.rbegin() + idx;
                        *stride_it = *shape_it * *previous_stride_it;
                    }


                    std::vector<size_t> permuted_strides(ir_layout_strides);
                    std::swap(permuted_strides[permutation.first], permuted_strides[permutation.second]);

                    // expand up to GGML_MAX_DIMS
                    std::vector<size_t> gguf_layout_shape_ex = expand_front(gguf_layout_shape, 1);
                    // stride for unused dims will be 0, has no effect on loop because dimension idx for that dim is always 0
                    permuted_strides = expand_front(permuted_strides, 0);



                    std::cout << "VSHAMPOR: writing tensor " << tensor_info.name.data << " with size " << tensor_info.size;
                    std::cout << " shape (GGUF layout) ";
                    for (auto dim: gguf_layout_shape) std::cout << dim << ",";
                    std::cout << " shape (IR layout) ";
                    for (auto dim : ir_layout_shape) std::cout << dim << ",";
                    std::cout << " stride (IR layout) ";
                    for (auto stride : ir_layout_strides) std::cout << stride << ",";
                    std::cout << " stride (IR layout, transposing) ";
                    for (auto stride : permuted_strides) std::cout << stride << ",";
                    std::cout << std::endl;

                    // TODO (vshampor): rewrite the loop below using recurrent templates?
                    // This relies on GGUF_MAX_DIMS == 4 and unused dims being equal to 1
                    size_t current_offset = 0;
                    size_t element_size = sizeof(float);
                    size_t num_bytes_written = 0;
                    for (size_t dim_0 = 0; dim_0 < gguf_layout_shape_ex[0]; dim_0++)
                        for (size_t dim_1 = 0; dim_1 < gguf_layout_shape_ex[1]; dim_1++)
                            for (size_t dim_2 = 0; dim_2 < gguf_layout_shape_ex[2]; dim_2++)
                                for (size_t dim_3 = 0; dim_3 < gguf_layout_shape_ex[3]; dim_3++) {
                                    current_offset = element_size * (dim_0 * permuted_strides[0] + dim_1 * permuted_strides[1] + dim_2 * permuted_strides[2] + dim_3 * permuted_strides[3]);
                                    if (increment_by_one_tensor_names.count(tensor_llama_name) != 0) { // gemma case
                                        write_float_plus_one(out, (float*) ir_tensor_data + current_offset);
                                    }
                                    else {
                                        out.write(ir_tensor_data + current_offset, element_size);
                                    }
                                    num_bytes_written += element_size;
                                }
                    std::cout << "VSHAMPOR: wrote " << num_bytes_written << std::endl;
                    OPENVINO_ASSERT(num_bytes_written == tensor_info.size);
                }
             }
        }

        struct ValueStorageForLifetimeExtension {
            std::list<std::string> kv_key_string_storage;
            std::list<std::string> kv_value_string_storage;
            std::list<std::vector<char*>> str_arr_storage;
            void* store_gguf_value_vector(const std::vector<gguf_value>& vec, gguf_type g_type) {
                size_t elt_size;
                switch (g_type) {
                    case GGUF_TYPE_UINT8:   elt_size = sizeof(uint8_t); break;
                    case GGUF_TYPE_INT8:    elt_size = sizeof(int8_t); break;
                    case GGUF_TYPE_UINT16:  elt_size = sizeof(uint16_t); break;
                    case GGUF_TYPE_INT16:   elt_size = sizeof(int16_t); break;
                    case GGUF_TYPE_UINT32:  elt_size = sizeof(uint32_t); break;
                    case GGUF_TYPE_INT32:   elt_size = sizeof(int32_t); break;
                    case GGUF_TYPE_FLOAT32: elt_size = sizeof(float); break;
                    case GGUF_TYPE_UINT64:  elt_size = sizeof(uint64_t); break;
                    case GGUF_TYPE_INT64:   elt_size = sizeof(int64_t); break;
                    case GGUF_TYPE_FLOAT64: elt_size = sizeof(double); break;
                    case GGUF_TYPE_BOOL:    elt_size = sizeof(bool); break;
                default:
                    OPENVINO_THROW("Unknown array type");
                }
                size_t size_in_bytes = vec.size() * elt_size;
                void* mem_ptr = new char[size_in_bytes];
                for (size_t i = 0; i < vec.size(); i++) {
                    switch (g_type) {
                        case GGUF_TYPE_UINT8:   ((uint8_t*) mem_ptr)[i] = vec[i].uint8;     break;
                        case GGUF_TYPE_INT8:    ((int8_t*) mem_ptr)[i] = vec[i].int8;      break;
                        case GGUF_TYPE_UINT16:  ((uint16_t*) mem_ptr)[i] = vec[i].uint16;    break;
                        case GGUF_TYPE_INT16:   ((int16_t*) mem_ptr)[i] = vec[i].int16;     break;
                        case GGUF_TYPE_UINT32:  ((uint32_t*) mem_ptr)[i] = vec[i].uint32;    break;
                        case GGUF_TYPE_INT32:   ((int32_t*) mem_ptr)[i] = vec[i].int32;     break;
                        case GGUF_TYPE_FLOAT32: ((float*) mem_ptr)[i] = vec[i].float32;   break;
                        case GGUF_TYPE_UINT64:  ((uint64_t*) mem_ptr)[i] = vec[i].uint64;    break;
                        case GGUF_TYPE_INT64:   ((int64_t*) mem_ptr)[i] = vec[i].int64;     break;
                        case GGUF_TYPE_FLOAT64: ((double*) mem_ptr)[i] = vec[i].float64;   break;
                        case GGUF_TYPE_BOOL:    ((bool*) mem_ptr)[i] = vec[i].bool_;     break;
                    default:
                        OPENVINO_THROW("Unknown array type");
                    }
                }
                return mem_ptr;
            }

            ValueStorageForLifetimeExtension() = default;
            ~ValueStorageForLifetimeExtension() {
                for (void* ptr: non_str_raw_storage) {
                    delete[] (char*) ptr;
                }
            }
            private:
            std::list<void*> non_str_raw_storage;
        };

        bool maybe_parse_single_element(gguf_type g_type, ov::Any rtmap_value, gguf_value& dst, ValueStorageForLifetimeExtension& store) {
                switch (g_type) {
                    case GGUF_TYPE_UINT8:   dst.uint8    = rtmap_value.as<uint8_t>();  break;
                    case GGUF_TYPE_INT8:    dst.int8     = rtmap_value.as<int8_t>(); ; break;
                    case GGUF_TYPE_UINT16:  dst.uint16   = rtmap_value.as<uint16_t>(); break;
                    case GGUF_TYPE_INT16:   dst.int16    = rtmap_value.as<int16_t>();  break;
                    case GGUF_TYPE_UINT32:  dst.uint32   = rtmap_value.as<uint32_t>(); break;
                    case GGUF_TYPE_INT32:   dst.int32    = rtmap_value.as<int32_t>();  break;
                    case GGUF_TYPE_FLOAT32: dst.float32  = rtmap_value.as<float>();    break;
                    case GGUF_TYPE_UINT64:  dst.uint64   = rtmap_value.as<uint64_t>(); break;
                    case GGUF_TYPE_INT64:   dst.int64    = rtmap_value.as<int64_t>();  break;
                    case GGUF_TYPE_FLOAT64: dst.float64  = rtmap_value.as<double>();   break;
                    case GGUF_TYPE_BOOL:    dst.bool_    = rtmap_value.as<bool>();     break;
                    case GGUF_TYPE_STRING: {
                        std::string string_value = rtmap_value.as<std::string>();
                        store.kv_value_string_storage.push_back(string_value);
                        dst.str.n = string_value.length();
                        dst.str.data = (char*) store.kv_value_string_storage.back().c_str(); // TODO (vshampor) see equivalent case below
                        break;
                    }
                    default:
                        return false;  // did not parse
                }
            return true; // parsed successfully
        }

        ov::Any get_any_associated_with_gguf_type(gguf_type g_type) {
            switch (g_type) {
                case GGUF_TYPE_UINT8:   return ov::Any(uint8_t());
                case GGUF_TYPE_INT8:    return ov::Any(int8_t());  
                case GGUF_TYPE_UINT16:  return ov::Any(uint16_t());
                case GGUF_TYPE_INT16:   return ov::Any(int16_t()); 
                case GGUF_TYPE_UINT32:  return ov::Any(uint32_t());
                case GGUF_TYPE_INT32:   return ov::Any(int32_t()); 
                case GGUF_TYPE_FLOAT32: return ov::Any(float());   
                case GGUF_TYPE_UINT64:  return ov::Any(uint64_t());
                case GGUF_TYPE_INT64:   return ov::Any(int64_t()); 
                case GGUF_TYPE_FLOAT64: return ov::Any(double());  
                case GGUF_TYPE_BOOL:    return ov::Any(bool());    
                case GGUF_TYPE_STRING:  return ov::Any(std::string());
                default:
                    OPENVINO_THROW("Unknown gguf_type to turn into ov::Any");
            }
        }


        LlamaCppModel::LlamaCppModel(const std::shared_ptr<ov::Model>& model,
                      const std::shared_ptr<const ov::IPlugin>& plugin,
                      const ov::SoPtr<ov::IRemoteContext>& context,
                      const std::shared_ptr<ov::threading::ITaskExecutor>& task_executor
                      ) : ICompiledModel(model, plugin, context, task_executor) {
            m_model = model;
            num_tokens_processed_ptr = new size_t; // TODO (vshampor): hack, remove
            *num_tokens_processed_ptr = 0;
            auto rt_info = model->get_rt_info();
            OPENVINO_ASSERT(rt_info.count("lcp_kv_params") != 0);
            OPENVINO_ASSERT(rt_info.count("lcp_kv_types") != 0);
            OPENVINO_ASSERT(rt_info.count("lcp_kv_array_types") != 0);
            OPENVINO_ASSERT(rt_info.count("lcp_tensor_name_map") != 0);
            OPENVINO_ASSERT(rt_info.count("lcp_tensor_shape_map") != 0);
            OPENVINO_ASSERT(rt_info.count("lcp_expected_tensor_shapes") != 0);
            OPENVINO_ASSERT(rt_info.count("lcp_transpose_permutations") != 0);

            RTMap& kv_params = model->get_rt_info<RTMap&>("lcp_kv_params");
            RTMap& kv_types = model->get_rt_info<RTMap&>("lcp_kv_types");
            RTMap& kv_array_types = model->get_rt_info<RTMap&>("lcp_kv_array_types");
            RTMap& tensor_name_map = model->get_rt_info<RTMap&>("lcp_tensor_name_map");
            RTMap& tensor_shape_map = model->get_rt_info<RTMap&>("lcp_tensor_shape_map");
            RTMap& expected_tensor_shapes_map = model->get_rt_info<RTMap&>("lcp_expected_tensor_shapes");
            RTMap& transpose_permutations_rtmap = model->get_rt_info<RTMap&>("lcp_transpose_permutations");

            size_t gguf_version = model->get_rt_info<size_t>("lcp_gguf_version");
            std::cout << "VSHAMPOR: parsed gguf_version " << gguf_version << std::endl;

            // kv params
            OPENVINO_ASSERT(kv_params.size() == kv_types.size());
            size_t n_kv = kv_params.size();
            std::vector<gguf_kv> kv_vector;
            ValueStorageForLifetimeExtension store;

            for (const auto& kv_pair: kv_params) {
                gguf_kv kv;

                const auto& key = kv_pair.first;
                kv.key.n = key.length();
                store.kv_key_string_storage.push_back(key);
                kv.key.data = (char*) store.kv_key_string_storage.back().c_str(); // TODO (vshampor) see equivalent case below

                uint32_t value_type = kv_types[key].as<uint32_t>();
                gguf_type gguf_value_type = (gguf_type) value_type;
                kv.type = gguf_value_type;
                if (gguf_value_type != GGUF_TYPE_ARRAY) {
                    bool is_parsed = maybe_parse_single_element(kv.type, kv_pair.second, kv.value, store);
                    OPENVINO_ASSERT(is_parsed, "Invalid type of a GGUF kv-value");
                }
                else { // array case
                    gguf_type element_type = (gguf_type) kv_array_types[key].as<uint32_t>();
                    kv.value.arr.type = element_type;
                    std::string serialized_array = kv_pair.second.as<std::string>();
                    std::stringstream ss{serialized_array};
                    std::vector<gguf_value> parsed_array;
                    while (!ss.eof()) {
                        gguf_value array_elt;
                        ov::Any ov_any = get_any_associated_with_gguf_type(element_type);
                        std::string token; ss >> token;
                        if (std::string(kv.key.data) == "tokenizer.ggml.merges") {
                            // tokenizer merges are pairs of tokens separated by whitespace, so need to read another to get a proper merge
                            // TODO (vshampor): think of another delimiting strategy in the rt_info and use that strategy here for more robust code
                            std::string another_token; ss >> another_token;
                            token += std::string(" ") + another_token;
                            ov_any = ov::Any::make<std::string>(token);
                        }
                        else {
                            std::stringstream tok_ss{token};
                            ov_any.read(tok_ss);
                        }
                        bool is_parsed = maybe_parse_single_element(element_type, ov_any, array_elt, store);
                        OPENVINO_ASSERT(is_parsed);
                        parsed_array.push_back(array_elt);
                    }
                    kv.value.arr.n = parsed_array.size();
                    if (element_type == GGUF_TYPE_STRING) {
                        // string element has already been lifetime-extended during parsing
                        std::vector<char*> cstr_vector(parsed_array.size());
                        for (size_t cstr_idx = 0; cstr_idx < parsed_array.size(); cstr_idx++) {
                            cstr_vector[cstr_idx] = parsed_array[cstr_idx].str.data;
                        }
                        store.str_arr_storage.push_back(cstr_vector);
                        kv.value.arr.data = store.str_arr_storage.back().data();
                    }
                    else {
                        void* data_ptr = store.store_gguf_value_vector(parsed_array, element_type);
                        kv.value.arr.data = data_ptr;
                    }
                }
                kv_vector.push_back(kv);
            }

            auto token_types_kv_it = std::find_if(kv_vector.begin(), kv_vector.end(), [](const gguf_kv& val) { return std::string(val.key.data) == "tokenizer.ggml.token_type"; });
            if (token_types_kv_it != kv_vector.end()) {
                auto tokens_kv_it = std::find_if(kv_vector.begin(), kv_vector.end(), [](const gguf_kv& val) { return std::string(val.key.data) == "tokenizer.ggml.tokens"; });
                if (tokens_kv_it != kv_vector.end()) {
                    size_t expected_num_tokens = token_types_kv_it->value.arr.n;
                    size_t actual_num_tokens = tokens_kv_it->value.arr.n;
                    if (actual_num_tokens < expected_num_tokens) {
                        std::cout << "VSHAMPOR: detected wrong vocab serialization/deserialization (expected " << expected_num_tokens << " tokens, parsed " << actual_num_tokens << " from vocab), filling tokens with bogus values" << std::endl;
                        std::vector<char*> new_vocab;
                        // char** old_vocab_data_ptr = (char**) tokens_kv_it->value.arr.data;
                        // std::copy(old_vocab_data_ptr, old_vocab_data_ptr + actual_num_tokens, new_vocab.begin());
                        // size_t extra_tokens_needed = expected_num_tokens - actual_num_tokens;
                        size_t extra_tokens_needed = expected_num_tokens;
                        for (size_t tok_idx = 0; tok_idx < extra_tokens_needed; tok_idx++) {
                            std::stringstream ss;
                            ss << "invalid_token_" << tok_idx;
                            std::string new_token = ss.str();
                            store.kv_value_string_storage.push_back(new_token);
                            char* str_data_ptr = (char*) store.kv_value_string_storage.back().c_str();
                            new_vocab.push_back(str_data_ptr);
                        }
                        OPENVINO_ASSERT(new_vocab.size() == expected_num_tokens);
                        store.str_arr_storage.push_back(new_vocab);
                        tokens_kv_it->value.arr.data = (void*) store.str_arr_storage.back().data();
                        tokens_kv_it->value.arr.n = expected_num_tokens;
                    }
                }
            }

            // tensors
            OPENVINO_ASSERT(tensor_name_map.size() == tensor_shape_map.size());
            size_t n_tensors_in_rtinfo = tensor_name_map.size();
            std::cout << "VSHAMPOR: got request for " << n_tensors_in_rtinfo << " tensors from rt_info\n";

            std::vector<struct gguf_tensor_info> tensor_infos;
            std::vector<void*> tensor_data_ptrs;

            std::map<std::string, ov::Shape> parsed_weights_to_search_for;
            for (const auto& llama_name_and_rtinfo_name : tensor_name_map) {
                const std::string& llama_name = llama_name_and_rtinfo_name.first;
                const std::string& rtinfo_name = llama_name_and_rtinfo_name.second.as<std::string>();
                ov::Shape expected_shape = tensor_shape_map[llama_name].as<std::string>();
                parsed_weights_to_search_for[rtinfo_name] = expected_shape;
            }

            TensorWeightMatcher matcher{model, parsed_weights_to_search_for};
            std::unordered_map<std::string, std::shared_ptr<ov::op::v0::Constant>> matches = matcher.get_matches();
            std::unordered_map<std::string, std::shared_ptr<ov::op::v0::Constant>> llama_name_to_constant_node_map;
            for (const auto& entry : tensor_name_map) {
                const auto& llama_name = entry.first;
                const auto& rtinfo_name = entry.second.as<std::string>();
                llama_name_to_constant_node_map[llama_name] = matches[rtinfo_name];
            }
            std::cout << "VSHAMPOR: requested tensors map to " << llama_name_to_constant_node_map.size() << " tensors to search in model (shared tensors considered)\n";


            std::list<std::string> llama_name_storage;

            size_t n_tensors = 0;

            size_t offset = 0; // each tensor_info has to have a correct offset including padding, checked for in gguf_write_to_buf
            for (const auto& matched_weight_pair : llama_name_to_constant_node_map) {
                // Need to store the names in the list so that the passed c_str() pointers in tensor_infos to the llama names stay valid
                // until they get deepcopied in gguf/llama functions
                llama_name_storage.push_back(matched_weight_pair.first);
                const std::string& llama_name = llama_name_storage.back();

                auto weight_const_node_ptr = matched_weight_pair.second;
                auto weight_shape = weight_const_node_ptr->get_shape();

                // does hf-to-gguf invert all tensor dimensions with shapes > 1?
                auto expected_weight_shape = ov::Shape(expected_tensor_shapes_map[llama_name].as<std::string>());
                OPENVINO_ASSERT(expected_weight_shape.size() < GGML_MAX_DIMS);

                gguf_tensor_info info;

                info.type = GGML_TYPE_F32; // TODO (vshampor): better type assignment based on actual element type of the Constant node

                info.name.n = llama_name.length();
                info.name.data = (char*) llama_name.c_str();  // TODO (vshampor): either do this via const_cast, or will have to implement own structures for
                                                              // read-only data passing to llama_load_model_from_data
                info.n_dims = weight_shape.size();
                std::fill(std::begin(info.ne), std::begin(info.ne) + GGML_MAX_DIMS, (uint64_t) 1);

                // looks like GGUF expects inverse order of dimensions when compared to e.g. torch and actual row-major layout, see gguf.gguf_writer.GGUFWriter.add_tensor_info
                // in gguf python package
                std::copy(expected_weight_shape.rbegin(), expected_weight_shape.rend(), info.ne);

                void* data_ptr = (void*)(weight_const_node_ptr->get_data_ptr()); // TODO (vshampor): danger - casts `const` away
                                                                                 // also - the expected_weight_shape is in general different from actual ov::Tensor shape,
                                                                                 // in particular it may be transposed, so we actually need to set the pointers to shape-corrected
                                                                                 // tensor storage, which we don't do here - we are only preparing this data to get a convenient
                                                                                 // gguf_context object to reuse metadata (header) writing code, tensor data transpositions will be done during
                                                                                 // actual file write

                info.size = weight_const_node_ptr->get_byte_size();
                info.offset = offset;

                const size_t size_pad = GGML_PAD(info.size, GGUF_DEFAULT_ALIGNMENT);
                offset += size_pad;

                info.data = data_ptr;

                tensor_infos.push_back(info);
                tensor_data_ptrs.push_back(data_ptr);
                n_tensors++;
            }

            std::cout << "VSHAMPOR: found " << matches.size() << "/" << parsed_weights_to_search_for.size() << " tensors" << std::endl;

            gguf_init_params gguf_params;
            gguf_params.no_alloc = false;
            gguf_params.ctx = nullptr;

            m_gguf_ctx = gguf_init_from_data(n_tensors, tensor_infos.data(), n_kv, kv_vector.data(), tensor_data_ptrs.data(), gguf_params);

            std::shared_ptr<const LlamaCppPlugin> llama_plugin_ptr = std::dynamic_pointer_cast<const LlamaCppPlugin>(plugin);
            m_converted_gguf_file_name = llama_plugin_ptr->get_current_gguf_file_path();

            std::cout << "VSHAMPOR: output filename is  " << m_converted_gguf_file_name << std::endl;
            std::cout << "VSHAMPOR: writing metadata (GGUF header) " << std::endl;
            gguf_write_to_file(m_gguf_ctx, m_converted_gguf_file_name.c_str(), /* only_meta = */ true);

            std::map<std::string, TransposePermutation> transpose_permutations;

            for (const auto& llama_name_and_permutation : transpose_permutations_rtmap) {
                std::string permutation_str = llama_name_and_permutation.second.as<std::string>();
                std::stringstream ss(permutation_str);
                TransposePermutation permutation;
                bool is_ok = true;
                is_ok &= static_cast<bool>(ss >> permutation.first);
                is_ok &= static_cast<bool>(ss >> permutation.second);
                OPENVINO_ASSERT(is_ok, "failed to read permutation");
                transpose_permutations[llama_name_and_permutation.first] = permutation;
            }

            std::set<std::string> gemma_tensor_names_to_increment;
            // FIXME (vshampor): tried setting up commands for incrementing *_norm.weight values by 1 like it is done
            // during llama.cpp HF-to-GGUF export, but it seems that it isn't necessary and IR stores the incremented weights already
            // Is this due to constant folding?

            // for (const auto& llama_name_and_rtinfo_name : tensor_name_map) {
            //     const std::string& llama_name = llama_name_and_rtinfo_name.first;
            //     const std::string& rtinfo_name = llama_name_and_rtinfo_name.second.as<std::string>();
            //     std::string gemma_norm_suffix = "norm.weight";
            //     if (rtinfo_name.size() < gemma_norm_suffix.size()) continue;
            //     if (rtinfo_name.substr(rtinfo_name.size() - gemma_norm_suffix.size()) == gemma_norm_suffix) gemma_tensor_names_to_increment.insert(llama_name);
            // }

            std::cout << "VSHAMPOR: writing tensor data (blob with transpositions) " << std::endl;
            append_tensor_data_with_transpositions(m_converted_gguf_file_name, tensor_infos, tensor_data_ptrs, transpose_permutations, gemma_tensor_names_to_increment);
            std::cout << "VSHAMPOR: write finished." << m_converted_gguf_file_name << std::endl;

            std::cout << "VSHAMPOR: loading llama model from written file..." << std::endl;
            llama_model_params mparams = llama_model_default_params();
            mparams.n_gpu_layers = 99;
            m_llama_model_ptr = llama_load_model_from_file(m_converted_gguf_file_name.c_str(), mparams);
            llama_context_params cparams = llama_context_default_params();
            m_llama_ctx = llama_new_context_with_model(m_llama_model_ptr, cparams);

            std::cout << "VSHAMPOR: llama model loaded successfully..." << std::endl;
        }


        LlamaCppModel::LlamaCppModel(const std::shared_ptr<ov::Model>& ov_model, std::istream& input_stream, const std::shared_ptr<const IPlugin>& plugin) :
            ICompiledModel(ov_model, plugin) {
            num_tokens_processed_ptr = new size_t; // TODO (vshampor): hack, remove
            *num_tokens_processed_ptr = 0;
            std::shared_ptr<const LlamaCppPlugin> llama_plugin = std::dynamic_pointer_cast<const LlamaCppPlugin>(plugin);
            std::string current_file_path = llama_plugin->get_current_gguf_file_path();
            std::ofstream output_stream(current_file_path, std::ios::binary);
            output_stream << input_stream.rdbuf();


            std::cout << "VSHAMPOR: loading llama model from imported and re-written file..." << std::endl;
            llama_model_params mparams = llama_model_default_params();
            mparams.n_gpu_layers = 99;
            m_llama_model_ptr = llama_load_model_from_file(current_file_path.c_str(), mparams);
            llama_context_params cparams = llama_context_default_params();
            m_llama_ctx = llama_new_context_with_model(m_llama_model_ptr, cparams);
            std::cout << "VSHAMPOR: llama model loaded successfully from cache..." << std::endl;
        }

        LlamaCppModel::LlamaCppModel(const std::string& gguf_fname, const std::shared_ptr<const IPlugin>& plugin) :
            ICompiledModel(nullptr, plugin) {
            num_tokens_processed_ptr = new size_t; // TODO (vshampor): hack, remove
            *num_tokens_processed_ptr = 0;
            std::cout << "VSHAMPOR: loading llama model directly from GGUF... " << std::endl;
            llama_model_params mparams = llama_model_default_params();
            mparams.n_gpu_layers = 99;
            m_llama_model_ptr = llama_load_model_from_file(gguf_fname.c_str(), mparams);
            llama_context_params cparams = llama_context_default_params();
            m_llama_ctx = llama_new_context_with_model(m_llama_model_ptr, cparams);
            std::cout << "VSHAMPOR: llama model loaded successfully from GGUF..." << std::endl;

            auto input_ids = std::make_shared<ov::opset13::Parameter>(ov::element::Type_t::i64, ov::PartialShape({-1, -1}));
            auto fake_convert = std::make_shared<ov::opset13::Convert>(input_ids->output(0), ov::element::Type_t::f32);
            auto logits = std::make_shared<ov::opset13::Result>(fake_convert->output(0));

            ov::ParameterVector inputs{input_ids};

            std::vector<std::pair<std::string, ov::element::Type_t>> unused_names_in_order = { { "attention_mask", ov::element::Type_t::i64 },
                                                                                               { "position_ids", ov::element::Type_t::i64 },
                                                                                               { "beam_idx", ov::element::Type_t::i32 } };
            for (const auto& descr : unused_names_in_order) {
                auto unused_inp = std::make_shared<ov::opset13::Parameter>(descr.second, ov::PartialShape({-1, -1}));
                inputs.push_back(unused_inp);
            }

            m_model = std::make_shared<ov::Model>(logits, inputs, "fake_ov_model_for_io_specification");

            m_model->inputs()[0].set_names({"input_ids"});
            for (size_t i = 0; i < unused_names_in_order.size(); i++) {
                m_model->inputs()[i + 1].set_names({unused_names_in_order[i].first});
            }

            m_model->outputs()[0].set_names({"logits"});

            for (auto input : m_model->inputs()) {
                m_fake_inputs.emplace_back(input);
            }
            for (auto output : m_model->outputs()) {
                m_fake_outputs.emplace_back(output);
            }
        }


        void LlamaCppModel::export_model(std::ostream& output_stream) const {
            std::cout << "VSHAMPOR: exporting model" << std::endl;

            // FIXME (vshampor): it's a shame that loading a model from cache does not have an option to
            // actually keep the already loaded model from xml and not be forced to deserialize an ov::Model
            // representation from cache as well. As it stands, will need to write the whole IR into the cache entry
            // along with the GGUF file.
            //
            std::stringstream xmlFile, binFile;
            ov::pass::Serialize serializer(xmlFile, binFile);
            serializer.run_on_model(m_model);

            auto m_constants = binFile.str();
            auto m_model = xmlFile.str();

            auto dataSize = static_cast<std::uint64_t>(m_model.size());
            output_stream.write(reinterpret_cast<char*>(&dataSize), sizeof(dataSize));
            output_stream.write(m_model.c_str(), dataSize);

            dataSize = static_cast<std::uint64_t>(m_constants.size());
            output_stream.write(reinterpret_cast<char*>(&dataSize), sizeof(dataSize));
            output_stream.write(reinterpret_cast<char*>(&m_constants[0]), dataSize);


            std::ifstream in(m_converted_gguf_file_name, std::ios::binary);
            output_stream << in.rdbuf();
        }

        std::shared_ptr<const ov::Model> LlamaCppModel::get_runtime_model() const {
            OPENVINO_THROW_NOT_IMPLEMENTED("VSHAMPOR: Not Implemented");
        }

        void LlamaCppModel::set_property(const ov::AnyMap& properties) {
            std::cout << "VSHAMPOR: attempted to set_property (did nothing)";
        }

        ov::Any LlamaCppModel::get_property(const std::string& name) const {
            if (ov::supported_properties == name) {
                return decltype(ov::supported_properties)::value_type(std::vector<PropertyName>());
            }
            OPENVINO_THROW_NOT_IMPLEMENTED("VSHAMPOR: Not Implemented");
        }

        std::shared_ptr<ov::ISyncInferRequest> LlamaCppModel::create_sync_infer_request() const {
             return std::make_shared<LlamaCppSyncInferRequest>(std::static_pointer_cast<const LlamaCppModel>(shared_from_this()));
        }

         const std::vector<ov::Output<const ov::Node>>& LlamaCppModel::inputs() const {
             return m_fake_inputs;
         };
         const std::vector<ov::Output<const ov::Node>>& LlamaCppModel::outputs() const {
             return m_fake_outputs;
         };
    }
}  // namespace ov
