#include "openvino/openvino.hpp"
#include <cstring>

int main(int argc, char* argv[]) {
    ov::Core core;
    core.set_property(ov::cache_dir("/tmp/my_cache_dir"));
    std::string model_path = "/home/vshampor/work/optimum-intel/ov_model/openvino_model.xml";

    std::cout << "VSHAMPOR: reading model\n";
    std::shared_ptr<ov::Model> model = core.read_model(model_path);

    std::cout << "VSHAMPOR: compiling model\n";
    ov::CompiledModel compiled_model = core.compile_model(model, "LLAMA_CPP");

    std::cout << "VSHAMPOR: compiled successfully\n";

    std::cout << "VSHAMPOR: creating infer request\n";
    ov::InferRequest infer_request = compiled_model.create_infer_request();
    std::cout << "VSHAMPOR: infer request created\n";

    // const ov::Output<const ov::Node>& input = compiled_model.input();
    // std::cout << "VSHAMPOR: got input\n";
    auto inputs = compiled_model.inputs();
    std::cout << "VSHAMPOR: model has " << inputs.size() << " inputs\n";
    for (const auto& input: inputs) {
        std::cout << input.get_node()->get_friendly_name() << std::endl;
    }

    for (size_t i = 0; i < inputs.size(); i++) {
        const auto& curr_input = inputs[i];
        auto shape = curr_input.get_partial_shape();
        if (shape.is_dynamic()) {
            std::cout << "VSHAMPOR: processing input " << i << " with a dynamic shape of " << shape.to_string() << std::endl;
            ov::Rank r = shape.rank();
            if (r.get_length() == 2) {
                ov::Tensor input_tensor{curr_input.get_element_type(), ov::Shape({1, 128})};
                int64_t* data_ptr = input_tensor.data<int64_t>();
                // fill with something
                for (size_t elt_idx = 0; elt_idx < input_tensor.get_size(); elt_idx++) {
                    data_ptr[elt_idx] = 42;
                }
                infer_request.set_input_tensor(i, input_tensor);
            }
            else {  // past_key_values
                ov::Tensor input_tensor{curr_input.get_element_type(), ov::Shape({1, 12, 128, 64})};
                infer_request.set_input_tensor(i, input_tensor);
            }
        }
        else {
            std::cout << "VSHAMPOR: processing input " << i << " with a non-dynamic shape of " << shape.to_string() << std::endl;
            ov::Tensor input_tensor{curr_input.get_element_type(), curr_input.get_shape()};
            infer_request.set_input_tensor(i, input_tensor);
        }
    }
    std::cout << "VSHAMPOR: successfully set input tensor\n";

    infer_request.infer();
    std::cout << "VSHAMPOR: inferred successfully\n";

    ov::Tensor output = infer_request.get_tensor("logits");
    std::cout << "VSHAMPOR: got output tensor, shape " << output.get_shape().to_string() << std::endl;

    size_t n_output_elts = 10;
    std::cout << "VSHAMPOR: first " << n_output_elts << " elements are:" << std::endl;

    float* output_data_ptr = output.data<float>();
    for (size_t elt_idx = 0; elt_idx < n_output_elts; elt_idx++) {
        std::cout << output_data_ptr[elt_idx] << " ";
    }

    std::cout << std::endl;
    return 0;
}
