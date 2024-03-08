#ifndef LLAMA_CPP_INFER_REQUEST_HPP
#define LLAMA_CPP_INFER_REQUEST_HPP

#include "openvino/openvino.hpp"
#include "compiled_model.hpp"

namespace ov {
namespace llama_cpp_plugin {

class LlamaCppSyncInferRequest : public ISyncInferRequest {
public:
    explicit LlamaCppSyncInferRequest(const std::shared_ptr<const LlamaCppModel>& compiled_model);
    // explicit LlamaCppSyncInferRequest(const std::shared_ptr<const LlamaCppModel>& compiled_model): ov::ISyncInferRequest(compiled_model) {
    //         std::cout << "VSHAMPOR: infer request ctor called\n";
    //     }
    virtual ~LlamaCppSyncInferRequest() {};

    virtual void set_tensors_impl(const ov::Output<const ov::Node> port,
                                  const std::vector<ov::SoPtr<ov::ITensor>>& tensors) override;

    virtual void infer() override;
    virtual std::vector<ov::ProfilingInfo> get_profiling_info() const override;
    virtual std::vector<ov::SoPtr<ov::IVariableState>> query_state() const override;
private:
    std::shared_ptr<const LlamaCppModel> m_compiled_model_ptr;
};

}  // namespace LlamaCppPlugin
};  // namespace ov

#endif /* LLAMA_CPP_INFER_REQUEST_HPP */
