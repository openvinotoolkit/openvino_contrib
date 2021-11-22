#include "rnn_sequence_components.hpp"

#include "constant_factory.hpp"
#include "ngraph/shape.hpp"

namespace CUDAPlugin::RNN::Details {

TransposeTensorAdapterBase::TransposeTensorAdapterBase(cudaDataType_t element_type,
                                                       size_t element_size,
                                                       const std::vector<int64_t>& src_shape,
                                                       const std::vector<int64_t>& dst_shape,
                                                       const std::vector<int>& mode)
    : element_type_{element_type},
      element_size_{element_size},
      src_shape_{src_shape},
      dst_shape_{dst_shape},
      src_mode_(mode.size()),
      dst_mode_{mode} {
    std::iota(src_mode_.begin(), src_mode_.end(), 0);
    const auto num_elements = ngraph::shape_size(src_shape_);
    Expects(num_elements > 0);
    Expects(num_elements == ngraph::shape_size(dst_shape_));
    Expects(src_shape_.size() == dst_shape_.size());
    Expects(src_shape_.size() == src_mode_.size());
    Expects(src_mode_.size() == dst_mode_.size());
}

void TransposeTensorAdapterBase::requestWorkbuffer(std::vector<size_t>& workbuffers_sizes) {
    workbuffer_.addRequest(workbuffers_sizes, ngraph::shape_size(src_shape_) * element_size_);
}

void* TransposeTensorAdapterBase::dnnApiPtr(const std::vector<Workbuffers::mutable_buffer>& mutable_buffers) const {
    return workbuffer_.requiredPtr(mutable_buffers);
}

void TransposeTensorAdapterBase::execute(const InferenceRequestContext& context, const void* src, void* dst) const {
    cutensorTensorDescriptor_t src_desc{}, dst_desc{};
    initCuTensorDescriptor(context.getThreadContext().cuTensorHandle(), src_shape_, src_desc);
    initCuTensorDescriptor(context.getThreadContext().cuTensorHandle(), dst_shape_, dst_desc);
    throwIfError(::cutensorPermutation(&context.getThreadContext().cuTensorHandle().get(),
                                       &NumericConst<constants::one>(element_type_),
                                       src,
                                       &src_desc,
                                       src_mode_.data(),
                                       dst,
                                       &dst_desc,
                                       dst_mode_.data(),
                                       element_type_,
                                       context.getThreadContext().stream().get()));
}

void TransposeTensorAdapterBase::initCuTensorDescriptor(const CUDA::CuTensorHandle& handle,
                                                        const std::vector<int64_t>& shape,
                                                        cutensorTensorDescriptor_t& desc) const {
    std::vector<int64_t> strides;
    strides.reserve(shape.size());
    for (size_t i = 0; i < shape.size(); i++) strides.push_back(ngraph::row_major_stride(shape, i));
    throwIfError(::cutensorInitTensorDescriptor(
        &handle.get(), &desc, shape.size(), shape.data(), strides.data(), element_type_, CUTENSOR_OP_IDENTITY));
}

void TransposeInputTensorAdapter::execute(const InferenceRequestContext& context,
                                          CUDA::DevicePointer<const void*> input,
                                          const std::vector<Workbuffers::mutable_buffer>& dst) const {
    TransposeTensorAdapterBase::execute(context, input.get(), workbuffer_.requiredPtr(dst));
}

void TransposeOutputTensorAdapter::execute(const InferenceRequestContext& context,
                                           const std::vector<Workbuffers::mutable_buffer>& src,
                                           CUDA::DevicePointer<void*> output) const {
    TransposeTensorAdapterBase::execute(context, workbuffer_.requiredPtr(src), output.get());
}

}  // namespace CUDAPlugin::RNN::Details
