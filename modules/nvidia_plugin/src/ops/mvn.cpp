// Copyright (C) 2021-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "mvn.hpp"

#include "openvino/core/shape_util.hpp"
#include "openvino/core/type.hpp"
#include "openvino/op/constant.hpp"
#include "openvino/op/mvn.hpp"

#include <cuda/descriptor_utils.hpp>
#include <cuda_operation_registry.hpp>

#include "converters.hpp"

namespace ov {
namespace nvidia_gpu {

inline bool isTypeSupported(cudnnDataType_t type) {
    switch (type) {
        case CUDNN_DATA_FLOAT:
        case CUDNN_DATA_DOUBLE:
        case CUDNN_DATA_HALF:
        case CUDNN_DATA_INT8:
            return true;
        default:
            return false;
    }
}

MvnOp::MvnOp(const CreationContext& context,
             const ov::Node& node,
             IndexCollection&& inputIds,
             IndexCollection&& outputIds)
    : OperationCuDnn{context, node, move(inputIds), move(outputIds)},
      mvn_op_v1_{dynamic_cast<const ov::op::v0::MVN*>(&node)},
      mvn_op_v6_{dynamic_cast<const ov::op::v6::MVN*>(&node)},
      version_{validateAndGetVersion(node)},
      normalize_variance_{version_ == MvnV1 ? mvn_op_v1_->get_normalize_variance()
                                            : mvn_op_v6_->get_normalize_variance()},
      epsilon_{version_ == MvnV1 ? mvn_op_v1_->get_eps() : mvn_op_v6_->get_eps()},
      eps_mode_{version_ == MvnV1 ? ov::op::MVNEpsMode::INSIDE_SQRT : mvn_op_v6_->get_eps_mode()},
      comp_type_{ov::nvidia_gpu::convertDataType<cudnnDataType_t>(node.get_input_element_type(0))},
      op_desc_type_{comp_type_ != CUDNN_DATA_DOUBLE ? CUDNN_DATA_FLOAT : CUDNN_DATA_DOUBLE},
      reduce_mean_desc_{op_desc_type_},
      sub_desc_(CUDA::DnnOpTensorDescriptor{}.set(
          cudnnOpTensorOp_t::CUDNN_OP_TENSOR_ADD, op_desc_type_, cudnnNanPropagation_t::CUDNN_PROPAGATE_NAN)),
      mul_desc_(CUDA::DnnOpTensorDescriptor{}.set(
          cudnnOpTensorOp_t::CUDNN_OP_TENSOR_MUL, op_desc_type_, cudnnNanPropagation_t::CUDNN_PROPAGATE_NAN)),
      tensor_desc_{CUDA::makeInputDnnTensorDescr(node, 0)},
      shape_{node.get_input_shape(0)},
      reduced_shape_{makeReducedShape(node)},
      reduced_tensor_desc_{makeReducedTensorDescriptor(node)},
      reduce_workspace_size_{reduceWorkSpaceSizeCompute(context)} {
    if (!isTypeSupported(op_desc_type_)) {
        throw_ov_exception(fmt::format("MvnOp: unsupported argument type: {}", toString(op_desc_type_)));
    }
    if (!reduced_shape_.empty()) {
        size_t size = shape_size(reduced_shape_);
        unsigned max_threads_per_block = context.device().props().maxThreadsPerBlock;
        unsigned blocks_number = 1 + size / max_threads_per_block;
        unsigned threads_per_block = (blocks_number == 1) ? size : max_threads_per_block;
        variance_normalization_factor_kernel_ = kernel::VarianceNormalizationFactor(
            blocks_number,
            threads_per_block,
            epsilon_,
            size,
            convertDataType<ov::nvidia_gpu::kernel::Type_t>(node.get_input_element_type(0)),
            eps_mode_ == ov::op::MVNEpsMode::INSIDE_SQRT);
    }
}

void MvnOp::Execute(const InferenceRequestContext& context,
                    Inputs inputTensors,
                    Outputs outputTensors,
                    const Workbuffers& workbuffers) const {
    Context opContext{context, workbuffers, *this};
    if (reduced_shape_.empty()) {
        // this is not documented case, but valid case in reference implementation and tests are present for it
        opContext.subtract(
            {tensor_desc_, inputTensors[0]}, {tensor_desc_, inputTensors[0]}, {tensor_desc_, outputTensors[0]});
        return;
    }
    auto reducedTensor = getReducedTensorBuffer(workbuffers);
    opContext.reduceMean({tensor_desc_, inputTensors[0]}, {reduced_tensor_desc_, reducedTensor});
    opContext.subtract({tensor_desc_, inputTensors[0]},
                       {reduced_tensor_desc_, reducedTensor.cast<const void*>()},
                       {tensor_desc_, outputTensors[0]});
    if (!normalize_variance_) return;
    auto tmpTensor = getTmpTensorBuffer(workbuffers);
    opContext.multiply({tensor_desc_, outputTensors[0].cast<const void*>()},
                       {tensor_desc_, outputTensors[0].cast<const void*>()},
                       {tensor_desc_, tmpTensor});
    opContext.reduceMean({tensor_desc_, tmpTensor.cast<const void*>()}, {reduced_tensor_desc_, reducedTensor});
    opContext.computeVarianceNormalizationFactor({reduced_tensor_desc_, reducedTensor});
    opContext.multiply({tensor_desc_, outputTensors[0].cast<const void*>()},
                       {reduced_tensor_desc_, reducedTensor.cast<const void*>()},
                       {tensor_desc_, outputTensors[0]});
}

CudaGraphCompatibility MvnOp::GetCudaGraphCompatibility() const { return CudaGraphCompatibility::FULL; }

void MvnOp::Context::reduceMean(ConstTensor input, Tensor output) {
    context.getThreadContext().dnnHandle().reduceTensor(op.reduce_mean_desc_,
                                                        op.getReduceWorkspaceBuffer(workbuffers),
                                                        CUDA::DnnScaleFactorOne{op.comp_type_},
                                                        input.descriptor,
                                                        input.data,
                                                        CUDA::DnnScaleFactorZero{op.comp_type_},
                                                        output.descriptor,
                                                        output.data);
}

void MvnOp::Context::subtract(ConstTensor lhs, ConstTensor rhs, Tensor output) {
    context.getThreadContext().dnnHandle().opTensor(op.sub_desc_,
                                                    op.dOne,
                                                    lhs.descriptor,
                                                    lhs.data.get(),
                                                    op.dMinusOne,
                                                    rhs.descriptor,
                                                    rhs.data.get(),
                                                    op.dZero,
                                                    output.descriptor,
                                                    output.data.get());
}

void MvnOp::Context::multiply(ConstTensor lhs, ConstTensor rhs, Tensor output) {
    context.getThreadContext().dnnHandle().opTensor(op.mul_desc_,
                                                    op.dOne,
                                                    lhs.descriptor,
                                                    lhs.data.get(),
                                                    op.dOne,
                                                    rhs.descriptor,
                                                    rhs.data.get(),
                                                    op.dZero,
                                                    output.descriptor,
                                                    output.data.get());
}

void MvnOp::Context::computeVarianceNormalizationFactor(Tensor in_out) {
    OPENVINO_ASSERT(op.variance_normalization_factor_kernel_);
    (*op.variance_normalization_factor_kernel_)(context.getThreadContext().stream().get(), in_out.data.get());
}

MvnOp::MvnVersion MvnOp::validateAndGetVersion(const ov::Node& node) {
    auto mvnOp_v1 = dynamic_cast<const ov::op::v0::MVN*>(&node);
    auto mvnOp_v6 = dynamic_cast<const ov::op::v6::MVN*>(&node);
    MvnVersion version;
    OPENVINO_ASSERT(mvnOp_v1 || mvnOp_v6);
    if (mvnOp_v1) {
        version = MvnV1;
        OPENVINO_ASSERT(node.get_input_size() == 1);
        if (mvnOp_v1->get_eps() <= 0) {
            throw_ov_exception(
                fmt::format("The eps attribute of the MVN-1 operation must be positive number, but value is {}.",
                            mvnOp_v1->get_eps()));
        }
    } else {
        version = MvnV6;
        OPENVINO_ASSERT(node.get_input_size() == 2);
        if (mvnOp_v6->get_eps() <= 0) {
            throw_ov_exception(
                fmt::format("The eps attribute of the MVN-6 operation must be positive number, but value is {}.",
                            mvnOp_v6->get_eps()));
        }
        if (ov::as_type_ptr<op::v0::Constant>(node.get_input_node_shared_ptr(1)) == nullptr) {
            throw_ov_exception("The nvidia_gpu MVN-6 operation implemented only for constant axes input.");
        }
    }
    if (!node.get_input_partial_shape(0).rank().is_static()) {
        throw_ov_exception("For not static input shape the MVN-1 operation was not implemented.");
    }
    OPENVINO_ASSERT(node.get_output_size() == 1);
    auto inputShape = node.get_input_shape(0);
    auto outputShape = node.get_output_shape(0);
    OPENVINO_ASSERT(inputShape == outputShape);
    if (version == MvnV6) {
        auto inputAxesShape = node.get_input_shape(1);
        OPENVINO_ASSERT(inputAxesShape.size() == 1);
        OPENVINO_ASSERT(inputAxesShape[0] <= inputShape.size());
    }
    OPENVINO_ASSERT(node.get_input_element_type(0) == node.get_output_element_type(0));
    const size_t max_shape_size = 5;  // cudnnOpTensor operation limit. See note here
                                      // https://docs.nvidia.com/deeplearning/cudnn/api/index.html#cudnnOpTensor
    if (outputShape.size() > max_shape_size) {
        throw_ov_exception(
            fmt::format("ov::nvidia_gpu::MvnOp: the tensor shape size ({}) is exceeded maximum supported value of {}.",
                        outputShape.size(),
                        max_shape_size));
    }
    return version;
}

size_t MvnOp::reduceWorkSpaceSizeCompute(const CreationContext& context) {
    if (!reduced_shape_.empty())
        return context.dnnHandle().getReductionWorkspaceSize(reduce_mean_desc_, tensor_desc_, reduced_tensor_desc_);
    return 0;
}

ov::Shape MvnOp::makeReducedShape(const ov::Node& node) {
    if (version_ == MvnV1) {
        auto reducedShape = node.get_input_shape(0);
        if (mvn_op_v1_->get_reduction_axes().empty()) {
            return {};
        } else {
            for (auto& reductionAxisIndex : mvn_op_v1_->get_reduction_axes()) {
                OPENVINO_ASSERT(reductionAxisIndex < reducedShape.size(), "Node name: ", GetName());
                reducedShape[reductionAxisIndex] = 1;
            }
        }
        return reducedShape;
    }
    if (version_ == MvnV6) {
        const auto signed_axes =
            ov::as_type_ptr<op::v0::Constant>(node.get_input_node_shared_ptr(1))->cast_vector<int64_t>();
        auto reducedShape = node.get_input_shape(0);
        ov::AxisSet axes;
        for (auto v : signed_axes) {
            auto size = static_cast<int64_t>(reducedShape.size());
            if (v >= size || v < -size) {
                throw_ov_exception(
                    fmt::format("ov::nvidia_gpu::MVN-6: the axes entry ({}) out of range [{}; {}].", v, -size, size - 1));
            }
            axes.emplace(static_cast<size_t>((v + size) % size));
        }
        reducedShape = ov::util::reduce_keep_dims(reducedShape, axes);
        if (reducedShape == node.get_input_shape(0)) return {};
        return reducedShape;
    }
    return {};
}

CUDA::DnnTensorDescriptor MvnOp::makeReducedTensorDescriptor(const ov::Node& node) {
    if (reduced_shape_.empty()) return {};
    return CUDA::makeDnnTensorDescr(node.get_input_element_type(0), reduced_shape_);
}

OPERATION_REGISTER(MvnOp, MVN);
}  // namespace nvidia_gpu
}  // namespace ov
