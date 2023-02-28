// Copyright (C) 2022-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "details/type_validator.hpp"
#include "interpolate_base.hpp"

namespace ov {
namespace nvidia_gpu {
namespace kernel {

InterpolateBase::InterpolateBase(Type_t elemenent_type) {
    using InterpolateTypesSwitch = ElementTypesSwitch<Type_t::f32,
#ifdef CUDA_HAS_BF16_TYPE
                                                      Type_t::bf16,
#endif
                                                      Type_t::f16,
                                                      Type_t::i8,
                                                      Type_t::u8>;
    TypeValidator<InterpolateTypesSwitch>::check(elemenent_type);
}

}  // namespace kernel
}  // namespace nvidia_gpu
}  // namespace ov
