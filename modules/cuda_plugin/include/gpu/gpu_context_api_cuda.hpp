// Copyright (C) 2018-2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

/**
 * @brief a header that defines wrappers for internal GPU plugin-specific
 * CUDA context and CUDA shared memory blobs
 *
 * @file gpu_context_api_cuda.hpp
 */
#pragma once

#include <ie_remote_context.hpp>
#include <memory>
#include <string>
#include <utility>

#include "gpu/details/gpu_context_helpers.hpp"
#include "gpu/gpu_params.hpp"

namespace InferenceEngine {

namespace gpu {
/**
 * @brief This class represents an abstraction for GPU plugin remote context
 * which is shared with OpenCL context object.
 * The plugin object derived from this class can be obtained either with
 * GetContext() method of Executable network or using CreateContext() Core call.
 */
class CudaContext : public RemoteContext, public details::param_map_obj_getter {
public:
    /**
     * @brief A smart pointer to the ClContext object
     */
    using Ptr = std::shared_ptr<CudaContext>;
    using WeakPtr = std::weak_ptr<CudaContext>;

    // TODO: Add additional functions
};

}  // namespace gpu

}  // namespace InferenceEngine
