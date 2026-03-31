// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "plugin/gfx_property_lists.hpp"

#include "openvino/gfx_plugin/properties.hpp"
#include "openvino/runtime/internal_properties.hpp"

namespace ov {
namespace gfx_plugin {

std::vector<ov::PropertyName> gfx_plugin_supported_properties() {
    std::vector<ov::PropertyName> ro{
        ov::available_devices,
        ov::supported_properties,
        ov::internal::supported_properties,
        ov::device::full_name,
        ov::device::architecture,
        ov::device::type,
        ov::device::capabilities,
        ov::execution_devices,
        ov::range_for_async_infer_requests,
    };
    std::vector<ov::PropertyName> rw{
        ov::device::id,
        ov::cache_dir,
        ov::enable_profiling,
        ov::PropertyName{kGfxProfilingLevelProperty, ov::PropertyMutability::RW},
        ov::PropertyName{kGfxEnableFusionProperty, ov::PropertyMutability::RW},
        ov::hint::performance_mode,
        ov::hint::num_requests,
        ov::hint::execution_mode,
        ov::hint::inference_precision,
        ov::num_streams,
        ov::inference_num_threads,
        ov::log::level,
        ov::PropertyName{kGfxBackendProperty, ov::PropertyMutability::RW},
        ov::PropertyName{"PERF_COUNT", ov::PropertyMutability::RW},
    };
    std::vector<ov::PropertyName> supported;
    supported.reserve(ro.size() + rw.size());
    supported.insert(supported.end(), ro.begin(), ro.end());
    supported.insert(supported.end(), rw.begin(), rw.end());
    return supported;
}

std::vector<ov::PropertyName> gfx_internal_supported_properties() {
    return {
        ov::PropertyName{ov::internal::caching_properties.name(), ov::PropertyMutability::RO},
        ov::PropertyName{ov::internal::exclusive_async_requests.name(), ov::PropertyMutability::RW},
        ov::PropertyName{ov::internal::threads_per_stream.name(), ov::PropertyMutability::RW},
        ov::PropertyName{ov::internal::compiled_model_runtime_properties.name(), ov::PropertyMutability::RO},
        ov::PropertyName{ov::internal::compiled_model_runtime_properties_supported.name(),
                         ov::PropertyMutability::RO},
        ov::PropertyName{ov::internal::cache_header_alignment.name(), ov::PropertyMutability::RO},
    };
}

std::vector<ov::PropertyName> gfx_caching_properties() {
    return {
        ov::PropertyName{ov::device::architecture.name(), ov::PropertyMutability::RO},
        ov::PropertyName{ov::device::id.name(), ov::PropertyMutability::RW},
        ov::PropertyName{kGfxBackendProperty, ov::PropertyMutability::RW},
        ov::PropertyName{ov::hint::inference_precision.name(), ov::PropertyMutability::RW},
    };
}

std::vector<ov::PropertyName> gfx_compiled_model_default_ro_properties() {
    return {
        ov::model_name,
        ov::supported_properties,
        ov::execution_devices,
        ov::loaded_from_cache,
        ov::optimal_number_of_infer_requests,
    };
}

std::vector<ov::PropertyName> gfx_compiled_model_supported_properties() {
    auto props = gfx_compiled_model_default_ro_properties();
    props.push_back(ov::PropertyName{ov::hint::inference_precision.name(), ov::PropertyMutability::RW});
    props.push_back(ov::PropertyName{ov::enable_profiling.name(), ov::PropertyMutability::RW});
    props.push_back(ov::PropertyName{ov::cache_dir.name(), ov::PropertyMutability::RW});
    props.push_back(ov::PropertyName{"PERF_COUNT", ov::PropertyMutability::RW});
    props.push_back(ov::PropertyName{kGfxProfilingLevelProperty, ov::PropertyMutability::RW});
    props.push_back(ov::PropertyName{kGfxEnableFusionProperty, ov::PropertyMutability::RW});
    props.push_back(ov::PropertyName{kGfxBackendProperty, ov::PropertyMutability::RO});
    props.push_back(ov::PropertyName{kGfxProfilingReportProperty, ov::PropertyMutability::RO});
    props.push_back(ov::PropertyName{kGfxMemStatsProperty, ov::PropertyMutability::RO});
    return props;
}

}  // namespace gfx_plugin
}  // namespace ov
