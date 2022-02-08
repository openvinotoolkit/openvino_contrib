// Copyright (C) 2021-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <algorithm>
#include <cmath>
#include <cuda/runtime.hpp>
#include <map>
#include <vector>

#include "cuda_type_traits.hpp"
#include "tensor_helpers.hpp"

namespace CUDAPlugin {
namespace kernel {

template <typename TDataType>
struct __align__(1) DetectionOutputResult {
    TDataType img;
    TDataType cls;
    TDataType conf;
    TDataType xmin;
    TDataType ymin;
    TDataType xmax;
    TDataType ymax;
};

class DetectionOutput {
public:
    struct Attrs {
        enum CodeType {
            Caffe_PriorBoxParameter_CORNER = 0,
            Caffe_PriorBoxParameter_CENTER_SIZE,
        };

        size_t num_images;
        size_t prior_size;
        size_t num_priors;
        size_t num_loc_classes;
        size_t offset;
        size_t num_results;
        size_t out_total_size;

        int num_classes;
        int background_label_id = 0;
        int top_k = -1;
        bool variance_encoded_in_target = false;
        int keep_top_k{};
        CodeType code_type = Caffe_PriorBoxParameter_CORNER;
        bool share_location = true;
        float nms_threshold;
        float confidence_threshold = 0;
        bool clip_after_nms = false;
        bool clip_before_nms = false;
        bool decrease_label_id = false;
        bool normalized = false;
        size_t input_height = 1;
        size_t input_width = 1;
        float objectness_score = 0;
    };

    enum {
        kDetectionOutputAttrsWBIdx = 0,
        kLocationsWBIdx,
        kConfPredsWBIdx,
        kPriorBboxesWBIdx,
        kPriorVariancesWBIdx,
        kDecodeBboxesWBIdx,
        kTempDecodeBboxesWBIdx,
        kTempScorePerClassPrioIdxs0WBIdx,
        kTempScorePerClassPrioIdxs1WBIdx,
        kPrioBoxIdxsByClassWBIdx,
        kNumDetectionsWBIdx,
        kNumRequiredWB,
        kArmLocationsWBIdx = kNumRequiredWB,
        kNumOptionalWB,
    };

    DetectionOutput(Type_t element_type,
                    size_t max_threads_per_block,
                    size_t location_size,
                    size_t confidence_size,
                    size_t priors_size,
                    size_t arm_confidence_size,
                    size_t arm_location_size,
                    size_t result_size,
                    Attrs attrs);

    void operator()(cudaStream_t stream,
                    const void* locationPtr,
                    const void* confidencePtr,
                    const void* priorsPtr,
                    const void* armLocationPtr,
                    const void* armConfidencePtr,
                    void* const* _workbuffers,
                    void* result) const;
    std::vector<size_t> getMutableWorkbufferSize() const;

protected:
    template <typename TDataType>
    std::vector<size_t> getMutableWorkbufferSize() const;
    template <typename TDataType>
    void call(const CUDA::Stream& stream,
              CUDA::DevicePointer<const void*> location,
              CUDA::DevicePointer<const void*> confidence,
              CUDA::DevicePointer<const void*> priors,
              const void* armConfidence,
              const void* armLocation,
              std::vector<CUDA::DevicePointer<void*>> mutableWorkbuffers,
              CUDA::DevicePointer<void*> result) const;

private:
    Type_t element_type_{};
    Attrs attrs_;

    size_t max_threads_per_block_;

    size_t location_size_;
    size_t confidence_size_;
    size_t priors_size_;
    size_t arm_confidence_size_;
    size_t arm_location_size_;
    size_t result_size_;
};

}  // namespace kernel
}  // namespace CUDAPlugin
