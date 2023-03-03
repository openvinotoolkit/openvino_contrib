// Copyright (C) 2021-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <cuda_runtime_api.h>
#include <fmt/format.h>

#include <cuda/float16.hpp>
#include <cuda/math.cuh>
#include <cuda/stl/algorithms/sort.cuh>
#include <cuda/stl/array.cuh>
#include <cuda/stl/atomic.cuh>
#include <cuda/stl/mdspan.cuh>
#include <cuda/stl/mdvector.cuh>
#include <cuda/stl/span.cuh>

#include "details/error.hpp"
#include "details/type_validator.hpp"
#include "detection_output.hpp"

namespace ov {
namespace nvidia_gpu {
namespace kernel {

namespace cumath = CUDA::math;

__device__ unsigned get_image_idx() { return blockIdx.x; }
__device__ int get_image_idx(const DetectionOutput::Attrs& attrs) {
    const int i = get_image_idx() * blockDim.x + threadIdx.x;
    if (i >= attrs.num_images) {
        return -1;
    }
    return i;
}
__device__ int get_class_idx() { return blockIdx.y; }
__device__ int get_class_idx(const DetectionOutput::Attrs& attrs) {
    const int i = get_class_idx() * blockDim.x + threadIdx.x;
    if (i >= attrs.num_classes) {
        return -1;
    }
    return i;
}
__device__ int get_prio_idx() { return blockIdx.z; }
__device__ int get_prio_idx(const DetectionOutput::Attrs& attrs) {
    const int i = get_prio_idx() * blockDim.x + threadIdx.x;
    if (i >= attrs.num_priors) {
        return -1;
    }
    return i;
}

__device__ int get_global_work_group_size() { return gridDim.z; }
__device__ int get_global_work_item_id() { return blockIdx.z; }

__device__ int get_work_group_size() { return blockDim.x; }
__device__ int get_work_item_id() { return threadIdx.x; }

template <typename TDataType, typename T>
struct SortScorePairDescend {
    __device__ bool operator()(const CUDA::Pair<TDataType, T>& pair1, const CUDA::Pair<TDataType, T>& pair2) {
        return pair1.first > pair2.first;
    }
};

template <typename TDataType>
struct NormalizedBBox {
    TDataType xmin{};
    TDataType ymin{};
    TDataType xmax{};
    TDataType ymax{};

    template <typename TNormalizedBBox>
    __device__ NormalizedBBox& operator=(const TNormalizedBBox& bbox) {
        xmin = bbox.xmin;
        ymin = bbox.ymin;
        xmax = bbox.xmax;
        ymax = bbox.ymax;
        return *this;
    }

    template <typename TOtherNormalizedBBox>
    __device__ bool operator==(const TOtherNormalizedBBox& other_bbox) const {
        return this->xmin == other_bbox.xmin && this->ymin == other_bbox.ymin && this->xmax == other_bbox.xmax &&
               this->ymax == other_bbox.ymax;
    }

    template <typename TOtherNormalizedBBox>
    __device__ bool operator!=(const TOtherNormalizedBBox& other_bbox) const {
        return !(this->template operator==(other_bbox));
    }
};

template <typename TDataType>
struct InvalidNormalizedBBox {
    TDataType xmin{-1.0};
    TDataType ymin{-1.0};
    TDataType xmax{-1.0};
    TDataType ymax{-1.0};
};

template <typename TDataType>
__device__ void decode_bbox(const DetectionOutput::Attrs& attrs,
                            const NormalizedBBox<TDataType>& priorBboxes,
                            const NormalizedBBox<TDataType>& bbox,
                            NormalizedBBox<TDataType>& decodeBbox) {
    TDataType priorXmin = priorBboxes.xmin;
    TDataType priorYmin = priorBboxes.ymin;
    TDataType priorXmax = priorBboxes.xmax;
    TDataType priorYmax = priorBboxes.ymax;

    if (!attrs.normalized) {
        priorXmin /= static_cast<double>(attrs.input_width);
        priorYmin /= static_cast<double>(attrs.input_height);
        priorXmax /= static_cast<double>(attrs.input_width);
        priorYmax /= static_cast<double>(attrs.input_height);
    }

    if (attrs.code_type == DetectionOutput::Attrs::CodeType::Caffe_PriorBoxParameter_CORNER) {
        decodeBbox.xmin = priorXmin + bbox.xmin;
        decodeBbox.ymin = priorYmin + bbox.ymin;
        decodeBbox.xmax = priorXmax + bbox.xmax;
        decodeBbox.ymax = priorYmax + bbox.ymax;
    } else if (attrs.code_type == DetectionOutput::Attrs::CodeType::Caffe_PriorBoxParameter_CENTER_SIZE) {
        const TDataType priorWidth = priorXmax - priorXmin;
        const TDataType priorHeight = priorYmax - priorYmin;
        const TDataType priorCenterX = (priorXmin + priorXmax) / TDataType{2.0};
        const TDataType priorCenterY = (priorYmin + priorYmax) / TDataType{2.0};
        const TDataType decodeBboxCenterX = bbox.xmin * priorWidth + priorCenterX;
        const TDataType decodeBboxCenterY = bbox.ymin * priorHeight + priorCenterY;
        const TDataType decodeBboxWidth = cumath::exp(bbox.xmax) * priorWidth;
        const TDataType decodeBboxHeight = cumath::exp(bbox.ymax) * priorHeight;
        decodeBbox.xmin = decodeBboxCenterX - decodeBboxWidth / TDataType{2.0};
        decodeBbox.ymin = decodeBboxCenterY - decodeBboxHeight / TDataType{2.0};
        decodeBbox.xmax = decodeBboxCenterX + decodeBboxWidth / TDataType{2.0};
        decodeBbox.ymax = decodeBboxCenterY + decodeBboxHeight / TDataType{2.0};
    }
}

template <typename TDataType>
__device__ void decode_bbox(const DetectionOutput::Attrs& attrs,
                            const NormalizedBBox<TDataType>& priorBboxes,
                            const CUDA::Array<TDataType, 4>& priorVariances,
                            const NormalizedBBox<TDataType>& bbox,
                            NormalizedBBox<TDataType>& decodeBbox) {
    TDataType priorXmin = priorBboxes.xmin;
    TDataType priorYmin = priorBboxes.ymin;
    TDataType priorXmax = priorBboxes.xmax;
    TDataType priorYmax = priorBboxes.ymax;

    if (!attrs.normalized) {
        priorXmin = priorXmin / TDataType{static_cast<float>(attrs.input_width)};
        priorYmin = priorYmin / TDataType{static_cast<float>(attrs.input_height)};
        priorXmax = priorXmax / TDataType{static_cast<float>(attrs.input_width)};
        priorYmax = priorYmax / TDataType{static_cast<float>(attrs.input_height)};
    }

    if (attrs.code_type == DetectionOutput::Attrs::CodeType::Caffe_PriorBoxParameter_CORNER) {
        decodeBbox.xmin = priorXmin + priorVariances[0] * bbox.xmin;
        decodeBbox.ymin = priorYmin + priorVariances[1] * bbox.ymin;
        decodeBbox.xmax = priorXmax + priorVariances[2] * bbox.xmax;
        decodeBbox.ymax = priorYmax + priorVariances[3] * bbox.ymax;
    } else if (attrs.code_type == DetectionOutput::Attrs::CodeType::Caffe_PriorBoxParameter_CENTER_SIZE) {
        const TDataType priorWidth = priorXmax - priorXmin;
        const TDataType priorHeight = priorYmax - priorYmin;
        const TDataType priorCenterX = (priorXmin + priorXmax) / TDataType{2.0};
        const TDataType priorCenterY = (priorYmin + priorYmax) / TDataType{2.0};
        const TDataType decodeBboxCenterX = priorVariances[0] * bbox.xmin * priorWidth + priorCenterX;
        const TDataType decodeBboxCenterY = priorVariances[1] * bbox.ymin * priorHeight + priorCenterY;
        const TDataType decodeBboxWidth = cumath::exp(priorVariances[2] * bbox.xmax) * priorWidth;
        const TDataType decodeBboxHeight = cumath::exp(priorVariances[3] * bbox.ymax) * priorHeight;
        decodeBbox.xmin = decodeBboxCenterX - decodeBboxWidth / TDataType{2.0};
        decodeBbox.ymin = decodeBboxCenterY - decodeBboxHeight / TDataType{2.0};
        decodeBbox.xmax = decodeBboxCenterX + decodeBboxWidth / TDataType{2.0};
        decodeBbox.ymax = decodeBboxCenterY + decodeBboxHeight / TDataType{2.0};
    }
}

template <typename TDataType>
__device__ void intersect_bbox(const NormalizedBBox<TDataType>& bbox1,
                               const NormalizedBBox<TDataType>& bbox2,
                               NormalizedBBox<TDataType>& intersectBbox) {
    if (bbox2.xmin > bbox1.xmax || bbox2.xmax < bbox1.xmin || bbox2.ymin > bbox1.ymax || bbox2.ymax < bbox1.ymin) {
        intersectBbox.xmin = 0.0;
        intersectBbox.ymin = 0.0;
        intersectBbox.xmax = 0.0;
        intersectBbox.ymax = 0.0;
    } else {
        intersectBbox.xmin = cumath::max<TDataType>(bbox1.xmin, bbox2.xmin);
        intersectBbox.ymin = cumath::max<TDataType>(bbox1.ymin, bbox2.ymin);
        intersectBbox.xmax = cumath::min<TDataType>(bbox1.xmax, bbox2.xmax);
        intersectBbox.ymax = cumath::min<TDataType>(bbox1.ymax, bbox2.ymax);
    }
}

template <typename TDataType>
__device__ TDataType bbox_size(const NormalizedBBox<TDataType>& bbox) {
    if (bbox.xmax < bbox.xmin || bbox.ymax < bbox.ymin) {
        return 0.0;
    } else {
        const TDataType width = bbox.xmax - bbox.xmin;
        const TDataType height = bbox.ymax - bbox.ymin;
        return width * height;
    }
}

template <typename TDataType>
__device__ TDataType jaccard_overlap(const NormalizedBBox<TDataType>& bbox1, const NormalizedBBox<TDataType>& bbox2) {
    NormalizedBBox<TDataType> intersectBbox;
    intersect_bbox(bbox1, bbox2, intersectBbox);

    const TDataType intersectWidth = intersectBbox.xmax - intersectBbox.xmin;
    const TDataType intersectHeight = intersectBbox.ymax - intersectBbox.ymin;
    if (intersectWidth > TDataType{0.0} && intersectHeight > TDataType{0.0}) {
        const TDataType intersect_size = intersectWidth * intersectHeight;
        const TDataType bbox1_size = bbox_size(bbox1);
        const TDataType bbox2_size = bbox_size(bbox2);
        return intersect_size / (bbox1_size + bbox2_size - intersect_size);
    } else {
        return 0.0f;
    }
}

template <typename TDataType>
__device__ void get_loc_predictions(const DetectionOutput::Attrs& attrs,
                                    CUDA::Span<const TDataType> locData,
                                    CUDA::MDSpan<NormalizedBBox<TDataType>, CUDA::DExtents<3>> locPreds) {
    const auto image_idx = get_image_idx();
    const auto class_idx = get_class_idx();
    const auto prio_idx = get_prio_idx(attrs);
    if (prio_idx == -1) {
        return;
    }

    const unsigned offsetIdx = image_idx * attrs.num_priors * attrs.num_loc_classes * 4;
    const unsigned startIdx = prio_idx * attrs.num_loc_classes * 4;
    const unsigned locIdx = offsetIdx + startIdx + class_idx * 4;
    const unsigned label = attrs.share_location ? 0 : class_idx;
    auto& nbox = locPreds(image_idx, label, prio_idx);
    nbox.xmin = locData[locIdx];
    nbox.ymin = locData[locIdx + 1];
    nbox.xmax = locData[locIdx + 2];
    nbox.ymax = locData[locIdx + 3];
}

template <typename TDataType>
__global__ void detection_output_initialization(CUDA::Span<CUDA::DeviceAtomic<unsigned>> numDets) {
    const unsigned i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < numDets.size()) {
        numDets[i] = 0;
    }
}

template <typename TDataType>
__global__ void detection_output_stage_0_confidence_scores(const DetectionOutput::Attrs& attrs,
                                                           CUDA::Span<const TDataType> confData,
                                                           CUDA::MDSpan<TDataType, CUDA::DExtents<3>> confPreds) {
    const auto image_idx = get_image_idx();
    const auto class_idx = get_class_idx();
    const auto prio_idx = get_prio_idx(attrs);
    if (prio_idx == -1) {
        return;
    }

    const unsigned offsetIdx = image_idx * attrs.num_priors * attrs.num_classes;
    const unsigned startIdx = prio_idx * attrs.num_classes;
    const unsigned confIdx = offsetIdx + startIdx + class_idx;
    if (confIdx < confData.size()) {
        confPreds(image_idx, class_idx, prio_idx) = confData[confIdx];
    } else {
        confPreds(image_idx, class_idx, prio_idx) = -1.0;
    }
}

template <typename TDataType>
__global__ void detection_output_stage_0_os_confidence_scores(const DetectionOutput::Attrs& attrs,
                                                              CUDA::Span<const TDataType> confData,
                                                              CUDA::Span<const TDataType> armConfData,
                                                              CUDA::MDSpan<TDataType, CUDA::DExtents<3>> confPreds) {
    const auto image_idx = get_image_idx();
    const auto class_idx = get_class_idx();
    const auto prio_idx = get_prio_idx(attrs);
    if (prio_idx == -1) {
        return;
    }

    const unsigned armOffsetIdx = image_idx * attrs.num_priors * 2;
    if (static_cast<float>(armConfData[armOffsetIdx + prio_idx * 2 + 1]) < attrs.objectness_score) {
        if (class_idx == attrs.background_label_id) {
            confPreds(image_idx, class_idx, prio_idx) = 1.0;
        } else {
            confPreds(image_idx, class_idx, prio_idx) = 0.0;
        }
    } else {
        const unsigned offsetIdx = image_idx * attrs.num_priors * attrs.num_classes;
        const unsigned startIdx = prio_idx * attrs.num_classes;
        const unsigned confIdx = offsetIdx + startIdx + class_idx;
        if (confIdx < confData.size()) {
            confPreds(image_idx, class_idx, prio_idx) = confData[confIdx];
        } else {
            confPreds(image_idx, class_idx, prio_idx) = -1.0;
        }
    }
}

template <typename TDataType>
__global__ void detection_output_stage_0_get_prior_bboxes(
    const DetectionOutput::Attrs& attrs,
    CUDA::Span<const TDataType> priorData,
    CUDA::MDSpan<NormalizedBBox<TDataType>, CUDA::DExtents<2>> priorBboxes,
    CUDA::MDSpan<CUDA::Array<TDataType, 4>, CUDA::DExtents<2>> priorVariances) {
    const auto image_idx = get_image_idx();
    const auto prio_idx = get_prio_idx(attrs);
    if (prio_idx == -1) {
        return;
    }

    const auto offsetImage = attrs.variance_encoded_in_target ? (attrs.num_priors * attrs.prior_size)
                                                              : (2 * attrs.num_priors * attrs.prior_size);
    const unsigned offsetIdx = image_idx * offsetImage;
    const unsigned startIdx = prio_idx * attrs.prior_size;
    const unsigned priorIdx = offsetIdx + startIdx + attrs.offset;
    auto& bbox = priorBboxes(image_idx, prio_idx);
    bbox.xmin = priorData[priorIdx + 0];
    bbox.ymin = priorData[priorIdx + 1];
    bbox.xmax = priorData[priorIdx + 2];
    bbox.ymax = priorData[priorIdx + 3];
    if (!attrs.variance_encoded_in_target) {
        const TDataType* priorVar = &priorData[offsetIdx + attrs.num_priors * attrs.prior_size];
        const unsigned idx = prio_idx * 4;
        auto& var = priorVariances(image_idx, prio_idx);
        for (int j = 0; j < 4; ++j) {
            var[j] = priorVar[idx + j];
        }
    }
}

template <typename TDataType>
__device__ void decode_bboxes(const DetectionOutput::Attrs& attrs,
                              const CUDA::Span<NormalizedBBox<TDataType>> priorBboxes,
                              const CUDA::Span<CUDA::Array<TDataType, 4>> priorVariances,
                              const CUDA::Span<NormalizedBBox<TDataType>> labelLocPreds,
                              CUDA::Span<NormalizedBBox<TDataType>> decodeBboxes) {
    const auto prio_idx = get_prio_idx(attrs);
    if (prio_idx == -1) {
        return;
    }

    NormalizedBBox<TDataType> decodeBbox;
    if (attrs.variance_encoded_in_target) {
        decode_bbox(attrs, priorBboxes[prio_idx], labelLocPreds[prio_idx], decodeBbox);
    } else {
        decode_bbox(attrs, priorBboxes[prio_idx], priorVariances[prio_idx], labelLocPreds[prio_idx], decodeBbox);
    }
    if (attrs.clip_before_nms) {
        decodeBbox.xmin = cumath::max<TDataType>(0.0, cumath::min<TDataType>(1.0, decodeBbox.xmin));
        decodeBbox.ymin = cumath::max<TDataType>(0.0, cumath::min<TDataType>(1.0, decodeBbox.ymin));
        decodeBbox.xmax = cumath::max<TDataType>(0.0, cumath::min<TDataType>(1.0, decodeBbox.xmax));
        decodeBbox.ymax = cumath::max<TDataType>(0.0, cumath::min<TDataType>(1.0, decodeBbox.ymax));
    }
    decodeBboxes[prio_idx] = decodeBbox;
}

template <typename TDataType>
__global__ void detection_output_stage_0_decode_bboxes_all(
    const DetectionOutput::Attrs& attrs,
    CUDA::MDSpan<NormalizedBBox<TDataType>, CUDA::DExtents<3>> locPreds,
    CUDA::MDSpan<NormalizedBBox<TDataType>, CUDA::DExtents<2>> priorBboxes,
    CUDA::MDSpan<CUDA::Array<TDataType, 4>, CUDA::DExtents<2>> priorVariances,
    CUDA::MDSpan<NormalizedBBox<TDataType>, CUDA::DExtents<3>> decodeBboxes) {
    const auto image_idx = get_image_idx();
    const auto class_idx = get_class_idx();
    const auto prio_idx = get_prio_idx(attrs);
    if (prio_idx == -1) {
        return;
    }

    decodeBboxes(image_idx, class_idx, prio_idx) = InvalidNormalizedBBox<TDataType>{};

    unsigned pboxIdx = image_idx;
    if (priorBboxes.extent(0) == 1) {
        pboxIdx = 0;
    }
    int label = attrs.share_location ? -1 : class_idx;
    if (attrs.background_label_id > -1 && label == attrs.background_label_id) {
        return;
    }
    if (label == -1) {
        label = 0;
    }
    CUDA::Span<NormalizedBBox<TDataType>> currPriorBboxes{&priorBboxes(pboxIdx, 0), priorBboxes.extent(1)};
    CUDA::Span<CUDA::Array<TDataType, 4>> currPriorVariances{&priorVariances(pboxIdx, 0), priorVariances.extent(1)};
    CUDA::Span<NormalizedBBox<TDataType>> labelLocPreds{&locPreds(image_idx, label, 0), locPreds.extent(2)};
    CUDA::Span<NormalizedBBox<TDataType>> labelDecodePriorBboxes{&decodeBboxes(image_idx, label, 0),
                                                                 decodeBboxes.extent(2)};
    decode_bboxes(attrs, currPriorBboxes, currPriorVariances, labelLocPreds, labelDecodePriorBboxes);
}

template <typename TDataType>
__global__ void detection_output_stage_0_cas_decode_bboxes_all(
    const DetectionOutput::Attrs& attrs,
    CUDA::MDSpan<NormalizedBBox<TDataType>, CUDA::DExtents<3>> locPreds,
    CUDA::MDSpan<NormalizedBBox<TDataType>, CUDA::DExtents<2>> priorBboxes,
    CUDA::MDSpan<CUDA::Array<TDataType, 4>, CUDA::DExtents<2>> priorVariances,
    CUDA::MDSpan<NormalizedBBox<TDataType>, CUDA::DExtents<3>> armLocPreds,
    CUDA::MDSpan<NormalizedBBox<TDataType>, CUDA::DExtents<3>> decodePriorBboxes,
    CUDA::MDSpan<NormalizedBBox<TDataType>, CUDA::DExtents<3>> decodeBboxes) {
    const auto image_idx = get_image_idx();
    const auto class_idx = get_class_idx();
    const auto prio_idx = get_prio_idx(attrs);
    if (prio_idx == -1) {
        return;
    }

    decodeBboxes(image_idx, class_idx, prio_idx) = InvalidNormalizedBBox<TDataType>{};

    unsigned pboxIdx = image_idx;
    if (priorBboxes.extent(0) == 1) {
        pboxIdx = 0;
    }

    int label = attrs.share_location ? -1 : class_idx;
    if (attrs.background_label_id > -1 && label == attrs.background_label_id) {
        return;
    }
    if (label == -1) {
        label = 0;
    }
    CUDA::Span<NormalizedBBox<TDataType>> currPrBbox{&priorBboxes(pboxIdx, 0), priorBboxes.extent(1)};
    CUDA::Span<CUDA::Array<TDataType, 4>> currPrVar{&priorVariances(pboxIdx, 0), priorVariances.extent(1)};
    CUDA::Span<NormalizedBBox<TDataType>> labelLocPreds{&locPreds(image_idx, label, 0), locPreds.extent(2)};
    CUDA::Span<NormalizedBBox<TDataType>> labelArmLocPreds{&armLocPreds(image_idx, label, 0), armLocPreds.extent(2)};
    CUDA::Span<NormalizedBBox<TDataType>> labelDecodeBboxes{&decodeBboxes(image_idx, label, 0), decodeBboxes.extent(2)};
    CUDA::Span<NormalizedBBox<TDataType>> labelDecodePriorBboxes{&decodePriorBboxes(image_idx, label, 0),
                                                                 decodePriorBboxes.extent(2)};
    decode_bboxes(attrs, currPrBbox, currPrVar, labelArmLocPreds, labelDecodePriorBboxes);
    decode_bboxes(attrs, labelDecodePriorBboxes, currPrVar, labelLocPreds, labelDecodeBboxes);
}

template <typename TDataType>
__global__ void detection_output_stage_0_get_loc_predictions(
    const DetectionOutput::Attrs& attrs,
    CUDA::Span<const TDataType> locData,
    CUDA::MDSpan<NormalizedBBox<TDataType>, CUDA::DExtents<3>> locPreds) {
    get_loc_predictions(attrs, locData, locPreds);
}

template <typename TDataType>
__global__ void detection_output_stage_0_get_loc_predictions_with_arm(
    const DetectionOutput::Attrs& attrs,
    CUDA::Span<const TDataType> locData,
    CUDA::Span<const TDataType> armLocData,
    CUDA::MDSpan<NormalizedBBox<TDataType>, CUDA::DExtents<3>> locPreds,
    CUDA::MDSpan<NormalizedBBox<TDataType>, CUDA::DExtents<3>> armLocPreds) {
    get_loc_predictions(attrs, locData, locPreds);
    get_loc_predictions(attrs, armLocData, armLocPreds);
}

template <typename TDataType>
__global__ void detection_output_stage_1_caffe_nms(
    const DetectionOutput::Attrs& attrs,
    const CUDA::MDSpan<NormalizedBBox<TDataType>, CUDA::DExtents<3>> decodeBboxes,
    const CUDA::MDSpan<TDataType, CUDA::DExtents<3>> confPreds,
    CUDA::MDVector<CUDA::Pair<TDataType, CUDA::Pair<int, int>>, 2> scorePerClassPrioIdxs,
    CUDA::MDVector<int, 2> prioBoxIdxsByClass,
    CUDA::Span<CUDA::DeviceAtomic<unsigned>> numDets) {
    const auto image_idx = get_image_idx();
    const auto class_idx = get_class_idx();
    // TODO: Check if performance with changes after moving class_idx in warp dimensions
    //    const auto class_idx = get_class_idx(attrs);
    //    if (class_idx == -1) { return; }

    if (class_idx == attrs.background_label_id) {
        return;
    }

    int label = attrs.share_location ? 0 : class_idx;
    auto& numDet = numDets[image_idx];
    CUDA::Span<const NormalizedBBox<TDataType>> bboxes{&decodeBboxes(image_idx, label, 0), decodeBboxes.extent(2)};
    CUDA::Span<const TDataType> scores{&confPreds(image_idx, class_idx, 0), confPreds.extent(2)};
    auto localScorePerClassPrioIdxs = scorePerClassPrioIdxs(image_idx, class_idx);
    for (int priorIdx = 0; priorIdx < scores.size(); ++priorIdx) {
        TDataType conf = scores[priorIdx];
        if (conf > TDataType{attrs.confidence_threshold}) {
            localScorePerClassPrioIdxs.push_back(CUDA::make_pair(conf, CUDA::make_pair(class_idx, priorIdx)));
        }
    }
    CUDA::algorithms::partial_quick_sort_iterative(localScorePerClassPrioIdxs.begin(),
                                                   localScorePerClassPrioIdxs.end(),
                                                   attrs.top_k,
                                                   SortScorePairDescend<TDataType, CUDA::Pair<int, int>>{});

    if (-1 != attrs.top_k && localScorePerClassPrioIdxs.size() > static_cast<size_t>(attrs.top_k)) {
        localScorePerClassPrioIdxs.resize(attrs.top_k);
    }

    auto prioBoxIdxs = prioBoxIdxsByClass(image_idx, class_idx);
    for (const auto& scorePerPrioIdx : localScorePerClassPrioIdxs) {
        const int priorIdx = scorePerPrioIdx.second.second;
        bool keep = true;
        for (const auto& keptPrioIdx : prioBoxIdxs) {
            const TDataType overlap = jaccard_overlap(bboxes[priorIdx], bboxes[keptPrioIdx]);
            if (overlap > TDataType{attrs.nms_threshold}) {
                keep = false;
                break;
            }
        }
        if (keep) {
            prioBoxIdxs.push_back(priorIdx);
            numDet += 1;
        }
    }
}

template <typename TDataType>
__global__ void detection_output_stage_1_mxnet_nms(
    const DetectionOutput::Attrs& attrs,
    const CUDA::MDSpan<NormalizedBBox<TDataType>, CUDA::DExtents<3>> decodeBboxes,
    const CUDA::MDSpan<TDataType, CUDA::DExtents<3>> confPreds,
    CUDA::MDVector<CUDA::Pair<TDataType, CUDA::Pair<int, int>>, 1> tempScorePerClassPrioIdxs,
    CUDA::MDVector<int, 2> prioBoxIdxsByClass,
    CUDA::Span<CUDA::DeviceAtomic<unsigned>> numDets) {
    const auto image_idx = get_image_idx();

    auto& numDet = numDets[image_idx];
    CUDA::MDSpan<const TDataType, CUDA::DExtents<2>> imageConfPreds{
        &confPreds(image_idx, 0, 0), confPreds.extent(1), confPreds.extent(2)};

    for (int prio_idx = 0; prio_idx < attrs.num_priors; ++prio_idx) {
        TDataType conf = -1.0;
        int bestClsIdx = 0;
        for (int clsIdx = 1; clsIdx < attrs.num_classes; ++clsIdx) {
            if (attrs.background_label_id > -1 && clsIdx == attrs.background_label_id) {
                continue;
            }
            TDataType temp = imageConfPreds(clsIdx, prio_idx);
            if (temp > conf) {
                conf = temp;
                bestClsIdx = clsIdx;
            }
        }
        if (bestClsIdx > 0 && conf >= TDataType{attrs.confidence_threshold}) {
            tempScorePerClassPrioIdxs(image_idx).push_back(
                CUDA::make_pair(conf, CUDA::make_pair(bestClsIdx, prio_idx)));
        }
    }

    auto localScorePerClassPrioIdxs = tempScorePerClassPrioIdxs(image_idx);
    CUDA::algorithms::partial_quick_sort_iterative(localScorePerClassPrioIdxs.begin(),
                                                   localScorePerClassPrioIdxs.end(),
                                                   attrs.top_k,
                                                   SortScorePairDescend<TDataType, CUDA::Pair<int, int>>{});
    if (attrs.top_k != -1) {
        if (localScorePerClassPrioIdxs.size() > static_cast<size_t>(attrs.top_k)) {
            localScorePerClassPrioIdxs.resize(attrs.top_k);
        }
    }

    for (const auto& scorePerPrioIdx : localScorePerClassPrioIdxs) {
        const int clsIdx = scorePerPrioIdx.second.first;
        const int priorIdx = scorePerPrioIdx.second.second;
        auto prioBoxIdxs = prioBoxIdxsByClass(image_idx, clsIdx);
        const unsigned label = attrs.share_location ? 0 : clsIdx;
        CUDA::Span<const NormalizedBBox<TDataType>> bboxes{&decodeBboxes(image_idx, label, 0), decodeBboxes.extent(2)};
        bool keep = true;
        for (const auto& keptPrioIdx : prioBoxIdxs) {
            const TDataType overlap = jaccard_overlap(bboxes[priorIdx], bboxes[keptPrioIdx]);
            if (overlap > TDataType{attrs.nms_threshold}) {
                keep = false;
                break;
            }
        }
        if (keep) {
            prioBoxIdxs.push_back(priorIdx);
            numDet += 1;
        }
    }
}

template <typename TDataType>
__global__ void detection_output_stage_1_keep_top_scores(
    const DetectionOutput::Attrs& attrs,
    CUDA::MDSpan<TDataType, CUDA::DExtents<3>> confPreds,
    CUDA::MDVector<CUDA::Pair<TDataType, CUDA::Pair<int, int>>, 1> tempScorePerClassPrioIdxs,
    CUDA::MDVector<int, 2> prioBoxIdxsByClass,
    CUDA::Span<CUDA::DeviceAtomic<unsigned>> numDets) {
    const auto image_idx = get_image_idx();

    auto& numDet = numDets[image_idx];

    auto localScorePerClassPrioIdxs = tempScorePerClassPrioIdxs(image_idx);
    localScorePerClassPrioIdxs.clear();

    if (attrs.keep_top_k > -1 && numDet.load() > attrs.keep_top_k) {
        for (int clsIdx = 0; clsIdx < prioBoxIdxsByClass.extent(1); ++clsIdx) {
            CUDA::Span<const TDataType> localConfPreds{&confPreds(image_idx, clsIdx, 0), confPreds.extent(2)};

            if (localConfPreds[0] == TDataType{-1.0}) {
                continue;
            }
            const auto prioBoxIdxs = prioBoxIdxsByClass(image_idx, clsIdx);
            for (size_t j = 0; j < prioBoxIdxs.size(); ++j) {
                int prioIdx = prioBoxIdxs[j];
                localScorePerClassPrioIdxs.push_back(
                    CUDA::make_pair(localConfPreds[prioIdx], CUDA::make_pair(clsIdx, prioIdx)));
            }
        }
        CUDA::algorithms::partial_quick_sort_iterative(localScorePerClassPrioIdxs.begin(),
                                                       localScorePerClassPrioIdxs.end(),
                                                       attrs.keep_top_k,
                                                       SortScorePairDescend<TDataType, CUDA::Pair<int, int>>{});
        localScorePerClassPrioIdxs.resize(attrs.keep_top_k);

        for (int clsIdx = 0; clsIdx < prioBoxIdxsByClass.extent(1); ++clsIdx) {
            prioBoxIdxsByClass(image_idx, clsIdx).clear();
        }
        for (size_t j = 0; j < localScorePerClassPrioIdxs.size(); ++j) {
            int clsIdx = localScorePerClassPrioIdxs[j].second.first;
            int prioIdx = localScorePerClassPrioIdxs[j].second.second;
            prioBoxIdxsByClass(image_idx, clsIdx).push_back(prioIdx);
        }
        if (localScorePerClassPrioIdxs.size() < attrs.keep_top_k) {
            numDet = localScorePerClassPrioIdxs.size();
        } else {
            numDet = attrs.keep_top_k;
        }
    }
}

template <typename TDataType>
__global__ void detection_output_stage_2_results(
    const DetectionOutput::Attrs& attrs,
    CUDA::MDSpan<TDataType, CUDA::DExtents<3>> confPreds,
    CUDA::MDSpan<NormalizedBBox<TDataType>, CUDA::DExtents<3>> decodeBboxes,
    CUDA::MDVector<int, 2> prioBoxIdxsByClass,
    CUDA::Span<CUDA::DeviceAtomic<unsigned>> numDets,
    CUDA::Span<DetectionOutputResult<TDataType>> results) {
    const auto image_idx = get_image_idx();

    size_t count = 0;
    for (size_t i = 0; i < image_idx; ++i) {
        count += numDets[i].load();
    }

    if (count >= (attrs.num_results - 1)) {
        return;
    }

    CUDA::MDSpan<const TDataType, CUDA::DExtents<2>> imageConfPreds{
        &confPreds(image_idx, 0, 0), confPreds.extent(1), confPreds.extent(2)};
    CUDA::MDSpan<const NormalizedBBox<TDataType>, CUDA::DExtents<2>> imageDecodeBboxes{
        &decodeBboxes(image_idx, 0, 0), decodeBboxes.extent(1), decodeBboxes.extent(2)};
    for (int clsIdx = 0; clsIdx < prioBoxIdxsByClass.extent(1); ++clsIdx) {
        if (imageConfPreds(clsIdx, 0) == TDataType{-1.0}) {
            continue;
        }
        int label = attrs.share_location ? 0 : clsIdx;
        CUDA::Span<const NormalizedBBox<TDataType>> bboxes{&imageDecodeBboxes(label, 0), imageDecodeBboxes.extent(1)};
        const auto indices = prioBoxIdxsByClass(image_idx, clsIdx);
        for (size_t j = 0; j < indices.size(); ++j) {
            const int prioIdx = indices[j];
            if (bboxes[prioIdx] == InvalidNormalizedBBox<TDataType>{}) {
                continue;
            }
            results[count].img = static_cast<double>(image_idx);
            results[count].cls = static_cast<double>(attrs.decrease_label_id ? (clsIdx - 1) : clsIdx);
            results[count].conf = static_cast<double>(imageConfPreds(clsIdx, prioIdx));

            const auto& bbox = bboxes[prioIdx];
            TDataType xmin = bbox.xmin;
            TDataType ymin = bbox.ymin;
            TDataType xmax = bbox.xmax;
            TDataType ymax = bbox.ymax;

            if (attrs.clip_after_nms) {
                xmin = cumath::max<TDataType>(0.0, cumath::min<TDataType>(1.0, xmin));
                ymin = cumath::max<TDataType>(0.0, cumath::min<TDataType>(1.0, ymin));
                xmax = cumath::max<TDataType>(0.0, cumath::min<TDataType>(1.0, xmax));
                ymax = cumath::max<TDataType>(0.0, cumath::min<TDataType>(1.0, ymax));
            }

            results[count].xmin = xmin;
            results[count].ymin = ymin;
            results[count].xmax = xmax;
            results[count].ymax = ymax;
            ++count;
        }
    }
    if (0 == image_idx) {
        size_t max_count = 0;
        for (size_t i = 0; i < numDets.size(); ++i) {
            max_count += numDets[i].load();
        }
        if (max_count < attrs.num_results) {
            results[max_count].img = -1.0;
        }
    }
}

namespace experimental {

template <size_t NumPartitions, typename TDataType>
__global__ void detection_output_stage_1_caffe_nms(
    const DetectionOutput::Attrs& attrs,
    const CUDA::MDSpan<NormalizedBBox<TDataType>, CUDA::DExtents<3>> decodeBboxes,
    const CUDA::MDSpan<TDataType, CUDA::DExtents<3>> confPreds,
    CUDA::MDVector<CUDA::Pair<TDataType, CUDA::Pair<int, int>>, 2> scorePerClassPrioIdxs,
    CUDA::MDVector<int, 2> prioBoxIdxsByClass,
    CUDA::Span<CUDA::DeviceAtomic<unsigned>> numDets) {
    const auto image_idx = get_image_idx();
    const auto class_idx = get_class_idx();
    const auto globalWorkItemId = get_global_work_item_id();
    const auto globalNumWorkItems = get_global_work_group_size();
    const auto workItemId = get_work_item_id();
    const auto numWorkItems = get_work_group_size();
    if (workItemId >= NumPartitions) {
        __trap();
    }

    if (class_idx == attrs.background_label_id) {
        return;
    }

    int label = attrs.share_location ? 0 : class_idx;
    auto& numDet = numDets[image_idx];
    CUDA::Span<const NormalizedBBox<TDataType>> bboxes{&decodeBboxes(image_idx, label, 0), decodeBboxes.extent(2)};
    CUDA::Span<const TDataType> scores{&confPreds(image_idx, class_idx, 0), confPreds.extent(2)};
    auto localScorePerClassPrioIdxs = scorePerClassPrioIdxs(image_idx, class_idx);
    if (globalWorkItemId == 0 && workItemId == 0) {
        for (int priorIdx = 0; priorIdx < scores.size(); ++priorIdx) {
            TDataType conf = scores[priorIdx];
            if (conf > TDataType{attrs.confidence_threshold}) {
                localScorePerClassPrioIdxs.push_back(CUDA::make_pair(conf, CUDA::make_pair(class_idx, priorIdx)));
            }
        }
    }
    __syncthreads();
    SortScorePairDescend<TDataType, CUDA::Pair<int, int>> comparer{};
    CUDA::algorithms::parallel::quick_sort_iterative<NumPartitions>(
        localScorePerClassPrioIdxs.begin(), localScorePerClassPrioIdxs.end(), comparer, workItemId, numWorkItems);
    __syncthreads();

    if (globalWorkItemId == 0 && workItemId == 0) {
        if (-1 != attrs.top_k && localScorePerClassPrioIdxs.size() > static_cast<size_t>(attrs.top_k)) {
            localScorePerClassPrioIdxs.resize(attrs.top_k);
        }

        auto prioBoxIdxs = prioBoxIdxsByClass(image_idx, class_idx);
        for (const auto& scorePerPrioIdx : localScorePerClassPrioIdxs) {
            const int priorIdx = scorePerPrioIdx.second.second;
            bool keep = true;
            for (const auto& keptPrioIdx : prioBoxIdxs) {
                const TDataType overlap = jaccard_overlap(bboxes[priorIdx], bboxes[keptPrioIdx]);
                if (overlap > TDataType{attrs.nms_threshold}) {
                    keep = false;
                    break;
                }
            }
            if (keep) {
                prioBoxIdxs.push_back(priorIdx);
                numDet += 1;
            }
        }
    }
}

template <size_t NumPartitions, typename TDataType>
__global__ void detection_output_stage_1_keep_top_scores(
    const DetectionOutput::Attrs& attrs,
    CUDA::MDSpan<TDataType, CUDA::DExtents<3>> confPreds,
    CUDA::MDVector<CUDA::Pair<TDataType, CUDA::Pair<int, int>>, 1> tempScorePerClassPrioIdxs,
    CUDA::MDVector<int, 2> prioBoxIdxsByClass,
    CUDA::Span<CUDA::DeviceAtomic<unsigned>> numDets) {
    const auto image_idx = get_image_idx();
    const auto workItemId = get_work_item_id();
    const auto numWorkItems = get_work_group_size();
    if (workItemId >= NumPartitions) {
        __trap();
    }

    auto& numDet = numDets[image_idx];

    auto localScorePerClassPrioIdxs = tempScorePerClassPrioIdxs(image_idx);
    localScorePerClassPrioIdxs.clear();
    __syncthreads();

    if (attrs.keep_top_k > -1 && numDet.load() > attrs.keep_top_k) {
        if (workItemId == 0) {
            for (int clsIdx = 0; clsIdx < prioBoxIdxsByClass.extent(1); ++clsIdx) {
                if (confPreds(image_idx, clsIdx, 0) == TDataType{-1.0}) {
                    continue;
                }
                const auto prioBoxIdxs = prioBoxIdxsByClass(image_idx, clsIdx);
                for (size_t j = 0; j < prioBoxIdxs.size(); ++j) {
                    int prioIdx = prioBoxIdxs[j];
                    localScorePerClassPrioIdxs.push_back(
                        CUDA::make_pair(confPreds(image_idx, clsIdx, prioIdx), CUDA::make_pair(clsIdx, prioIdx)));
                }
            }
        }
        __syncthreads();
        SortScorePairDescend<TDataType, CUDA::Pair<int, int>> comparer{};
        CUDA::algorithms::parallel::quick_sort_iterative<NumPartitions>(
            localScorePerClassPrioIdxs.begin(), localScorePerClassPrioIdxs.end(), comparer, workItemId, numWorkItems);
        __syncthreads();

        if (workItemId == 0) {
            localScorePerClassPrioIdxs.resize(attrs.keep_top_k);

            for (int clsIdx = 0; clsIdx < prioBoxIdxsByClass.extent(1); ++clsIdx) {
                prioBoxIdxsByClass(image_idx, clsIdx).clear();
            }
            for (size_t j = 0; j < localScorePerClassPrioIdxs.size(); ++j) {
                int clsIdx = localScorePerClassPrioIdxs[j].second.first;
                int prioIdx = localScorePerClassPrioIdxs[j].second.second;
                prioBoxIdxsByClass(image_idx, clsIdx).push_back(prioIdx);
            }
            if (localScorePerClassPrioIdxs.size() < attrs.keep_top_k) {
                numDet = localScorePerClassPrioIdxs.size();
            } else {
                numDet = attrs.keep_top_k;
            }
        }
    }
}

}  // namespace experimental

#ifdef CUDA_KERNEL_PRINT_LOG
namespace debug {

template <typename TDataType>
__global__ void detection_output_stage_0_debug(
    const DetectionOutput::Attrs& attrs,
    CUDA::MDSpan<TDataType, CUDA::DExtents<3>> confPreds,
    CUDA::MDSpan<NormalizedBBox<TDataType>, CUDA::DExtents<3>> locPreds,
    CUDA::MDSpan<NormalizedBBox<TDataType>, CUDA::DExtents<2>> priorBboxes,
    CUDA::MDSpan<CUDA::Array<TDataType, 4>, CUDA::DExtents<2>> priorVariances,
    CUDA::MDSpan<NormalizedBBox<TDataType>, CUDA::DExtents<3>> decodeBboxes) {
    printf(">>>>>>>>>>>>>>>>>> detection_output_stage_0_debug >>>>>>>>>>>>>>>>>>\n");
    for (size_t image_idx = 0; image_idx < confPreds.extent(0); ++image_idx) {
        for (int class_idx = 0; class_idx < confPreds.extent(1); ++class_idx) {
            for (size_t prio_idx = 0; prio_idx < confPreds.extent(2); ++prio_idx) {
                if ((class_idx >= 25 && class_idx < 70) && (prio_idx >= 1000 && prio_idx < 1200)) {
                    printf("confPreds(%lu, %d, %lu) = %f\n",
                           image_idx,
                           class_idx,
                           prio_idx,
                           confPreds.at(image_idx, class_idx, prio_idx));
                }
            }
        }
        for (size_t prio_idx = 0; prio_idx < confPreds.extent(2); ++prio_idx) {
            const auto& bbox = priorBboxes(image_idx, prio_idx);
            printf("priorBboxes(%lu, %lu):\n", image_idx, prio_idx);
            printf("xmin = %f\n", bbox.xmin);
            printf("ymin = %f\n", bbox.ymin);
            printf("xmax = %f\n", bbox.xmax);
            printf("ymax = %f\n", bbox.ymax);
        }
        for (size_t prio_idx = 0; prio_idx < confPreds.extent(2); ++prio_idx) {
            const auto& bbox = priorVariances(image_idx, prio_idx);
            printf("priorVariances(%lu, %lu):\n", image_idx, prio_idx);
            printf("bbox[0] = %f\n", bbox[0]);
            printf("bbox[1] = %f\n", bbox[1]);
            printf("bbox[2] = %f\n", bbox[2]);
            printf("bbox[3] = %f\n", bbox[3]);
        }
        if (attrs.share_location) {
            for (size_t prio_idx = 0; prio_idx < confPreds.extent(2); ++prio_idx) {
                const auto& bbox = locPreds(image_idx, 0, prio_idx);
                printf("share locPreds(%lu, %d, %lu):\n", image_idx, -1, prio_idx);
                printf("xmin = %f\n", bbox.xmin);
                printf("ymin = %f\n", bbox.ymin);
                printf("xmax = %f\n", bbox.xmax);
                printf("ymax = %f\n", bbox.ymax);
            }
        } else {
            for (int class_idx = 0; class_idx < confPreds.extent(1); ++class_idx) {
                for (size_t prio_idx = 0; prio_idx < confPreds.extent(2); ++prio_idx) {
                    const auto& bbox = locPreds(image_idx, class_idx, prio_idx);
                    printf("locPreds(%lu, %d, %lu):\n", image_idx, class_idx, prio_idx);
                    printf("xmin = %f\n", bbox.xmin);
                    printf("ymin = %f\n", bbox.ymin);
                    printf("xmax = %f\n", bbox.xmax);
                    printf("ymax = %f\n", bbox.ymax);
                }
            }
        }
        if (attrs.share_location) {
            for (size_t prio_idx = 0; prio_idx < confPreds.extent(2); ++prio_idx) {
                if (prio_idx >= 0 && prio_idx < 100) {
                    const auto& bbox = decodeBboxes(image_idx, 0, prio_idx);
                    printf("share decodeBboxes(%lu, %d, %lu):\n", image_idx, -1, prio_idx);
                    printf("xmin = %f\n", bbox.xmin);
                    printf("ymin = %f\n", bbox.ymin);
                    printf("xmax = %f\n", bbox.xmax);
                    printf("ymax = %f\n", bbox.ymax);
                }
            }
        } else {
            for (int class_idx = 0; class_idx < confPreds.extent(1); ++class_idx) {
                for (size_t prio_idx = 0; prio_idx < confPreds.extent(2); ++prio_idx) {
                    if ((class_idx >= 25 && class_idx < 70) && (prio_idx >= 0 && prio_idx < 100)) {
                        const auto& bbox = decodeBboxes(image_idx, class_idx, prio_idx);
                        printf("decodeBboxes(%lu, %d, %lu):\n", image_idx, class_idx, prio_idx);
                        printf("xmin = %f\n", bbox.xmin);
                        printf("ymin = %f\n", bbox.ymin);
                        printf("xmax = %f\n", bbox.xmax);
                        printf("ymax = %f\n", bbox.ymax);
                    }
                }
            }
        }
    }
    printf("\n");
    printf("<<<<<<<<<<<<<<<<<< detection_output_stage_0_debug <<<<<<<<<<<<<<<<<<\n");
}

template <typename TDataType>
__global__ void detection_output_stage_1_debug(const DetectionOutput::Attrs& attrs,
                                               CUDA::MDVector<int, 2> prioBoxIdxsByClass,
                                               CUDA::Span<CUDA::DeviceAtomic<unsigned>> numDets) {
    printf(">>>>>>>>>>>>>>>>>> detection_output_stage_1_debug >>>>>>>>>>>>>>>>>>\n");
    for (size_t image_idx = 0; image_idx < prioBoxIdxsByClass.extent(0); ++image_idx) {
        printf("prioBoxIds(image_idx = %lu): numDet = %u\n", image_idx, numDets[image_idx].load());
        for (int label = 0; label < prioBoxIdxsByClass.extent(1); ++label) {
            if (prioBoxIdxsByClass(image_idx, label).size() > 0) {
                printf("prioBoxIds(image_idx = %lu, label = %d):\n", image_idx, label);
                for (const auto& prioBoxId : prioBoxIdxsByClass(image_idx, label)) {
                    printf("prioBoxId = %d\n", prioBoxId);
                }
            }
        }
    }
    printf("\n");
    printf("<<<<<<<<<<<<<<<<<< detection_output_stage_1_debug <<<<<<<<<<<<<<<<<<\n");
}

template <typename TDataType>
__global__ void detection_output_stage_2_debug(const DetectionOutput::Attrs& attrs,
                                               CUDA::Span<DetectionOutputResult<TDataType>> results,
                                               CUDA::Span<CUDA::DeviceAtomic<unsigned>> numDets) {
    printf(">>>>>>>>>>>>>>>>>> detection_output_stage_2_debug >>>>>>>>>>>>>>>>>>\n");
    size_t count = 0;
    for (int i = 0; i < numDets.size(); ++i) {
        const auto& numDet = numDets[i];

        for (int o = 0; o < numDet.load(); ++o) {
            const int index_offset = count + o;
            printf("result(image_idx = %d, idx = %d):\n", i, o);
            printf("  class_idx = %f\n", results[index_offset].cls);
            printf("  score = %f\n", results[index_offset].conf);
            printf("  xmin = %f\n", results[index_offset].xmin);
            printf("  ymin = %f\n", results[index_offset].ymin);
            printf("  xmax = %f\n", results[index_offset].xmax);
            printf("  ymax = %f\n", results[index_offset].ymax);
        }

        count += numDet.load();
    }

    printf("\n");
    printf("<<<<<<<<<<<<<<<<<< detection_output_stage_2_debug <<<<<<<<<<<<<<<<<<\n");
}

}  // namespace debug
#endif

DetectionOutput::DetectionOutput(const Type_t element_type,
                                 const size_t max_threads_per_block,
                                 const size_t location_size,
                                 const size_t confidence_size,
                                 const size_t priors_size,
                                 const size_t arm_confidence_size,
                                 const size_t arm_location_size,
                                 const size_t result_size,
                                 const Attrs attrs)
    : element_type_{element_type},
      attrs_{attrs},
      max_threads_per_block_{max_threads_per_block},
      location_size_{location_size},
      confidence_size_{confidence_size},
      priors_size_{priors_size},
      arm_confidence_size_{arm_confidence_size},
      arm_location_size_{arm_location_size},
      result_size_{result_size} {
    TypeValidator<FloatElementTypesSwitch>::check(element_type_);
}

template <typename TDataType>
std::vector<size_t> DetectionOutput::getMutableWorkbufferSizes() const {
    const auto locationsSize = CUDA::MDSpan<NormalizedBBox<TDataType>, CUDA::DExtents<3>>::size_of(
        attrs_.num_images, attrs_.num_loc_classes, attrs_.num_priors);
    const auto confPredsSize =
        CUDA::MDSpan<TDataType, CUDA::DExtents<3>>::size_of(attrs_.num_images, attrs_.num_classes, attrs_.num_priors);
    const auto priorBboxesSize =
        CUDA::MDSpan<NormalizedBBox<TDataType>, CUDA::DExtents<2>>::size_of(attrs_.num_images, attrs_.num_priors);
    const auto priorVariancesSize =
        CUDA::MDSpan<CUDA::Array<TDataType, 4>, CUDA::DExtents<2>>::size_of(attrs_.num_images, attrs_.num_priors);
    const auto decodeBboxesSize = CUDA::MDSpan<NormalizedBBox<TDataType>, CUDA::DExtents<3>>::size_of(
        attrs_.num_images, attrs_.num_loc_classes, attrs_.num_priors);
    const auto tempDecodeBboxesSize = CUDA::MDSpan<NormalizedBBox<TDataType>, CUDA::DExtents<3>>::size_of(
        attrs_.num_images, attrs_.num_loc_classes, attrs_.num_priors);
    const auto numDetsSize = CUDA::Span<CUDA::DeviceAtomic<unsigned>>::size_of(attrs_.num_images);
    const auto tempScorePerClassPrioIdxs0Size = CUDA::MDVector<CUDA::Pair<TDataType, CUDA::Pair<int, int>>, 2>::size_of(
        attrs_.num_priors, attrs_.num_images, attrs_.num_classes);
    const auto tempScorePerClassPrioIdxs1Size = CUDA::MDVector<CUDA::Pair<TDataType, CUDA::Pair<int, int>>, 1>::size_of(
        attrs_.num_classes * attrs_.num_priors, attrs_.num_images);
    const auto prioBoxIdxsByClassSize =
        CUDA::MDVector<int, 2>::size_of(attrs_.num_priors, attrs_.num_images, attrs_.num_classes);
    const auto armLocationsSize = CUDA::MDSpan<NormalizedBBox<TDataType>, CUDA::DExtents<3>>::size_of(
        attrs_.num_images, attrs_.num_loc_classes, attrs_.num_priors);

    std::vector<size_t> workbuffers(arm_location_size_ > 0 ? kNumOptionalWB : kNumRequiredWB);
    workbuffers.at(kLocationsWBIdx) = locationsSize;
    workbuffers.at(kConfPredsWBIdx) = confPredsSize;
    workbuffers.at(kPriorBboxesWBIdx) = priorBboxesSize;
    workbuffers.at(kPriorVariancesWBIdx) = priorVariancesSize;
    workbuffers.at(kDecodeBboxesWBIdx) = decodeBboxesSize;
    workbuffers.at(kTempDecodeBboxesWBIdx) = tempDecodeBboxesSize;
    workbuffers.at(kNumDetectionsWBIdx) = numDetsSize;
    workbuffers.at(kTempScorePerClassPrioIdxs0WBIdx) = tempScorePerClassPrioIdxs0Size;
    workbuffers.at(kTempScorePerClassPrioIdxs1WBIdx) = tempScorePerClassPrioIdxs1Size;
    workbuffers.at(kPrioBoxIdxsByClassWBIdx) = prioBoxIdxsByClassSize;
    if (arm_location_size_ > 0) {
        workbuffers.at(kArmLocationsWBIdx) = armLocationsSize;
    }
    return std::move(workbuffers);
}

std::vector<size_t> DetectionOutput::getMutableWorkbufferSizes() const {
    switch (element_type_) {
#ifdef CUDA_HAS_BF16_TYPE
        case Type_t::bf16:
            return getMutableWorkbufferSizes<__nv_bfloat16>();
#endif
        case Type_t::f16:
            return getMutableWorkbufferSizes<__half>();
        case Type_t::f32:
            return getMutableWorkbufferSizes<float>();
        case Type_t::f64:
            return getMutableWorkbufferSizes<double>();
        default:
            throwIEException(
                fmt::format("Input element type = {} is not supported by Split operation "
                            "!!",
                            static_cast<Type_t>(element_type_)));
    }
}

std::vector<size_t> DetectionOutput::getImmutableWorkbufferSizes() const { return {sizeof(attrs_)}; }

void DetectionOutput::initSharedImmutableWorkbuffers(const Buffers& buffers) {
    CUDA::DefaultStream::stream().upload(buffers.at(0), &attrs_, sizeof(attrs_));
    dattrs_ptr_ = static_cast<Attrs*>(buffers.at(0).get());
}

void DetectionOutput::operator()(const CUDA::Stream& stream,
                                 CUDA::DevicePointer<const void*> location,
                                 CUDA::DevicePointer<const void*> confidence,
                                 CUDA::DevicePointer<const void*> priors,
                                 const void* armLocation,
                                 const void* armConfidence,
                                 std::vector<CUDA::DevicePointer<void*>> mutableWorkbuffers,
                                 CUDA::DevicePointer<void*> result) const {
    switch (element_type_) {
#ifdef CUDA_HAS_BF16_TYPE
        case Type_t::bf16:
            return call<__nv_bfloat16>(
                stream, location, confidence, priors, armLocation, armConfidence, mutableWorkbuffers, result);
#endif
        case Type_t::f16:
            return call<__half>(
                stream, location, confidence, priors, armLocation, armConfidence, mutableWorkbuffers, result);
        case Type_t::f32:
            return call<float>(
                stream, location, confidence, priors, armLocation, armConfidence, mutableWorkbuffers, result);
        case Type_t::f64:
            return call<double>(
                stream, location, confidence, priors, armLocation, armConfidence, mutableWorkbuffers, result);
        default:
            throwIEException(
                fmt::format("Input element type = {} is not supported by Split operation "
                            "!!",
                            static_cast<Type_t>(element_type_)));
    }
}

template <typename TDataType>
void DetectionOutput::call(const CUDA::Stream& stream,
                           CUDA::DevicePointer<const void*> location,
                           CUDA::DevicePointer<const void*> confidence,
                           CUDA::DevicePointer<const void*> priors,
                           const void* armConfidence,
                           const void* armLocation,
                           std::vector<CUDA::DevicePointer<void*>> mutableWorkbuffers,
                           CUDA::DevicePointer<void*> result) const {
    auto locData = CUDA::Span<const TDataType>(static_cast<const TDataType*>(location.get()), location_size_);
    auto confData = CUDA::Span<const TDataType>(static_cast<const TDataType*>(confidence.get()), confidence_size_);
    auto priorData = CUDA::Span<const TDataType>(static_cast<const TDataType*>(priors.get()), priors_size_);
    auto results = CUDA::Span<DetectionOutputResult<TDataType>>(
        static_cast<DetectionOutputResult<TDataType>*>(result.get()), result_size_);

    auto& dattrs = *dattrs_ptr_;

    assertThrow(location_size_ / (4 * attrs_.num_images * (attrs_.share_location ? 1 : attrs_.num_classes)) ==
                    attrs_.num_priors,
                "location_size_ / (4 * attrs_.num_images * (attrs_.share_location ? 1 : attrs_.num_classes)) != "
                "attrs_.num_priors");
    auto locPreds = CUDA::MDSpan<NormalizedBBox<TDataType>, CUDA::DExtents<3>>{
        mutableWorkbuffers[kLocationsWBIdx].get(), attrs_.num_images, attrs_.num_loc_classes, attrs_.num_priors};
    assertThrow(confidence_size_ / (attrs_.num_images * attrs_.num_classes) == attrs_.num_priors,
                "confidence_size_ / (attrs_.num_images * attrs_.num_classes) != attrs_.num_priors");
    auto confPreds = CUDA::MDSpan<TDataType, CUDA::DExtents<3>>{
        mutableWorkbuffers[kConfPredsWBIdx].get(), attrs_.num_images, attrs_.num_classes, attrs_.num_priors};
    auto priorBboxes = CUDA::MDSpan<NormalizedBBox<TDataType>, CUDA::DExtents<2>>{
        mutableWorkbuffers[kPriorBboxesWBIdx].get(), attrs_.num_images, attrs_.num_priors};
    auto priorVariances = CUDA::MDSpan<CUDA::Array<TDataType, 4>, CUDA::DExtents<2>>{
        mutableWorkbuffers[kPriorVariancesWBIdx].get(), attrs_.num_images, attrs_.num_priors};
    auto decodeBboxes = CUDA::MDSpan<NormalizedBBox<TDataType>, CUDA::DExtents<3>>{
        mutableWorkbuffers[kDecodeBboxesWBIdx].get(), attrs_.num_images, attrs_.num_loc_classes, attrs_.num_priors};
    auto tempDecodeBboxes = CUDA::MDSpan<NormalizedBBox<TDataType>, CUDA::DExtents<3>>{
        mutableWorkbuffers[kTempDecodeBboxesWBIdx].get(), attrs_.num_images, attrs_.num_loc_classes, attrs_.num_priors};
    auto numDets =
        CUDA::Span<CUDA::DeviceAtomic<unsigned>>{mutableWorkbuffers[kNumDetectionsWBIdx].get(), attrs_.num_images};
    auto tempScorePerClassPrioIdxs0 = CUDA::MDVector<CUDA::Pair<TDataType, CUDA::Pair<int, int>>, 2>{
        stream,
        attrs_.num_priors,
        mutableWorkbuffers[kTempScorePerClassPrioIdxs0WBIdx].get(),
        attrs_.num_images,
        attrs_.num_classes};
    auto tempScorePerClassPrioIdxs1 = CUDA::MDVector<CUDA::Pair<TDataType, CUDA::Pair<int, int>>, 1>{
        stream,
        attrs_.num_classes * attrs_.num_priors,
        mutableWorkbuffers[kTempScorePerClassPrioIdxs1WBIdx].get(),
        attrs_.num_images};
    auto prioBoxIdxsByClass = CUDA::MDVector<int, 2>{stream,
                                                     attrs_.num_priors,
                                                     mutableWorkbuffers[kPrioBoxIdxsByClassWBIdx].get(),
                                                     attrs_.num_images,
                                                     attrs_.num_classes};

    const unsigned num_blocks = (attrs_.num_images % max_threads_per_block_ == 0)
                                    ? (attrs_.num_images / max_threads_per_block_)
                                    : (attrs_.num_images / max_threads_per_block_ + 1);
    const unsigned threads_per_block = (num_blocks == 1) ? attrs_.num_images : max_threads_per_block_;
    detection_output_initialization<TDataType><<<num_blocks, threads_per_block, 0, stream.get()>>>(numDets);

    const unsigned num_blocks_priors = (attrs_.num_priors % max_threads_per_block_ == 0)
                                           ? (attrs_.num_priors / max_threads_per_block_)
                                           : (attrs_.num_priors / max_threads_per_block_ + 1);
    const unsigned threads_per_block_priors = (num_blocks_priors == 1) ? attrs_.num_priors : max_threads_per_block_;

    if (arm_confidence_size_ > 0) {
        auto armConfData =
            CUDA::Span<const TDataType>(static_cast<const TDataType*>(armConfidence), arm_confidence_size_);
        auto armLocData = CUDA::Span<const TDataType>(static_cast<const TDataType*>(armLocation), arm_location_size_);
        auto armLocPreds = CUDA::MDSpan<NormalizedBBox<TDataType>, CUDA::DExtents<3>>{
            mutableWorkbuffers[kArmLocationsWBIdx].get(), attrs_.num_images, attrs_.num_loc_classes, attrs_.num_priors};
        detection_output_stage_0_os_confidence_scores<<<dim3(attrs_.num_images, attrs_.num_classes, num_blocks_priors),
                                                        threads_per_block_priors,
                                                        0,
                                                        stream.get()>>>(dattrs, confData, armConfData, confPreds);
        detection_output_stage_0_get_loc_predictions_with_arm<<<
            dim3(attrs_.num_images, attrs_.num_loc_classes, num_blocks_priors),
            threads_per_block_priors,
            0,
            stream.get()>>>(dattrs, locData, armLocData, locPreds, armLocPreds);
        detection_output_stage_0_get_prior_bboxes<<<dim3(attrs_.num_images, 1, num_blocks_priors),
                                                    threads_per_block_priors,
                                                    0,
                                                    stream.get()>>>(dattrs, priorData, priorBboxes, priorVariances);
        detection_output_stage_0_cas_decode_bboxes_all<<<
            dim3(attrs_.num_images, attrs_.num_loc_classes, num_blocks_priors),
            threads_per_block_priors,
            0,
            stream.get()>>>(dattrs, locPreds, priorBboxes, priorVariances, armLocPreds, tempDecodeBboxes, decodeBboxes);
    } else {
        detection_output_stage_0_confidence_scores<<<dim3(attrs_.num_images, attrs_.num_classes, num_blocks_priors),
                                                     threads_per_block_priors,
                                                     0,
                                                     stream.get()>>>(dattrs, confData, confPreds);
        detection_output_stage_0_get_loc_predictions<<<
            dim3(attrs_.num_images, attrs_.num_loc_classes, num_blocks_priors),
            threads_per_block_priors,
            0,
            stream.get()>>>(dattrs, locData, locPreds);
        detection_output_stage_0_get_prior_bboxes<<<dim3(attrs_.num_images, 1, num_blocks_priors),
                                                    threads_per_block_priors,
                                                    0,
                                                    stream.get()>>>(dattrs, priorData, priorBboxes, priorVariances);
        detection_output_stage_0_decode_bboxes_all<<<dim3(attrs_.num_images, attrs_.num_loc_classes, num_blocks_priors),
                                                     threads_per_block_priors,
                                                     0,
                                                     stream.get()>>>(
            dattrs, locPreds, priorBboxes, priorVariances, decodeBboxes);
    }

#ifdef CUDA_KERNEL_PRINT_LOG
    debug::detection_output_stage_0_debug<<<1, 1, 0, stream.get()>>>(
        dattrs, confPreds, locPreds, priorBboxes, priorVariances, decodeBboxes);
#endif

    if (!attrs_.decrease_label_id) {
        detection_output_stage_1_caffe_nms<<<dim3(attrs_.num_images, attrs_.num_classes, 1), 1, 0, stream.get()>>>(
            dattrs, decodeBboxes, confPreds, tempScorePerClassPrioIdxs0, prioBoxIdxsByClass, numDets);
    } else {
        detection_output_stage_1_mxnet_nms<<<dim3(attrs_.num_images), 1, 0, stream.get()>>>(
            dattrs, decodeBboxes, confPreds, tempScorePerClassPrioIdxs1, prioBoxIdxsByClass, numDets);
    }

#ifdef CUDA_KERNEL_PRINT_LOG
    debug::detection_output_stage_1_debug<TDataType><<<1, 1, 0, stream.get()>>>(dattrs, prioBoxIdxsByClass, numDets);
#endif

    detection_output_stage_1_keep_top_scores<<<dim3(attrs_.num_images), 1, 0, stream.get()>>>(
        dattrs, confPreds, tempScorePerClassPrioIdxs1, prioBoxIdxsByClass, numDets);

#ifdef CUDA_KERNEL_PRINT_LOG
    debug::detection_output_stage_1_debug<TDataType><<<1, 1, 0, stream.get()>>>(dattrs, prioBoxIdxsByClass, numDets);
#endif

    detection_output_stage_2_results<<<dim3(attrs_.num_images), 1, 0, stream.get()>>>(
        dattrs, confPreds, decodeBboxes, prioBoxIdxsByClass, numDets, results);

#ifdef CUDA_KERNEL_PRINT_LOG
    debug::detection_output_stage_2_debug<TDataType><<<1, 1, 0, stream.get()>>>(dattrs, resultData, numDets);
#endif
}

}  // namespace kernel
}  // namespace nvidia_gpu
}  // namespace ov
