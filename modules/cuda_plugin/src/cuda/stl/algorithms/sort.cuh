// Copyright (C) 2021-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <cuda/math.cuh>
#include <cuda/stl/pair.cuh>

namespace CUDA {
namespace algorithms {

/**
 * @brief Greater comparator
 */
template <typename T>
struct Greater {
    __device__ bool operator()(const T& l, const T& r) { return l > r; }
};

/**
 * @brief Less comparator
 */
template <typename T>
struct Less {
    __device__ bool operator()(const T& l, const T& r) { return l < r; }
};

/**
 * @brief Swaps first and second arguments
 */
template <typename T>
__device__ void swap(T& a, T& b) {
    T t = a;
    a = b;
    b = t;
}

/**
 * Partition sequence between begin and end on smaller elements
 * then pivot and bigger elements than pivot
 * @tparam T Type of data to sort
 * @tparam Comparer Type of custom comparer
 * @param arr Sequence to partition by indexes
 * @param l First index of element in @param arr to partition
 * @param h Last index (including) of element in @param arr to partition
 * @param comparer Custom comparer
 * @return Returns index to pivot element after partitioning
 */
template <typename T, typename Comparer>
__device__ int partition(T* arr, int l, int h, Comparer comparer) {
    int pivot_id = (h - l) / 2 + l;
    swap(arr[pivot_id], arr[h]);
    const T pivot = arr[h];

    int i = (l - 1);
    for (int j = l; j <= h - 1; j++) {
        if (comparer(arr[j], pivot)) {
            i++;
            swap(arr[i], arr[j]);
        }
    }
    swap(arr[i + 1], arr[h]);
    return (i + 1);
}

/**
 * Bubble sort iterative algorithm
 * @tparam T Type of data to sort
 * @tparam Comparer Type of custom comparer
 * @param begin Begin of sequence to iterate
 * @param end End of sequence to iterate (last element + 1)
 * @param comparer Custom comparer
 */
template <typename T, typename Comparer>
__device__ void bubble_sort_iterative(T* begin, T* end, Comparer comparer) {
    if (end <= begin) {
        return;
    }

    const int l = 0;
    const int h = (end - begin) - 1;
    for (int i = 0; i < h - l; i++) {
        bool swapped = false;
        for (int j = l; j < h - i; j++) {
            if (comparer(begin[j], begin[j + 1])) {
                swap(begin[j], begin[j + 1]);
                swapped = true;
            }
        }
        if (!swapped) {
            break;
        }
    }
}

/**
 * Quick sort iterative algorithm
 * @tparam T Type of data to sort
 * @tparam Comparer Type of custom comparer
 * @param begin Begin of sequence to iterate
 * @param end End of sequence to iterate (last element + 1)
 * @param comparer Custom comparer
 */
template <typename T, typename Comparer>
__device__ void quick_sort_iterative(T* begin, T* end, Comparer comparer) {
    if (end <= begin) {
        return;
    }

    constexpr const int kStackSize = 100;
    int stack[kStackSize];

    int l = 0;
    int h = (end - begin) - 1;
    int top = -1;

    stack[++top] = l;
    stack[++top] = h;

    while (top >= 0) {
        h = stack[top--];
        l = stack[top--];

        const int p = partition(begin, l, h, comparer);

        const int before_p = p - 1;
        const int after_p = p + 1;
        if (before_p > l) {
            if (top >= (kStackSize - 1)) {
                bubble_sort_iterative(begin + l, begin + p, comparer);
            } else {
                stack[++top] = l;
                stack[++top] = before_p;
            }
        }

        if (after_p < h) {
            if (top >= (kStackSize - 1)) {
                bubble_sort_iterative(begin + after_p, begin + h + 1, comparer);
            } else {
                stack[++top] = after_p;
                stack[++top] = h;
            }
        }
    }
}

/**
 * Quick sort iterative algorithm
 * @tparam T Type of data to sort
 * @tparam Comparer Type of custom comparer
 * @param begin Begin of sequence to iterate
 * @param end End of sequence to iterate (last element + 1)
 * @param stack Additional workspace for storing partition pivots
 * @param comparer Custom comparer
 */
template <typename T, typename Comparer>
__device__ void quick_sort_iterative(T* begin, T* end, int* stack, Comparer comparer) {
    if (end <= begin) {
        return;
    }

    int l = 0;
    int h = (end - begin) - 1;
    int top = -1;

    stack[++top] = l;
    stack[++top] = h;

    while (top >= 0) {
        h = stack[top--];
        l = stack[top--];

        const int p = partition(begin, l, h, comparer);

        const int before_p = p - 1;
        const int after_p = p + 1;
        if (before_p > l) {
            stack[++top] = l;
            stack[++top] = before_p;
        }

        if (after_p < h) {
            stack[++top] = after_p;
            stack[++top] = h;
        }
    }
}

/**
 * Partial quick sort iterative algorithm
 * @tparam T Type of data to sort
 * @tparam Comparer Type of custom comparer
 * @param begin Begin of sequence to iterate
 * @param end End of sequence to iterate (last element + 1)
 * @param stack Additional workspace for storing partition pivots
 * @param topk Number of elements to keep sorting from the beginning of the sequence
 * @param comparer Custom comparer
 */
template <typename T, typename Comparer>
__device__ void partial_quick_sort_iterative(T* begin, T* end, int* stack, size_t topk, Comparer comparer) {
    if (end <= begin) {
        return;
    }

    int l = 0;
    int h = (end - begin) - 1;
    int top = -1;

    stack[++top] = l;
    stack[++top] = h;

    while (top >= 0) {
        h = stack[top--];
        l = stack[top--];

        const int p = partition(begin, l, h, comparer);

        const int before_p = p - 1;
        const int after_p = p + 1;
        if (before_p > l) {
            if (l <= topk) {
                stack[++top] = l;
                stack[++top] = before_p;
            }
        }

        if (after_p < h) {
            if (after_p <= topk) {
                stack[++top] = after_p;
                stack[++top] = h;
            }
        }
    }
}

/**
 * Partial quick sort iterative algorithm
 * @tparam T Type of data to sort
 * @tparam Comparer Type of custom comparer
 * @param begin Begin of sequence to iterate
 * @param end End of sequence to iterate (last element + 1)
 * @param topk Number of elements to keep sorting from the beginning of the sequence
 * @param comparer Custom comparer
 */
template <typename T, typename Comparer>
__device__ void partial_quick_sort_iterative(T* begin, T* end, size_t topk, Comparer comparer) {
    if (end <= begin) {
        return;
    }

    constexpr const int kStackSize = 100;
    int stack[kStackSize];

    int l = 0;
    int h = (end - begin) - 1;
    int top = -1;

    stack[++top] = l;
    stack[++top] = h;

    while (top >= 0) {
        h = stack[top--];
        l = stack[top--];

        const int p = partition(begin, l, h, comparer);

        const int before_p = p - 1;
        const int after_p = p + 1;
        if (before_p > l) {
            if (l <= topk) {
                if (top >= (kStackSize - 1)) {
                    bubble_sort_iterative(begin + l, begin + p, comparer);
                } else {
                    stack[++top] = l;
                    stack[++top] = before_p;
                }
            }
        }

        if (after_p < h) {
            if (after_p <= topk) {
                if (top >= (kStackSize - 1)) {
                    bubble_sort_iterative(begin + after_p, begin + h + 1, comparer);
                } else {
                    stack[++top] = after_p;
                    stack[++top] = h;
                }
            }
        }
    }
}

namespace parallel {

/**
 * Parallel quick sort iterative algorithm
 * @tparam NumPartitions  The maximum number of partitions
 * @tparam T Type of data to sort
 * @tparam Comparer Type of custom comparer
 * @param begin Begin of sequence to iterate
 * @param end End of sequence to iterate (last element + 1)
 * @param stack Additional workspace for storing partition pivots
 * @param comparer Custom comparer
 * @param workItemId Id of work item
 * @param numWorkItems Number of work items
 */
template <size_t NumPartitions, typename T, typename Comparer>
__forceinline__ __device__ void quick_sort_iterative(
    T* begin, T* end, int* stack, Comparer comparer, const int workItemId, const size_t numWorkItems) {
    if (end <= begin) {
        return;
    }

    __shared__ int partitions[NumPartitions][2];
    if (workItemId < NumPartitions) {
        if (workItemId == 0) {
            partitions[workItemId][0] = 0;
            partitions[workItemId][1] = (end - begin) - 1;
        } else {
            partitions[workItemId][0] = 0;
            partitions[workItemId][1] = 0;
        }
    }
    __syncthreads();

    const int first_id = workItemId;
    for (int range_step = 1, maxWorkingNum = 1, chunkSize = end - begin;
         range_step < numWorkItems && range_step < NumPartitions;
         range_step *= 2, maxWorkingNum *= 2, chunkSize /= 2) {
        const int second_id = first_id + range_step;
        if (second_id < numWorkItems && second_id < NumPartitions) {
            const int begin_id = partitions[first_id][0];
            const int end_id = partitions[first_id][1];
            if (begin_id < end_id) {
                const int pivot = algorithms::partition(begin, begin_id, end_id, comparer);
                partitions[first_id][0] = begin_id;
                partitions[first_id][1] = CUDA::math::max(pivot - 1, begin_id);
                partitions[second_id][0] = CUDA::math::min(pivot + 1, end_id);
                partitions[second_id][1] = end_id;
            }
        }
        __syncthreads();
    }

    if (workItemId < NumPartitions) {
        auto begin_id = partitions[workItemId][0];
        auto end_id = partitions[workItemId][1];
        if (begin_id < end_id) {
            algorithms::quick_sort_iterative(begin + begin_id, begin + end_id + 1, stack + begin_id, comparer);
        }
    }
}

/**
 * Parallel quick sort iterative algorithm
 * @tparam NumPartitions  The maximum number of partitions
 * @tparam T Type of data to sort
 * @tparam Comparer Type of custom comparer
 * @param begin Begin of sequence to iterate
 * @param end End of sequence to iterate (last element + 1)
 * @param comparer Custom comparer
 * @param workItemId Id of work item
 * @param numWorkItems Number of work items
 */
template <size_t NumPartitions, typename T, typename Comparer>
__forceinline__ __device__ void quick_sort_iterative(
    T* begin, T* end, Comparer comparer, const int workItemId, const size_t numWorkItems) {
    if (end <= begin) {
        return;
    }

    __shared__ int partitions[NumPartitions][2];
    if (workItemId < NumPartitions) {
        if (workItemId == 0) {
            partitions[workItemId][0] = 0;
            partitions[workItemId][1] = (end - begin) - 1;
        } else {
            partitions[workItemId][0] = 0;
            partitions[workItemId][1] = 0;
        }
    }
    __syncthreads();

    for (int range_step = 1; range_step < numWorkItems && range_step < NumPartitions; range_step *= 2) {
        const int first_id = workItemId;
        const int second_id = first_id + range_step;
        if (second_id < numWorkItems && second_id < NumPartitions) {
            const int begin_id = partitions[first_id][0];
            const int end_id = partitions[first_id][1];
            if (begin_id < end_id) {
                const int pivot = algorithms::partition(begin, begin_id, end_id, comparer);
                partitions[first_id][0] = begin_id;
                partitions[first_id][1] = CUDA::math::max(pivot - 1, begin_id);
                partitions[second_id][0] = CUDA::math::min(pivot + 1, end_id);
                partitions[second_id][1] = end_id;
            }
        }
        __syncthreads();
    }

    if (workItemId < NumPartitions) {
        auto begin_id = partitions[workItemId][0];
        auto end_id = partitions[workItemId][1];
        if (begin_id < end_id) {
            algorithms::quick_sort_iterative(begin + begin_id, begin + end_id + 1, comparer);
        }
    }
}

}  // namespace parallel

}  // namespace algorithms
}  // namespace CUDA
