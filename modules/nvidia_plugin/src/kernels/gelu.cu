#include "gelu.hpp"

namespace ov {
namespace nvidia_gpu {
namespace kernel {

namespace cumath = CUDA::math;

template <typename T>
struct GeluErfOpImpl {
    __device__ static inline T op(T x) {
        return static_cast<T>(0.5) * x * (static_cast<T>(1.0) +
            static_cast<T>(cumath::erff(x / static_cast<T>(cumath::sqrt(static_cast<T>(2.0))))));
    }
};

template <typename T>
struct GeluTanhOpImpl {
    __device__ static inline T op(T x) {
        return static_cast<T>(0.5) * x * (static_cast<T>(1.0) +
            static_cast<T>(cumath::tanh(static_cast<T>(0.7978845608028654) * (x + static_cast<T>(0.044715) * x * x * x))));
    }
};

GeluErf::GeluErf(Type_t element_type, size_t max_threads_per_block, size_t num_elements)
    : impl_{element_type, max_threads_per_block, num_elements} {}

void GeluErf::operator()(cudaStream_t stream, const void* in0, void* out) const {
    impl_(stream, in0, out);
}

GeluTanh::GeluTanh(Type_t element_type, size_t max_threads_per_block, size_t num_elements)
    : impl_{element_type, max_threads_per_block, num_elements} {}

void GeluTanh::operator()(cudaStream_t stream, const void* in0, void* out) const {
    impl_(stream, in0, out);
}
}  // namespace kernel
}  // namespace nvidia_gpu
}  // namespace ov
