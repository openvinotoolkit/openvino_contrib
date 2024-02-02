#include <map>
#include <vector>
#include <cinttypes>

#include "cpu_ops.hpp"

namespace {

template <typename scalar_t>
void reshape_and_cache_cpu_impl(
    const scalar_t *__restrict__ key, const scalar_t *__restrict__ value,
    scalar_t *__restrict__ key_cache, scalar_t *__restrict__ value_cache,
    const int64_t *__restrict__ slot_mapping, const int num_tokens,
    const int key_stride, const int value_stride, const int num_heads,
    const int head_size, const int block_size, const int x) {
  const int block_elem_num = num_heads * head_size * block_size;

#pragma omp parallel for collapse(2)
  for (int token_idx = 0; token_idx < num_tokens; ++token_idx) {
    for (int head_idx = 0; head_idx < num_heads; ++head_idx) {
      const int64_t slot_idx = slot_mapping[token_idx];
      if (slot_idx >= 0) {
        int src_key_head_idx = token_idx * key_stride + head_idx * head_size;
        int src_value_head_idx =
            token_idx * value_stride + head_idx * head_size;
        const scalar_t *src_key_head_ptr = key + src_key_head_idx;
        const scalar_t *src_value_head_ptr = value + src_value_head_idx;
        const int64_t block_index = slot_idx / block_size;
        const int64_t block_offset = slot_idx % block_size;
        scalar_t *target_key_head_ptr = key_cache +
                                        block_elem_num * block_index +
                                        head_idx * block_size * head_size;
        scalar_t *target_value_head_ptr = value_cache +
                                          block_elem_num * block_index +
                                          head_idx * block_size * head_size;

        for (int src_key_idx = 0; src_key_idx < head_size; src_key_idx += x) {
          const int64_t target_offset =
              src_key_idx * block_size + block_offset * x;
          for (int i = 0; i < x; ++i) {
            target_key_head_ptr[target_offset + i] =
                src_key_head_ptr[src_key_idx + i];
          }
        }

        for (int src_value_idx = 0; src_value_idx < head_size;
             ++src_value_idx) {
          const int64_t target_offset =
              src_value_idx * block_size + block_offset;
          target_value_head_ptr[target_offset] =
              src_value_head_ptr[src_value_idx];
        }
      }
    }
  }
}
}; // namespace

void reshape_and_cache(ov::Tensor key, ov::Tensor value,
                       ov::Tensor key_cache, ov::Tensor value_cache,
                       ov::Tensor slot_mapping) {
  ov::Shape key_shape = key.get_shape(), key_cache_shape = key_cache.get_shape();
  int num_tokens = key_shape[0];
  int num_heads = key_shape[1];
  int head_size = key_shape[2];
  int block_size = key_cache_shape[3];
  int x = key_cache_shape[4];

  ov::Strides key_strides = key.get_strides();
  int key_stride = key_strides[0];
  ov::Strides value_strides = value.get_strides();
  int value_stride = value_strides[0];

    switch (key.get_element_type()) {
        case ov::element::f32:
        case ov::element::i32:
        case ov::element::u32:
            reshape_and_cache_cpu_impl<float>(
                    key.data<float>(), value.data<float>(),
                    key_cache.data<float>(), value_cache.data<float>(),
                    slot_mapping.data<int64_t>(), num_tokens, key_stride,
                    value_stride, num_heads, head_size, block_size, x);
            break;
        case ov::element::f16:
        case ov::element::i16:
        case ov::element::u16:
            reshape_and_cache_cpu_impl<std::int16_t>(
                    key.data<std::int16_t>(), value.data<std::int16_t>(),
                    key_cache.data<std::int16_t>(), value_cache.data<std::int16_t>(),
                    slot_mapping.data<int64_t>(), num_tokens, key_stride,
                    value_stride, num_heads, head_size, block_size, x);
            break;
        default:
            OPENVINO_THROW("Unsupported key type");
    };
}
