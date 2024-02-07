#include "cpu_types.hpp"
#include "cpu_ops.hpp"

namespace {

// template <typename scalar_t, int HEAD_SIZE, int BLOCK_SIZE>
// struct paged_attention_v1_impl {
//   static void
//   call(scalar_t *__restrict__ out,           // [num_seqs, num_heads, head_size]
//        const scalar_t *__restrict__ q,       // [num_seqs, num_heads, head_size]
//        const scalar_t *__restrict__ k_cache, // [num_blocks, num_kv_heads,
//                                              // head_size/x, block_size, x]
//        const scalar_t *__restrict__ v_cache, // [num_blocks, num_kv_heads,
//                                              // head_size, block_size]
//        const int num_kv_heads, const float scale,
//        const int
//            *__restrict__ block_tables, // [num_seqs, max_num_blocks_per_seq]
//        const int *__restrict__ context_lens, // [num_seqs]
//        const int max_num_blocks_per_seq,
//        const int q_stride, const int kv_block_stride, const int kv_head_stride,
//        const int num_seqs, const int num_heads) {
//     OPENVINO_ASSERT(HEAD_SIZE % 16 == 0);
//     constexpr int x = 16 / sizeof(scalar_t);
//     const int num_queries_per_kv = num_heads / num_kv_heads;

//     int max_context_len = max_num_blocks_per_seq * BLOCK_SIZE;
//     int max_context_len_padded = (max_context_len + 15) & 0xFFFFFFF0;
//     OPENVINO_ASSERT((max_context_len_padded * sizeof(float)) % 64 == 0);

//     size_t logits_bytes = num_heads * max_context_len_padded * sizeof(float);
//     float *logits = (float *)std::aligned_alloc(
//         64, logits_bytes); // Cacheline alignment for each context token.
//                            // [head_num, max_context_len_padded]

//     std::memset(out, 0, num_seqs * num_heads * HEAD_SIZE * sizeof(scalar_t));

//     for (int seq_idx = 0; seq_idx < num_seqs; ++seq_idx) {
//       int context_len = context_lens[seq_idx];
//       const int *seq_block_table =
//           block_tables + max_num_blocks_per_seq * seq_idx;
//       const int block_num = (context_len + BLOCK_SIZE - 1) / BLOCK_SIZE;
//       std::memset(logits, 0, logits_bytes);

//       // Compute attention logits
// #pragma omp parallel for collapse(2)
//       for (int block_idx = 0; block_idx < block_num; ++block_idx) {
//         for (int head_idx = 0; head_idx < num_heads; ++head_idx) {
//           const int64_t kv_head_idx = head_idx / num_queries_per_kv;
//           const int64_t physical_block_idx = seq_block_table[block_idx];
//           const scalar_t *__restrict__ q_vec_ptr =
//               q + seq_idx * q_stride + head_idx * HEAD_SIZE;
//           const scalar_t *__restrict__ k_block_cache_ptr =
//               k_cache + physical_block_idx * kv_block_stride +
//               kv_head_idx * kv_head_stride;
//           float *__restrict__ head_block_logits =
//               logits + head_idx * max_context_len_padded +
//               block_idx * BLOCK_SIZE;

//           for (int q_offset = 0; q_offset < HEAD_SIZE;
//                q_offset += x, q_vec_ptr += x) {
//             for (int token_idx = 0; token_idx < BLOCK_SIZE;
//                  ++token_idx, k_block_cache_ptr += x) {
//               for (int i = 0; i < x; ++i) {
//                 head_block_logits[token_idx] +=
//                     q_vec_ptr[i] * k_block_cache_ptr[i] * scale;
//               }
//             }
//           }
//         }
//       }

//       // Compute softmax
// #pragma omp parallel for
//       for (int head_idx = 0; head_idx < num_heads; ++head_idx) {
//         float *head_logit_ptr = logits + head_idx * max_context_len_padded;
//         float max_logit = head_logit_ptr[0];
//         for (int i = 1; i < context_len; ++i) {
//           max_logit =
//               max_logit >= head_logit_ptr[i] ? max_logit : head_logit_ptr[i];
//         }

//         float sum = 0;
//         for (int i = 0; i < context_len; ++i) {
//           head_logit_ptr[i] = std::exp(head_logit_ptr[i] - max_logit);
//           sum += head_logit_ptr[i];
//         }

//         for (int i = 0; i < context_len; ++i) {
//           head_logit_ptr[i] /= sum;
//         }

//         int remaining_seq_upper = block_num * BLOCK_SIZE;
//         for (int i = context_len; i < remaining_seq_upper; ++i) {
//           head_logit_ptr[i] = 0;
//         }
//       }

//       // Compute value
//       constexpr int head_partition_num = HEAD_SIZE / 16;
// #pragma omp parallel for collapse(2)
//       for (int head_idx = 0; head_idx < num_heads; ++head_idx) {
//         for (int head_part_idx = 0; head_part_idx < head_partition_num;
//              ++head_part_idx) {
//           for (int block_idx = 0; block_idx < block_num; ++block_idx) {
//             const int64_t kv_head_idx = head_idx / num_queries_per_kv;
//             const int64_t physical_block_idx = seq_block_table[block_idx];
//             const float *__restrict__ prob_vec_ptr =
//                 logits + head_idx * max_context_len_padded +
//                 block_idx * BLOCK_SIZE;
//             const scalar_t *__restrict__ v_block_cache_ptr =
//                 v_cache + physical_block_idx * kv_block_stride +
//                 kv_head_idx * kv_head_stride + BLOCK_SIZE * head_part_idx * 16;
//             scalar_t *__restrict__ out_ptr =
//                 out + seq_idx * num_heads * HEAD_SIZE + head_idx * HEAD_SIZE +
//                 head_part_idx * 16;

//             for (int i = 0; i < 16; ++i, v_block_cache_ptr += BLOCK_SIZE) {
//               for (int j = 0; j < BLOCK_SIZE; ++j) {
//                 out_ptr[i] += prob_vec_ptr[j] * v_block_cache_ptr[j];
//               }
//             }
//           }
//         }
//       }
//     }
//     std::free(logits);
//   }
// };


template <typename scalar_t, int HEAD_SIZE, int BLOCK_SIZE>
struct paged_attention_v1_impl {
  static void
  call(scalar_t *__restrict__ out,           // [num_seqs, num_heads, head_size]
       const scalar_t *__restrict__ q,       // [num_seqs, num_heads, head_size]
       const scalar_t *__restrict__ k_cache, // [num_blocks, num_kv_heads,
                                             // head_size/x, block_size, x]
       const scalar_t *__restrict__ v_cache, // [num_blocks, num_kv_heads,
                                             // head_size, block_size]
       const int num_kv_heads, const float scale,
       const int
           *__restrict__ block_tables, // [num_seqs, max_num_blocks_per_seq]
       const int *__restrict__ context_lens, // [num_seqs]
       const int max_num_blocks_per_seq,
       const int q_stride, const int kv_block_stride, const int kv_head_stride,
       const int num_seqs, const int num_heads) {
    // TORCH_CHECK(HEAD_SIZE % 16 == 0);
    // TORCH_CHECK(alibi_slopes == nullptr, "Unsupport alibi_slopes for CPU");
    constexpr int x = 16 / sizeof(scalar_t);
    const int num_queries_per_kv = num_heads / num_kv_heads;

    int max_context_len = max_num_blocks_per_seq * BLOCK_SIZE;
    int max_context_len_padded = (max_context_len + 15) & 0xFFFFFFF0;
    // TORCH_CHECK((max_context_len_padded * sizeof(float)) % 64 == 0);

    size_t logits_bytes = num_heads * max_context_len_padded * sizeof(float);
    float *logits = (float *)std::aligned_alloc(
        64, logits_bytes); // Cacheline alignment for each context token.
                           // [head_num, max_context_len_padded]

    std::memset(out, 0, num_seqs * num_heads * HEAD_SIZE * sizeof(scalar_t));

    for (int seq_idx = 0; seq_idx < num_seqs; ++seq_idx) {
      int context_len = context_lens[seq_idx];
      const int *seq_block_table =
          block_tables + max_num_blocks_per_seq * seq_idx;
      const int block_num = (context_len + BLOCK_SIZE - 1) / BLOCK_SIZE;
      std::memset(logits, 0, logits_bytes);

      // Compute attention logits
#pragma omp parallel for collapse(2)
      for (int block_idx = 0; block_idx < block_num; ++block_idx) {
        for (int head_idx = 0; head_idx < num_heads; ++head_idx) {
          const int64_t kv_head_idx = head_idx / num_queries_per_kv;
          const int64_t physical_block_idx = seq_block_table[block_idx];
          const scalar_t *__restrict__ q_vec_ptr =
              q + seq_idx * q_stride + head_idx * HEAD_SIZE;
          const scalar_t *__restrict__ k_block_cache_ptr =
              k_cache + physical_block_idx * kv_block_stride +
              kv_head_idx * kv_head_stride;
          float *__restrict__ head_block_logits =
              logits + head_idx * max_context_len_padded +
              block_idx * BLOCK_SIZE;

          for (int q_offset = 0; q_offset < HEAD_SIZE;
               q_offset += x, q_vec_ptr += x) {
            for (int token_idx = 0; token_idx < BLOCK_SIZE;
                 ++token_idx, k_block_cache_ptr += x) {
              for (int i = 0; i < x; ++i) {
                head_block_logits[token_idx] +=
                    q_vec_ptr[i] * k_block_cache_ptr[i] * scale;
              }
            }
          }
        }
      }

      // std::cout << std::endl;
      // for (int i = 0; i < 40; ++i)
      //   std::cout << logits[i] << " ";
      // exit(1);

      // Compute softmax
#pragma omp parallel for
      for (int head_idx = 0; head_idx < num_heads; ++head_idx) {
        float *head_logit_ptr = logits + head_idx * max_context_len_padded;
        float max_logit = head_logit_ptr[0];
        for (int i = 1; i < context_len; ++i) {
          max_logit =
              max_logit >= head_logit_ptr[i] ? max_logit : head_logit_ptr[i];
        }

        float sum = 0;
        for (int i = 0; i < context_len; ++i) {
          head_logit_ptr[i] = std::exp(head_logit_ptr[i] - max_logit);
          sum += head_logit_ptr[i];
        }

        for (int i = 0; i < context_len; ++i) {
          head_logit_ptr[i] /= sum;
        }

        int remaining_seq_upper = block_num * BLOCK_SIZE;
        for (int i = context_len; i < remaining_seq_upper; ++i) {
          head_logit_ptr[i] = 0;
        }
      }

      // Compute value
      constexpr int head_partition_num = HEAD_SIZE / 16;
#pragma omp parallel for collapse(2)
      for (int head_idx = 0; head_idx < num_heads; ++head_idx) {
        for (int head_part_idx = 0; head_part_idx < head_partition_num;
             ++head_part_idx) {
          for (int block_idx = 0; block_idx < block_num; ++block_idx) {
            const int64_t kv_head_idx = head_idx / num_queries_per_kv;
            const int64_t physical_block_idx = seq_block_table[block_idx];
            const float *__restrict__ prob_vec_ptr =
                logits + head_idx * max_context_len_padded +
                block_idx * BLOCK_SIZE;
            const scalar_t *__restrict__ v_block_cache_ptr =
                v_cache + physical_block_idx * kv_block_stride +
                kv_head_idx * kv_head_stride + BLOCK_SIZE * head_part_idx * 16;
            scalar_t *__restrict__ out_ptr =
                out + seq_idx * num_heads * HEAD_SIZE + head_idx * HEAD_SIZE +
                head_part_idx * 16;

            for (int i = 0; i < 16; ++i, v_block_cache_ptr += BLOCK_SIZE) {
              for (int j = 0; j < BLOCK_SIZE; ++j) {
                out_ptr[i] += prob_vec_ptr[j] * v_block_cache_ptr[j];
              }
            }
          }
        }
      }
    }
    std::free(logits);
  }
};

#define LAUNCH_ATTENTION_KERNEL(T, HEAD_SIZE, BLOCK_SIZE)                      \
  paged_attention_v1_impl<T, HEAD_SIZE, BLOCK_SIZE>::call(                     \
      out_ptr, query_ptr, key_cache_ptr, value_cache_ptr, num_kv_heads, scale, \
      block_tables_ptr, context_lens_ptr, max_num_blocks_per_seq,              \
      q_stride, kv_block_stride, kv_head_stride, num_seqs,   \
      num_heads);

template <typename T, int BLOCK_SIZE>
void paged_attention_v1_impl_launcher(
    ov::Tensor out, ov::Tensor query, ov::Tensor key_cache,
    ov::Tensor value_cache, int num_kv_heads, float scale,
    ov::Tensor block_tables, ov::Tensor context_lens,
    int max_context_len) {
  int num_seqs = query.get_shape()[0];
  int num_heads = query.get_shape()[1];
  int head_size = query.get_shape()[2];
  int max_num_blocks_per_seq = block_tables.get_shape()[1];
  int q_stride = query.get_strides()[0] / query.get_element_type().size();
  int kv_block_stride = key_cache.get_strides()[0] / key_cache.get_element_type().size();
  int kv_head_stride = key_cache.get_strides()[1] / key_cache.get_element_type().size();
  OPENVINO_ASSERT(sizeof(float) == key_cache.get_element_type().size());

  // std::cout << "num_seqs " << num_seqs << std::endl;
  // std::cout << "num_heads " << num_heads << std::endl;
  // std::cout << "head_size " << head_size << std::endl;
  // std::cout << "max_num_blocks_per_seq " << max_num_blocks_per_seq << std::endl;
  // std::cout << "q_stride " << q_stride << std::endl;
  // std::cout << "kv_block_stride " << kv_block_stride << std::endl;
  // std::cout << "kv_head_stride " << kv_head_stride << std::endl;

  T *out_ptr = out.data<float>();
  T *query_ptr = query.data<float>();
  T *key_cache_ptr = key_cache.data<float>();
  T *value_cache_ptr = value_cache.data<float>();
  int *block_tables_ptr = block_tables.data<int>();
  int *context_lens_ptr = context_lens.data<int>();

  switch (head_size) {
  case 64:
    LAUNCH_ATTENTION_KERNEL(T, 64, BLOCK_SIZE);
    break;
  case 80:
    LAUNCH_ATTENTION_KERNEL(T, 80, BLOCK_SIZE);
    break;
  case 96:
    LAUNCH_ATTENTION_KERNEL(T, 96, BLOCK_SIZE);
    break;
  case 112:
    LAUNCH_ATTENTION_KERNEL(T, 112, BLOCK_SIZE);
    break;
  case 128:
    LAUNCH_ATTENTION_KERNEL(T, 128, BLOCK_SIZE);
    break;
  case 256:
    LAUNCH_ATTENTION_KERNEL(T, 256, BLOCK_SIZE);
    break;
  default:
    OPENVINO_ASSERT(false, "Unsupported head size: ", head_size);
    break;
  }
}

#define CALL_KERNEL_LAUNCHER(T, BLOCK_SIZE)                                    \
  paged_attention_v1_impl_launcher<T, BLOCK_SIZE>(                             \
      out, query, key_cache, value_cache, num_kv_heads, scale, block_tables,   \
      context_lens, max_context_len);

#define CALL_KERNEL_LAUNCHER_BLOCK_SIZE(T)                                     \
  switch (block_size) {                                                        \
  case 16:                                                                     \
    CALL_KERNEL_LAUNCHER(T, 16);                                               \
    break;                                                                     \
  default:                                                                     \
    OPENVINO_ASSERT(false, "Unsupported block size: ", block_size);            \
    break;                                                                     \
  }
} // namespace

void paged_attention_v1_cpu(ov::Tensor out, ov::Tensor query,
                            ov::Tensor key_cache,
                            ov::Tensor value_cache, int num_kv_heads,
                            float scale, ov::Tensor block_tables,
                            ov::Tensor context_lens, int block_size,
                            int max_context_len) {
    CALL_KERNEL_LAUNCHER_BLOCK_SIZE(float);
}
