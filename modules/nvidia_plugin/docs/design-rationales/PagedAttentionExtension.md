# Native PagedAttentionExtension for the OpenVINO NVIDIA (CUDA) Plugin — Implementation Design

Status: M0–M2 implemented; M3–M5 planned
Owner: NVIDIA plugin
Target repo: `modules/nvidia_plugin`
References cited inline as `path:line`. OpenVINO core = `.../OpenVINO`, genai = `.../openvino.genai`, plugin = `modules/nvidia_plugin`.

---

## 1. Goal & Scope

### 1.1 Goal
Enable **OpenVINO GenAI continuous batching** (paged KV cache, prefix caching, chunked prefill) to run on NVIDIA GPUs through this plugin by adding a **native `PagedAttentionExtension` op**. Today the plugin has no such op, so any model that GenAI rewrites via `SDPAToPagedAttention` (SDPA replaced by `PagedAttentionExtension`, model made stateless) fails to compile: the op is unknown to `OperationRegistry`, and `key_cache.N/value_cache.N` Parameters arrive with dynamic rank-4 shapes the plugin cannot allocate.

### 1.2 What GenAI hands us (the compile-time contract)
After `apply_paged_attention_transformations` (genai `continuous_batching/paged_attention_transformations.hpp:29`) the plugin receives a **stateless** `ov::Model`:
- Inputs: `input_ids` (i64,`[-1]`) or `inputs_embeds`, `position_ids` (i64,`[-1]`), `key_cache.0..N-1` / `value_cache.0..N-1` (`element::dynamic`, dynamic rank-4), `past_lens`/`subsequence_begins`/`block_indices`/`block_indices_begins` (i32,`[-1]`), `max_context_len` (i32 scalar). `beam_idx`/`attention_mask` are gone; no Variables/ReadValue/Assign.
- Each attention layer = one `PagedAttentionExtension` op, **exactly 25 inputs, 3 outputs**, with `rt_info` ints `num_k_heads / k_head_size / num_v_heads / v_head_size` (`state_management_pattern.cpp:741-744`).

### 1.3 MVP vs. later
**MVP — numerically correct single-GPU continuous batching:**
- Register `PagedAttentionExtension`; accept all 25 inputs, consume 0–12.
- KV-cache write (reshape-and-cache) + paged attention (prefill + decode + mixed batch) for **f16 and f32**, non-quantized cache.
- scale, causal mask, GQA, ALiBi, sliding_window.
- Cache-shape specialization pass (make inputs 3/4 static `[-1, num_kv_heads, 32, head_size]`).
- Device-resident persistent KV cache with correct host↔device coherence for genai's block copy/eviction.
- GPT-2 end-to-end through genai continuous batching.

**Out of MVP (feed empty consts / assert-empty; add later):**
- Score output (output 1) and adaptive-RKV diversity (output 2) — inputs 13, 21–24.
- Cache rotation (14–16), XAttention sparsity (17–19), attention sinks (20).
- Quantized/compressed KV cache (u8/u4/i8).
- Device-resident (RemoteTensor) zero-copy cache. These are Phase 4+.

Under genai default flags, inputs 13–24 arrive as folded empty `Shape{0}`/scalar-`0` Constants.

---

## 2. Op Contract & KV Cache Layout

### 2.1 Input/output mapping we implement
25 inputs, positional. MVP consumes 0–12; validates 13–24 are empty/no-op.

| idx | name | rank/dtype | MVP use |
|----|------|-----------|--------|
| 0 | query | 2, q-dtype `[num_tokens, num_heads*head_size]` | yes |
| 1 | key | 2 `[num_tokens, num_kv_heads*head_size]` | yes (cache write) |
| 2 | value | 2 `[num_tokens, num_kv_heads*v_head_size]` | yes (cache write) |
| 3 | key_cache | 4, cache-dtype | yes |
| 4 | value_cache | 4, cache-dtype | yes |
| 5 | past_lens | 1, i32 | yes |
| 6 | subsequence_begins | 1, i32 | yes |
| 7 | block_indices | 1, i32 | yes |
| 8 | block_indices_begins | 1, i32 | yes |
| 9 | scale | 0, real | yes |
| 10 | sliding_window | 0, i32 | yes |
| 11 | alibi_slopes | 1, real (may be empty) | yes |
| 12 | max_context_len | 0, i32 | yes (host, buffer sizing) |
| 13 | score_aggregation_window | 0/1, i32 | assert empty |
| 14–16 | rotation | i32/f16f32 | assert empty |
| 17–19 | xattention | | assert empty |
| 20 | sinks | 1/4 | assert empty |
| 21–24 | adaptive_rkv | i32 | assert empty |

**Outputs:**
- **Output 0** — attention result, shape `[num_tokens, num_heads*v_head_size]`, last dim = `query[1]*value[1]/key[1]`, dtype = `m_output_type[0]` override else query dtype. **This is the only output the default genai flow consumes.**
- Output 1 — scores, rank-1 dynamic; Output 2 — aux, rank-1 dynamic. MVP allocates them **zero-sized**.

**Head geometry — read from `rt_info`, NOT cache shapes** (cache Parameters arrive dynamic): op rt_info keys `num_k_heads`, `k_head_size`, `num_v_heads`, `v_head_size`. Derived at Execute time:
- `num_tokens = query.shape[0]`; `H = query.shape[1]/k_head_size`; `Hk = num_k_heads`; `h_each_group = H/Hk`.
- `S = k_head_size`, `SV = v_head_size` (may differ). Output width = `H*SV`.
- `B_seq = past_lens.shape[0]`.

### 2.2 KV cache block layout we adopt

**Decision:** adopt the **genai/OpenVINO default layout for BOTH K and V: `[num_blocks, num_kv_heads, block_size, head_size]`**, `block_size = 32`. Mandatory because:
1. genai reads back the cache Parameter's `partial_shape` and only overwrites `pshape[0]=num_kv_blocks`; it computes `block_size_in_bytes = pshape[1]*pshape[2]*pshape[3]*prec.size()` (`cache_manager.hpp:24-27,58-67`). The static dims [1],[2],[3] **must be** `{num_kv_heads, block_size, head_size}` in that order or genai's byte math and its `copy_blocks`/eviction corrupt.
2. **block_size = 32** is forced by genai's device detection: `all_gpu_device` requires the device name substring `"GPU"`; the plugin registers `"NVIDIA"` (`cuda_plugin.cpp:24`), so genai's branch is **CPU path → block_size = 32** (`cache_manager.hpp:40-43,76,89`). Hard-code 32 in the specialization pass.

Our CUDA kernels index the genai layout directly (no Intel K-transpose); coalescing is recovered by thread mapping in §3.

Logical→physical addressing, for subseq `i`, logical KV position `p`:
```
logical_block = p / 32
block_number  = block_indices[ block_indices_begins[i] + logical_block ]   // <0 ⇒ skip/pad
offset        = p % 32
addr(K) = key_cache  [block_number, hk, offset, :S]
addr(V) = value_cache[block_number, hk, offset, :SV]
```

Per-subsequence geometry:
```
q_len_i  = subsequence_begins[i+1] - subsequence_begins[i]
past_i   = past_lens[i]
kv_len_i = past_i + q_len_i
B_token  = Σ q_len_i = query.shape[0]
```

---

## 3. CUDA Kernel Design

Port strategy: **structure follows Intel GPU — two kernels: (A) reshape-and-cache, (B) paged attention with flash online-softmax**; numerics follow Intel CPU (the bit-reference). Correctness-first v1, then optimize.

New files: `src/kernels/paged_attention.{cu,hpp}` (device kernels + host launcher class mirroring existing kernel wrappers).

### 3.1 Kernel A — reshape_and_cache (KV write)
Scatters ports 1/2 (new K/V) into the paged cache using a `slot_mapping`:
```
for subseq i, for j in 0..q_len_i-1:
  block_offset = past_i + j
  block_number = block_indices[ block_indices_begins[i] + block_offset/32 ]
  slot         = block_number*32 + (block_offset % 32)
  slot_mapping[subsequence_begins[i]+j] = slot
```
Kernel grid `(B_token, Hk)`, threads over head_size. Each thread copies one element of the K row → `key_cache[slot/32, hk, slot%32, :]` and the V row → value_cache. f16/f32 straight copy; quantization deferred. Cheap relative to attention (only the new tokens).

### 3.2 Kernel B — paged attention (flash online-softmax)

**v1 (MVP) single-pass per (token, head), correctness-first.** Grid `(B_token, H)`, one warp (32 lanes) per query row × head. Serves both prefill and decode because the causal horizon is computed per row.

Per-row algorithm (online softmax):
```
row = global query token index (0..B_token-1)
i   = subsequence of row  (from a row→subseq map, §3.4)
h   = query head; hk = h / h_each_group
row_in_subseq = row - subsequence_begins[i]
cur_kv_len    = past_lens[i] + row_in_subseq + 1     // causal horizon

q[:S] = query[row, h*S : h*S+S] * scale              // scale Q once (fp16 range)

m = -inf; l = 0; acc[:SV] = 0
for p in 0 .. cur_kv_len-1:
    bn = block_indices[block_indices_begins[i] + p/32]; if (bn<0) continue
    if (sliding_window>0 && p < cur_kv_len - sliding_window) continue
    k  = key_cache[bn, hk, p%32, :S]
    s  = dot(q, k)
    if (alibi) s += alibi_slopes[h] * (-(cur_kv_len-1-p))
    m_new = max(m, s)
    corr  = exp(m - m_new)
    l     = l*corr + exp(s - m_new)
    acc[:SV] = acc[:SV]*corr + exp(s - m_new) * value_cache[bn, hk, p%32, :SV]
    m = m_new
out[row, h*SV : h*SV+SV] = acc[:SV] / l
```
- Dot product and V accumulation across the 32 warp lanes; `m/l` reductions via `__shfl_xor_sync`.
- Causal masking is **implicit via `cur_kv_len`** (no additive −inf).
- Default scale `1/sqrt(S)` if port 9 value is 0.
- ALiBi and sliding_window mutually exclusive; MVP honors both flags but not simultaneously.

**Prefill vs decode:** the *same* kernel serves both — `cur_kv_len` is `past+1` for decode rows and the causal triangle for prefill rows. Mixed prefill+decode batches work in one launch.

**v2 optimization notes (post-MVP):**
- **Two-pass flash-decoding (KV split)** for long contexts (Intel Stage 0 + Stage 1): partition KV into 256-token chunks, grid `(token, head, partition)`, partial `(acc,m,l)`, finalization kernel merges with `exp(m_p − m_global)` rescale (≈ vLLM `paged_attention_v2`).
- **Prefill blocked GEMM** (tensor-core `Q·Kᵀ`), or route long prefills through a dense SDPA micro-kernel.
- Cache Q in shared memory pre-scaled; vectorized `head_size`-wide loads; optional device-side K-transpose for coalesced QK.

### 3.3 GQA / masks summary
- GQA: `kv_head = q_head / (H/Hk)`. Constraint `H % Hk == 0`.
- `head_size % 16 == 0`, `v_head_size % 16 == 0` recommended for vectorization; assert and fall back to scalar loop otherwise.

### 3.4 Row→subsequence mapping & host-readable metadata
The i32 scheduling tensors (5–8, 12) are small and host-readable; genai fills them each step. Precompute a `row→subseq` map on the host from `subsequence_begins` (or a tiny device kernel) to avoid a per-row search. `max_context_len` (port 12) is host-only, sizes the two-pass scratch.

---

## 4. Plugin Integration

### 4.1 New files
- `src/kernels/paged_attention.hpp/.cu` — device kernels + host launcher class.
- `src/ops/paged_attention.hpp/.cpp` — `PagedAttentionOp : OperationBase`.
- `src/transformer/nvidia_convert_pagedattn_inputs.hpp/.cpp` — cache-shape specialization pass (§5).
- Tests under `tests/unit/` and functional tests (§7).

### 4.2 Op class & registration
```cpp
class PagedAttentionOp : public OperationBase {
public:
    using NodeOp = ov::op::PagedAttentionExtension;
    PagedAttentionOp(const CreationContext&, const NodeOp&, IndexCollection&& in, IndexCollection&& out);
    void Execute(const InferenceRequestContext&, Inputs, Outputs, const Workbuffers&) const override;
    CudaGraphCompatibility GetCudaGraphCompatibilityImpl() const override
        { return CudaGraphCompatibility::NONE; }   // dynamic shapes each step
    WorkbufferRequest GetWorkBufferRequest() const override;   // scratch for 2-pass + slot_mapping
};
OPERATION_REGISTER(PagedAttentionOp, PagedAttentionExtension);
```
- Registration string must equal `node.get_type_info().name == "PagedAttentionExtension"`.
- Constructor reads geometry from `node.get_rt_info()` keys `num_k_heads/k_head_size/num_v_heads/v_head_size`; reads scalars (scale/sliding_window; `has_alibi = alibi input size>0`).
- **CUDA graph compatibility = NONE** (PA reads dynamically-shaped inputs each step).

### 4.3 `is_operation_supported`
Whitelist `PagedAttentionExtension` in `Plugin::is_operation_supported` (`cuda_plugin.cpp:171`) so `query_model`/`compile_model` accept it.

### 4.4 How the 25 inputs arrive (I/O plumbing)
All 25 inputs arrive as **ordinary graph tensors** — no special I/O plumbing. Inputs 3–8,12,13 are model **Parameters** genai sets via `InferRequest::set_tensor`; 9,10,11 are folded Constants; 0,1,2 come from the model's Q/K/V projections. The existing `ParameterOp` delivers device pointers to `Execute`. Do **NOT** model the cache as OpenVINO state/`ReadValue`+`Assign` — genai owns cache lifecycle and feeds it as plain Parameters.

### 4.5 KV cache in the memory model (the core problem)
The cache is huge, mutable, persistent across steps, and **owned by genai as a host `ov::Tensor`** that genai resizes/copies/evicts itself. Naively, `key_cache.N` being a Parameter means generic `ParameterOp::Execute` would `upload()` multi-GB host→device every step and discard PA's device writes (`parameter.cpp:32`). Must special-case.

**Implemented (M2) — the host tensor is the source of truth, the device keeps a persistent mirror** (`src/cuda_paged_attention_context.hpp`):

On NVIDIA, genai allocates the caches as plain host tensors (its remote-tensor path is gated on device names containing "GPU"). Outside the model genai modifies them in exactly two places (genai `continuous_batching/scheduler.hpp`, `Scheduler::schedule()`, both before `infer()`):
- **growth** (`CacheManager::allocate_cache_if_needed`): a bigger tensor is allocated, the old contents are copied into its leading blocks, and it is bound with `set_tensor`. With the default `SchedulerConfig` (`cache_size == 0 && num_kv_blocks == 0`) this happens repeatedly, by 8 blocks at a time;
- **copy-on-write** (`CacheManager::copy_blocks`, beam search / parallel sampling / prefix caching): a shared, partially filled last block is copied into a new block, which becomes the sequence's last block and is appended to in the same step.

Coherence invariant: *after every infer, every slot written by PagedAttention is identical on host and device.* It is maintained by:
1. **No generic upload.** A cache Parameter consumed only by PagedAttention inputs 3/4 registers a 1-byte placeholder with its shape instead of uploading the whole cache every infer.
2. **`bind()`** each infer: the device mirror is keyed by the cache-input BufferID; on growth it is reallocated (geometrically) and the old contents are copied device-to-device (the host copy genai made equals them by the invariant). The replaced buffer is released only after the stream is synchronized. A new or desynchronized mirror is seeded with a full upload.
3. **Refresh before write.** Each block a step appends to at a non-zero in-block offset is re-uploaded from the host before `reshape_and_cache`: it may be a copy-on-write destination. The rule is stateless (it does not try to know whether genai copied), which is the only safe choice because block ids are recycled without the plugin seeing it. A block written from offset 0 needs no refresh.
4. **Write-through.** Every block written by a step is downloaded back to the host tensor. The downloads are deferred to `CudaInferRequest::wait_pipeline` (after all kernels are enqueued, before the stream is synchronized), so they neither stall kernel launches nor let `infer()` return before the host is up to date.
5. **Failure handling.** If an infer throws or is cancelled after some layers updated their mirrors, the context is marked desynchronized and every mirror is re-seeded from the host on the next bind.

Per step this transfers only the touched blocks (decode: one block per sequence per layer in each direction) instead of the whole cache. It relies on genai's write pattern above (verified at genai 933d76c); a future genai change that writes other blocks outside the model would need a matching refresh rule.

**Phase 4 (clean, zero-copy):** implement `create_context`/`get_default_context` + NVIDIA `RemoteTensor`/`RemoteContext`, make genai's `all_gpu_device` path fire so the cache is allocated directly in device memory — eliminates host↔device copies. Deferred.

Scratch (two-pass reduction, `slot_mapping`, `row→subseq`) from `GetWorkBufferRequest()` or dynamic `cudaMallocAsync` buffers.

### 4.6 Per-request lifetime
- One `PagedAttentionOp` instance per PA node (per layer), shared across infer requests (stateless op object).
- The persistent device cache buffer is bound per infer-request to the request's `key_cache.N` host tensor; on `set_tensor` rebind, re-establish the mapping. Scratch workbuffers are per-request from the pool.

---

## 5. Transformer Wiring

Two changes in `src/transformer/cuda_graph_transformer.cpp`:

### 5.1 Cache-shape specialization pass (mandatory)
Add `NvidiaConvertPagedAttnInputs`, mirroring OpenVINO `convert_pagedattn_inputs.cpp:84-152`:
- For each PA op, read rt_info `num_k_heads/k_head_size/num_v_heads/v_head_size`.
- For inputs 3 & 4 (`key_cache.N`/`value_cache.N` Parameters), `set_partial_shape({-1, num_kv_heads, 32, head_size})` and `set_element_type(cache_precision)`.
  - **block_size = 32** hard-coded. cache_precision = model inference precision (f16 default).
- Register **before** `run_passes` (`cuda_graph_transformer.cpp:173`).

### 5.2 Guard SDPA decomposition
`cuda_graph_transformer.cpp:153` unconditionally registers `ScaledDotProductAttentionDecomposition`. Guard it:
- Scan `model->get_ordered_ops()` for `ov::op::PagedAttentionExtension`; if present, `has_pa=true`.
- `if (!has_pa) pass_manager.register_pass<ScaledDotProductAttentionDecomposition>();`

### 5.3 Advertising support
`is_operation_supported` whitelist (§4.3) is what `query_model` uses to report the op is supported.

---

## 6. Phasing / Milestones

**M0 — Plumbing (compiles, no math).** Register op, whitelist, cache-shape pass, SDPA guard. Execute is a stub validating 25 inputs and asserting 13–24 empty. Success: a GenAI-transformed GPT-2 model *compiles* on NVIDIA.

**M1 — MVP single-sequence correctness.** ✅ Kernel A (f16/f32 write) + Kernel B v1. Persistent device cache (device-only: no host coherence yet). scale + causal + GQA. Success: one sequence, prefill+decode, matches CPU-REF; GPT-2 greedy matches CPU plugin.

**M2 — Batched / mixed / continuous batching.** ✅ Row→subseq map; mixed prefill+decode; multi-subsequence block-table; host/device cache coherence for cache growth and copy-on-write (§4.5); validated scheduling metadata; loud errors for unsupported features (KV cache rotation/eviction scores, sinks, adaptive R-KV, quantized caches). Success: genai `ContinuousBatchingPipeline` and default `LLMPipeline` (prefix caching + dynamic cache growth, chat) on GPT-2 match the CPU plugin token-for-token, including beam search, parallel sampling, chunked prefill, vLLM-mode scheduling and late-arriving requests.

**M3 — Feature completeness.** ALiBi, sliding_window. Success: an ALiBi model and a sliding-window model pass.

**M4 — Performance.** Two-pass flash-decoding, prefill tensor-core GEMM, shared-memory Q, coalesced layout. Success: throughput near a reference vLLM paged kernel; no correctness regression.

**M5 — Extensions & zero-copy (optional).** Quantized KV cache; score output; RemoteTensor device-resident cache.

---

## 7. Testing

**7.1 Unit — kernel correctness (host reference).** gtest under `tests/unit/`: synthetic `query/key/value`, a small paged cache `[num_blocks, Hk, 32, S]`, and metadata; run Kernel A+B, compare against a CPU golden. Cover: decode (`q_len==1`), prefill (causal triangle), mixed batch, GQA (`H!=Hk`), `S!=SV`, block boundary (`kv_len>32`), pad block id skip, default scale, ALiBi, sliding_window. Tolerance f32 ~1e-4, f16 ~2e-2. Plus slot-mapping builder test.

**7.2 Functional — op via plugin.** Tiny `ov::Model` with a single `PagedAttentionExtension` (25 inputs, rt_info geometry), compile on NVIDIA, compare output 0 to the CPU plugin. Verifies registration, cache-shape pass, cache handling, output shape.

**7.3 E2E — GPT-2 via genai continuous batching.** GPT-2 through `ContinuousBatchingPipeline` on `"NVIDIA"`, greedy, diff token ids vs CPU. Then multi-prompt concurrent and a long (>32-token, multi-block) prompt.

**7.4 Regression.** Add to CI, guarded by CUDA availability.

---

## 8. Risks & Open Questions

1. **KV-cache coherence / performance (highest risk).** genai owns the cache as a host tensor and does block copies/eviction on the host. MVP host↔device sync is correct but potentially slow/subtle. The clean fix is RemoteTensor (§4.5 Phase 4) — decide whether to bring it forward.
2. **block_size = 32 assumption.** Derived from genai's non-"GPU" device-name branch. If genai's detection changes, the pass value breaks. Assert consistency; CPU-REF also asserts `block_size==32`.
3. **13-input vs 25-input PA.** Target is the 25-input form; some genai builds may emit 13. Validate `get_input_size()`; confirm what the installed genai emits.
4. **rt_info geometry availability.** Depend on `num_k_heads/...` in rt_info. Fallback: derive `Hk` from the statically-specialized cache shape and `H` from `query[1]/S`; assert if neither present.
5. **Output dtype override.** Honor `m_output_type[0]` if static else query dtype; reject u8 output until quantization phase.
6. **Scale as runtime tensor vs constant.** Usually folded Constant, but some models produce runtime scalars. Execute must read ports 9/10 from device (or host-copy) rather than assuming compile-time constants.
7. **CUDA-graph incompatibility cost.** `NONE` disables graph capture for the PA region; acceptable for correctness, a latency cost. Revisit once shapes stabilize per step.
8. **Prefill efficiency.** v1 single-pass per-row prefill is O(q_len·kv_len) — correct but slow for long prompts. M4 blocked/tensor-core prefill needed for usable latency.
9. **Numerical parity target.** Pick **CPU-REF** as golden (deterministic); GPU-REF is the performance structure guide.

---

## Files to create/modify

Create: `src/ops/paged_attention.{hpp,cpp}`, `src/kernels/paged_attention.{hpp,cu}`, `src/transformer/nvidia_convert_pagedattn_inputs.{hpp,cpp}`.
Modify: `src/transformer/cuda_graph_transformer.cpp` (SDPA guard ~line 153, register pass ~line 173), `src/cuda_plugin.cpp` (`is_operation_supported` ~line 171).
