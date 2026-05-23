---
name: gfx-plugin-docs
description: Use when updating published documentation for modules/gfx_plugin after code changes, especially README, ARCHITECTURE, DEVELOPMENT, TESTING, or USAGE sync work for new runtime behavior, properties, backend routes, or test coverage.
---

# GFX Plugin Docs

This skill is for keeping `modules/gfx_plugin` documentation aligned with the current implementation.

## Use This Skill When

- The code changed and the user asks to update docs.
- Public properties, device semantics, backend routes, supported shapes, profiling, or test layout changed.
- `README.md` or any file under `docs/` may now be stale.

## Documentation Contract

- Published docs for this module live inside `modules/gfx_plugin/`.
- Keep module docs in English.
- Prefer updating existing source-of-truth files instead of adding ad-hoc notes.

Primary published docs:

- `README.md`
- `docs/ARCHITECTURE.md`
- `docs/DEVELOPMENT.md`
- `docs/TESTING.md`
- `docs/USAGE.md`

Additional operational docs:

- `docs/MICROBENCH_SCHEMA.md`
- `docs/PROFILING_RUNBOOK.md`

## Update Strategy

1. Inspect the code diff first.
2. Identify whether the change affects:
   - architecture or runtime flow
   - property semantics
   - backend-specific route selection
   - OpenCL source-artifact coverage, dynamic OpenCL runtime behavior, or non-Apple default-backend resolution
   - Metal placement-domain or storage selection such as `apple_mps` image/matrix stages versus `apple_msl` buffer dispatch
   - manifest-backed execution-kind or hybrid-stage planning such as vendor `MPSGemm` plus custom MSL epilogues
   - Apple MPS/MPSGraph vendor primitive descriptor support such as Conv2D, Pool2D, Resize2D, Softmax, TopK, GEMM, or SDPA
   - supported-shape or dispatch constraints
   - stateful execution semantics for `ReadValue` / `Assign`
   - dedicated lowering families such as `RMS`, `ScatterUpdate`, or `RoPE`
   - custom fused ops or backend-only graph rewrites such as `GfxSDPAWithCausalMask`
   - effective runtime type or const-packing behavior that changes kernel compilation assumptions
   - backend-aware transform preservation or decomposition rules
   - backend-aware transform fusion or regrouping rules such as rotate-half to `RoPE` or fused compressed `MatMul`
   - backend-side fused epilogues such as residual-add absorbed into `RMS`
   - liveness-managed workspace allocation for stage outputs
   - view-style aliasing rules for split outputs
   - output-alias routing or source-node-aware output resolution for fused stages
   - command-encoder or binding-reuse behavior that changes profiling counters or dispatch setup cost
   - MPSRT runtime-model, kernel-family-manifest, or external-buffer-ABI behavior on Metal
   - custom-kernel family classification, semantic input/output roles, or dispatch-grid policy
   - Metal MSL runtime binding plans, explicit kernel-buffer order, or inferred MSL buffer-argument counts
   - MLIR-owned Metal MSL source generation such as Apple MSL binding/dispatch/op-family helpers, Apple MPS/MPSGraph vendor source plans, MatMul direct/MPSRT helpers, compressed `MatMul`, SDPA, or causal SDPA helpers
   - SPIR-V fixed-argument binding adapters or compact Vulkan ABI metadata
   - Vulkan Conv2D output-channel blocking through `gfx.dispatch_channel_block`, SPIR-V cache metadata, or shared convolution lowering
   - `GfxTargetProfile`, `GpuExecutionDeviceInfo`, `extended.target_profile`, or `target_backend_*` profiling counters
   - infer submission dependency-window extension, soft-budget caps, or boundary-stage rules
   - typed MPSRT builder-plan/runtime-model or storage-bridge behavior on Metal
   - backend custom-kernel ABI adapters, runtime-value helpers, or diagnostic Metal placement properties
   - backend-neutral MPSRT runtime-model reconstruction, runtime resource tables, external-buffer bindings, prepared resource heaps, or model/transient resource lifetimes
   - typed `GfxMpsrtProgram` / generated `gfx_mpsrt_ops` behavior or cleanup of stale legacy MPSRT attrs
   - Apple stage-pipeline or typed `gfx.mpsrt` dialect behavior
   - stage-level profiling estimates such as `bytes_in`, `bytes_out`, `macs_est`, or `flops_est`
   - test layout or regression coverage
   - compare-runner CLI, real-image/golden-reference workflow, or debug workflow
   - public `ov::hint::inference_precision` behavior or precision-aware test tolerance policy
   - profiling or microbench workflows
3. Patch only the files whose contract changed.
4. Keep wording concrete and tied to actual code behavior.

## Mapping From Code Change To Doc File

- `src/plugin/` or property changes:
  - `README.md`
  - `docs/USAGE.md`
  - `docs/DEVELOPMENT.md`

- `src/runtime/`, `src/backends/*/runtime/`, scheduling, or cache changes:
  - `README.md`
  - `docs/ARCHITECTURE.md`
  - `docs/DEVELOPMENT.md`
  - `docs/PROFILING_RUNBOOK.md` if counters or submit-window diagnostics changed

- `src/backends/opencl/` or `src/kernel_ir/gfx_opencl_source_artifacts.*` changes:
  - `README.md`
  - `docs/ARCHITECTURE.md`
  - `docs/DEVELOPMENT.md`
  - `docs/TESTING.md`
  - `docs/USAGE.md` if public backend selection or runtime behavior changed

- `src/mlir/` or `src/transforms/` changes:
  - `README.md` when user-visible behavior shifts
  - `docs/ARCHITECTURE.md`
  - `docs/DEVELOPMENT.md`
  - `docs/TESTING.md` if regression coverage changes

- `src/runtime/gfx_mpsrt_model.*`, `src/backends/metal/runtime/mpsrt/`, or split Apple MSL/MPS source-plan changes:
  - `README.md`
  - `docs/ARCHITECTURE.md`
  - `docs/DEVELOPMENT.md`
  - `docs/TESTING.md` if resource-model, binding, or runtime coverage changed

- test-only changes:
  - `docs/TESTING.md`
  - optionally `README.md` if the public test story changed

- compare/profiling/microbench tool changes:
  - `docs/TESTING.md`
  - `docs/USAGE.md`
  - `docs/MICROBENCH_SCHEMA.md`
  - `docs/PROFILING_RUNBOOK.md`

- target-profile JSON or microbench device-fingerprint changes:
  - `docs/USAGE.md`
  - `docs/MICROBENCH_SCHEMA.md`
  - `docs/PROFILING_RUNBOOK.md`

## Writing Rules

- Do not describe historical or removed architectures as current implementation.
- Do not promise backend parity if the code is backend-specific.
- Use exact property names such as `ov::available_devices`, `ov::device::id`, and `GFX_BACKEND`.
- Mention concrete test files when documenting new regression coverage.
- Keep README public-facing; keep deeper mechanics in `docs/ARCHITECTURE.md` and `docs/DEVELOPMENT.md`.
- Keep the published tree clean: do not normalize local dumps, caches, backups, or machine-local helper outputs into public docs.

## Output Expectations

- State which docs changed and why.
- If no doc change is needed, say that explicitly after checking the diff.
