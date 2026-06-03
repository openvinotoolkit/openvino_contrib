---
name: gfx-plugin-docs
description: Use when updating published documentation for modules/gfx_plugin after code changes, especially README, ARCHITECTURE, DEVELOPMENT, TESTING, USAGE, profiling, or microbench docs.
---

# GFX Plugin Docs

This skill keeps `modules/gfx_plugin` documentation aligned with the current
implementation.

## Use This Skill When

- The code changed and docs may be stale.
- Public properties, device semantics, backend routes, supported shapes,
  profiling, microbench output, or test layout changed.
- The task asks to clean public docs, repo-local skills, or developer
  instructions.

## Published Docs

Primary files:

- `README.md`
- `docs/ARCHITECTURE.md`
- `docs/DEVELOPMENT.md`
- `docs/TESTING.md`
- `docs/USAGE.md`

Operational files:

- `docs/MICROBENCH_SCHEMA.md`
- `docs/PROFILING_RUNBOOK.md`

Repo-local skills can also be public orientation material:

- `.codex/skills/gfx-plugin-dev/SKILL.md`
- `.codex/skills/gfx-plugin-docs/SKILL.md`
- `.codex/skills/gfx-plugin-testing/SKILL.md`
- `.codex/skills/gfx-plugin-profiling/SKILL.md`
- `.codex/skills/gfx-plugin-release-sync/SKILL.md`

Do not edit `AGENTS.md` unless the user explicitly asks.

## Update Strategy

1. Inspect the live source tree and current diff first.
2. Identify whether the change affects:
   - backend availability or default backend resolution
   - configured backend availability generated from
     `src/compiler/backend_config.hpp.in`
   - compiler service, backend registry, lowering-plan, manifest, executable
     bundle, pipeline-stage builder, pipeline-stage fusion selection,
     pipeline-stage I/O plan, memory plan, cache envelope, stage-placement
     policy, stage compiler policy, fused-output lifetime plan, runtime
     pipeline-stage materializer, runtime session, or runtime descriptor
     behavior
   - compiler-owned tensor-layout classification
   - public properties
   - `query_model()` or compile behavior
   - MLIR support/lowering/backend hooks and backend-owned source planning
   - Metal placement, MPSRT records, MPS/MPSGraph descriptors,
     vendor attention artifact materialization,
     `VendorDescriptor` payloads, generated activation/elementwise/reduction/
     Softmax/LogSoftmax MSL routes, `Swish` static/runtime beta contracts,
     Pool2D vendor-route-only behavior, or MSL binding
   - OpenCL source-artifact coverage, runtime-shape handling, static f32
     scalars, constants, generated activation/elementwise/MatMul units,
     generated f32 and boolean reduction units, generated f32/f16 Softmax
     units, dynamic-static-rank Softmax units, generated f32/f16 Pool2D units,
     generated ShapeOf/Tile/Transpose units, generated compare/select and
     logical-bool elementwise units, generated Concat/Split helpers, `Swish`
     default/static/runtime beta artifacts, chunking, or boolean-buffer
     behavior
   - CLVK/CLSPV Raspberry OpenCL bundle wiring, OpenCL dynamic-loader search
     order, or third-party submodule publication
   - OpenCL runtime-bundle candidate ordering or bundled tool-path setup
   - backend-owned OpenCL payload materialization in
     `src/backends/opencl/compiler/opencl_kernel_artifacts.*`
   - removal or reintroduction risk around `BackendLowering`,
     `metal_lowering`, `mps_graph_attention_stage`, or source-signature ABI
     fallback behavior
   - backend stage placement, stage policy, parallelism, partitioning,
     submission, caches, or workspace allocation
   - stateful `ReadValue` / `Assign`
   - output aliasing, compiler-owned pipeline-stage I/O planning, fused-output
     lifetimes, pipeline-stage materialization, or source-node-aware output
     resolution
   - descriptor-backed view-only stages
   - compare-runner, profiling, trace sinks, microbench, or target-profile
     output
   - test layout, controlled test `plugins.xml`, gtest matrix checks,
     native/unavailable-adapter source-contract checks, disabled-pattern hooks,
     or validation workflow
3. Patch only docs/skills whose contract changed.
4. Keep wording concrete and tied to actual files that exist.

For documentation/security publication tasks, do not run build or test targets
unless the user explicitly asks for that validation. Use source inspection,
security/stale-reference grep, `git diff --check`, and staged diff review for
the publication gate.

## Mapping

- `src/plugin/` or property changes:
  - `README.md`
  - `docs/USAGE.md`
  - `docs/DEVELOPMENT.md`

- `src/compiler/`, compiler backend policies, manifests, executable bundles, or
  runtime descriptors:
  - `README.md`
  - `docs/ARCHITECTURE.md`
  - `docs/DEVELOPMENT.md`
  - `docs/TESTING.md`

- `src/runtime/`, scheduling, caches, memory, or profiling:
  - `README.md`
  - `docs/ARCHITECTURE.md`
  - `docs/DEVELOPMENT.md`
  - `docs/TESTING.md`
  - `docs/PROFILING_RUNBOOK.md` when counters or profiling flow changed

- `src/mlir/`, `src/kernel_ir/`, or `src/transforms/`:
  - `README.md` for user-visible support changes
  - `docs/ARCHITECTURE.md`
  - `docs/DEVELOPMENT.md`
  - `docs/TESTING.md`

- Metal backend or MPSRT changes:
  - `README.md`
  - `docs/ARCHITECTURE.md`
  - `docs/DEVELOPMENT.md`
  - `docs/TESTING.md`
  - `docs/USAGE.md` for public properties or diagnostics

- OpenCL backend or source-artifact changes:
  - `README.md`
  - `docs/ARCHITECTURE.md`
  - `docs/DEVELOPMENT.md`
  - `docs/TESTING.md`
  - `docs/USAGE.md`

- compare/profiling/microbench tools:
  - `docs/TESTING.md`
  - `docs/USAGE.md`
  - `docs/MICROBENCH_SCHEMA.md`
  - `docs/PROFILING_RUNBOOK.md`

## Writing Rules

- Do not describe removed architectures or removed backends as current.
- Do not promise backend parity unless tests and code support it.
- Use exact property names such as `GFX_BACKEND`, `ov::available_devices`, and
  `ov::device::id`.
- Mention concrete source or test files when that helps orientation.
- Keep README public-facing; keep deep mechanics in `docs/ARCHITECTURE.md` and
  `docs/DEVELOPMENT.md`.
- Keep docs in English.
- Keep local dumps, build artifacts, machine paths, sensitive access material, and agent notes
  out of public docs and commits.

## Output Expectations

- State which docs changed and why.
- State what stale or private content was removed.
- If no doc change is needed after inspection, say that explicitly.
