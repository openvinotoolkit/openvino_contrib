---
name: gfx-plugin-dev
description: Use when working on OpenVINO GFX plugin code in modules/gfx_plugin, especially MLIR lowering, runtime stages, backend routing, plugin properties, or architecture-sensitive refactors across Metal and Vulkan backends.
---

# GFX Plugin Development

This skill is for implementing or refactoring the `GFX` OpenVINO plugin in `modules/gfx_plugin/`.

## Use This Skill When

- The task touches `src/`, `include/`, `tests/`, `CMakeLists.txt`, or backend-specific code.
- The user asks for architecture changes, new ops, runtime fixes, property behavior, backend routing, or MLIR/codegen work.
- The task mentions Metal, Vulkan, `GFX`, `CompiledModel`, `InferRequest`, remote tensors, or MLIR lowering.

## Primary References

Read these first:

1. `README.md`
2. `docs/ARCHITECTURE.md`
3. `docs/DEVELOPMENT.md`
4. `docs/TESTING.md`

Then read the relevant code path:

- Plugin contract: `src/plugin/`
- Backend-neutral runtime: `src/runtime/`
- MLIR builders and stage planning: `src/mlir/`
- Parallel and graph rewrites: `src/transforms/`
- Metal backend: `src/backends/metal/`
- Vulkan backend: `src/backends/vulkan/`
- Public API surface: `include/openvino/gfx_plugin/`

## Development Rules

- Treat local module docs as the published source of truth.
- Reuse existing OpenVINO plugin patterns; do not invent unrelated abstractions.
- Keep shared logic backend-neutral unless the behavior is truly backend-specific.
- Prefer extending `gfx_parallelism.*`, `gfx_partitioning.*`, `gfx_stage_policy.*`, or shared caches before adding new special-purpose plumbing.
- For MLIR changes, keep compile-time support probing, lowering, and runtime behavior aligned.
- When changing plugin-visible behavior, also check properties, `query_model()`, and compiled-model/runtime property exposure.
- When changing stateful graph behavior, treat `ReadValue` / `Assign` as a dedicated infer-request-state path, not just generic stateless runtime stages.

## Common Workflows

### New or changed op path

1. Update support probing in `src/mlir/`.
2. Update lowering or specialized path in `src/mlir/` or `src/transforms/`.
3. Update backend runtime handling if needed.
4. Add or update focused unit tests.
5. Update docs if supported shapes, route selection, or backend behavior changed.

Check whether the change belongs to one of the current special families:

- dynamic-shape data movement and shape ops such as `ShapeOf`, `Concat`, `Broadcast`, `Select`, `StridedSlice`, and `Range`
- dedicated lowered ops such as `RMS` and `ScatterUpdate`
- stateful `ReadValue` / `Assign` handling through infer-request variable storage
- backend-specialized launch paths that now depend on final runtime shape or final shader binding counts

### Runtime or backend scheduling change

1. Inspect `src/runtime/gfx_stage_policy.*`, `gfx_parallelism.*`, and `gfx_partitioning.*`.
2. Check whether the change is backend-neutral or family-specific.
3. Verify interaction with infer submission, immutable const caches, and prepared binding reuse when applicable.
4. Add tests in `tests/unit/` and backend tests when behavior is externally visible.

### Stateful or reusable infer-path change

1. Inspect `src/plugin/infer_request_state.hpp`, `src/plugin/infer_pipeline.*`, `src/plugin/infer_io_utils.*`, and `src/plugin/stateful_execution.*`.
2. Keep variable-buffer lifetime, reusable host-output lifetime, and stage-output shape/type recovery aligned.
3. Treat `ReadValue` as a view-style stage and `Assign` as a persisted copy/update path unless the code explicitly changes that contract.

### Property or device-selection change

1. Update property parsing and lists in `src/plugin/`.
2. Check `ov::available_devices`, `ov::device::id`, `ov::device::full_name`, and cache-related properties.
3. Add or update `tests/unit/plugin_tests.cpp`.
4. Update `README.md` and `docs/USAGE.md` if the public contract changed.

## What To Avoid

- Do not document removed architectures as current behavior.
- Do not leave backend-specific fast-path changes undocumented when they affect supported shapes, dispatch rules, or profiling semantics.
- Do not add public-facing docs outside `modules/gfx_plugin/`.
- Do not modify `third_party/llvm-project/` unless the task explicitly requires vendored LLVM changes.

## Output Expectations

- Keep changes focused by subsystem.
- If the task changes architecture, runtime semantics, or public properties, update local docs in the same pass.
- If the task reaches commit/push stage, also apply the same plugin change set to the mirrored `ov-ext-labs/gfx-plugin` repository unless the user explicitly says otherwise.
