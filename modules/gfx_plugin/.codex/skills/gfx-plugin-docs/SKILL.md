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
   - supported-shape or dispatch constraints
   - stateful execution semantics for `ReadValue` / `Assign`
   - dedicated lowering families such as `RMS` or `ScatterUpdate`
   - effective runtime type or const-packing behavior that changes kernel compilation assumptions
   - test layout or regression coverage
   - compare-runner CLI or debug workflow
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

- `src/mlir/` or `src/transforms/` changes:
  - `README.md` when user-visible behavior shifts
  - `docs/ARCHITECTURE.md`
  - `docs/DEVELOPMENT.md`
  - `docs/TESTING.md` if regression coverage changes

- test-only changes:
  - `docs/TESTING.md`
  - optionally `README.md` if the public test story changed

- compare/profiling/microbench tool changes:
  - `docs/TESTING.md`
  - `docs/USAGE.md`
  - `docs/MICROBENCH_SCHEMA.md`
  - `docs/PROFILING_RUNBOOK.md`

## Writing Rules

- Do not describe historical or removed architectures as current implementation.
- Do not promise backend parity if the code is backend-specific.
- Use exact property names such as `ov::available_devices`, `ov::device::id`, and `GFX_BACKEND`.
- Mention concrete test files when documenting new regression coverage.
- Keep README public-facing; keep deeper mechanics in `docs/ARCHITECTURE.md` and `docs/DEVELOPMENT.md`.

## Output Expectations

- State which docs changed and why.
- If no doc change is needed, say that explicitly after checking the diff.
