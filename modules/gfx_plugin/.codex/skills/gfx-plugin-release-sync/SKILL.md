---
name: gfx-plugin-release-sync
description: Use when a GFX plugin task reaches commit, push, branch, or PR stage and the same plugin change set must be provided both in openvino_contrib/modules/gfx_plugin and in the mirrored ov-ext-labs/gfx-plugin repository.
---

# GFX Plugin Release Sync

This skill is for commit/push/release workflow on the mirrored GFX plugin repositories.

## Use This Skill When

- The user asks to commit, push, or prepare a PR for `gfx_plugin`.
- A completed plugin change must be published in both repositories.
- The task mentions the mirror repository `https://github.com/ov-ext-labs/gfx-plugin`.

## Mirrored Publication Rule

Unless the user explicitly says otherwise, treat dual publication as required:

1. `openvino_contrib/modules/gfx_plugin`
2. `ov-ext-labs/gfx-plugin`

The intended plugin content should stay identical across both publication targets.

## Workflow

### 1. Confirm scope locally

- Check tracked modifications in `modules/gfx_plugin`.
- Exclude local-use artifacts such as:
  - `AGENTS.md`
  - `__pycache__/`
  - `.DS_Store`
  - backup files such as `*.backup`
  - local reports, JSON dumps, temporary files, and machine-local helper outputs

### 2. Build the staged set intentionally

- Stage only the plugin code and documentation that belong to the change.
- Verify the cached set before commit.

### 3. Commit clearly

- Use concise `GFX: ...` subjects.
- Keep commit scope coherent.
- Mention docs in the same commit when they were updated to match behavior.

### 4. Push to both publication targets

- Push the openvino_contrib branch normally.
- Then apply the same plugin change set to the mirrored `ov-ext-labs/gfx-plugin` repository and push there as well.

If the mirrored repository is not present locally, first discover whether:

- there is a local clone elsewhere on disk
- a second remote should be added
- branch mapping must be resolved

Do not silently skip the mirror step when the task clearly reached publish stage. Surface the missing local mirror or missing branch mapping.

## Commit Hygiene

- Never include `AGENTS.md`.
- Never include local profiling artifacts, compare outputs, caches, or platform staging directories.
- Keep commit messages understandable to external reviewers.

## PR Guidance

When asked for PR text:

- describe the plugin behavior change, not just filenames
- mention affected backend(s)
- mention test coverage or docs sync
- frame the module as `GFX` OpenVINO plugin code, not a generic helper patch

## Output Expectations

- Report the final commit hash or hashes.
- State what was intentionally excluded from the commit.
- If mirror publication could not be completed, say exactly what was missing.
