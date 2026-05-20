# Project Overview

`openvino-notes` is an Android multi-module project for an AI-assisted notes app powered by OpenVINO.

## Modules

| Module | Responsibility | Current state |
| --- | --- | --- |
| `:app` | Android app module, Compose UI, and app wiring | Notes UI, editor AI actions, warm-up state, and release/debug APK outputs |
| `:domain` | Models, repository contracts, and use cases | Notes/folders use cases plus AI suggestion/apply contracts |
| `:data` | Repository implementations, storage, and mapping | Local note and media persistence |
| `:ai` | OpenVINO-facing inference and result processing | On-device OpenVINO GenAI backend, prompts, normalization, retry/fallback logic, and model validation tests |

## Build and Automation

- Root build logic lives in [build.gradle.kts](../../build.gradle.kts).
- Shared configuration lives in [settings.gradle.kts](../../settings.gradle.kts), [gradle.properties](../../gradle.properties), [detekt.yml](../../detekt.yml), and [lint.xml](../../lint.xml).
- GitHub Actions workflows live in `.github/workflows/`.
- Most reusable command logic lives in `.github/scripts/`.

## Current State

Already in place:

- a four-module Android build
- reusable GitHub Actions workflows
- shared formatting, lint, and coverage policy
- OpenVINO Android prebuild and LLM model bundle consumption from rolling GitHub prereleases
- on-device summary, tag, and rewrite suggestions backed by OpenVINO GenAI

Intentionally out of scope for the current AI path:

- image understanding; image tagging remains separate from the text LLM backend
- storing model or OpenVINO runtime binaries in git
- publishing local signing keys or machine-specific configuration

## Contributor Notes

- Start with [Local CI Reproduction](./ci-local.md) for day-to-day validation commands.
- See [On-Device AI](./on-device-ai.md) before changing model packaging, prompts, or runtime bootstrap.
- Check `.github/scripts/` before editing workflow YAML.
- If you change module boundaries, update Gradle settings, module build files, and docs together.
