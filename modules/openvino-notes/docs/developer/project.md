# Project Overview

`openvino-notes` is an Android multi-module project. The product direction is an AI-assisted notes app powered by OpenVINO, but the repository is still at an early implementation stage.

## Modules

| Module | Responsibility | Current state |
| --- | --- | --- |
| `:app` | Android app module, Compose UI, and app wiring | Basic shell, starter UI, and debug and androidTest APK outputs |
| `:domain` | Models, repository contracts, and use cases | Mostly placeholder contracts and use cases |
| `:data` | Repository implementations, storage, and mapping | Structure exists, implementation is still minimal |
| `:ai` | OpenVINO-facing inference and result processing | Integration points exist, production behavior is not implemented yet |

## Build and Automation

- Root build logic lives in [build.gradle.kts](/Users/anesterov/repos/openvino-notes/build.gradle.kts).
- Shared configuration lives in [settings.gradle.kts](/Users/anesterov/repos/openvino-notes/settings.gradle.kts), [gradle.properties](/Users/anesterov/repos/openvino-notes/gradle.properties), [detekt.yml](/Users/anesterov/repos/openvino-notes/detekt.yml), and [lint.xml](/Users/anesterov/repos/openvino-notes/lint.xml).
- GitHub Actions workflows live in `.github/workflows/`.
- Most reusable command logic lives in `.github/scripts/`.

## Current State

Already in place:

- a four-module Android build
- reusable GitHub Actions workflows
- shared formatting, lint, and coverage policy

Still mostly scaffolded:

- domain contracts
- data-layer behavior
- OpenVINO integration
- app-level product flows

## Contributor Notes

- Start with [Local CI Reproduction](./ci-local.md) for day-to-day validation commands.
- Check `.github/scripts/` before editing workflow YAML.
- If you change module boundaries, update Gradle settings, module build files, and docs together.
