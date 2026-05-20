[![CI](https://github.com/embedded-dev-research/openvino-notes/actions/workflows/ci.yml/badge.svg?branch=main)](https://github.com/embedded-dev-research/openvino-notes/actions/workflows/ci.yml?query=branch%3Amain)

# OpenVINO Notes

OpenVINO notes is an AI-powered notes app for Android.

The project is written in **Kotlin** and uses **OpenVINO** to run AI features directly on Android devices.

## 🎯 Goal
Build a lightweight Android notes application with on-device AI for fast, private, and efficient text processing.

## 🚀 Build & Run

### Prerequisites
* Android Studio (Ladybug+)
* JDK 17
* Android SDK 37
* A connected Android device or a running emulator for install tasks

### Terminal Instructions

1. Go to the module directory:
cd modules/openvino-notes

2. Build the project:
./gradlew assembleDebug

3. Start an emulator or connect a device, then install the debug build:
./gradlew installDebug

For a validated terminal-based emulator flow on, see [`docs/developer/ci-local.md`](./docs/developer/ci-local.md).

## 🏗 Architecture & Tech Stack

The project follows **Clean Architecture** principles with a multi-module setup:
* :app — UI & Dependency Injection
* :domain — Business Logic & Repository interfaces
* :data — Repository implementations & Storage
* :ai — OpenVINO Java API & AI logic

**Core Technologies:**
* Kotlin
* Android
* OpenVINO

## 📚 Documentation

Project documentation now lives in [`docs/`](./docs/README.md):

* [`docs/README.md`](./docs/README.md) - documentation index
* [`docs/developer/README.md`](./docs/developer/README.md) - contributor entry point
* [`docs/developer/project.md`](./docs/developer/project.md) - project overview
* [`docs/developer/ci-local.md`](./docs/developer/ci-local.md) - how to reproduce CI checks locally on Linux, macOS, and Windows
