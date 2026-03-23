# Testing Guide

This document summarizes how testing is organized for `modules/gfx_plugin`.

## Main Test Target
The primary test executable is `ov_gfx_func_tests`, created in `tests/CMakeLists.txt`.

Build it with:

```bash
cmake --build build-gfx-plugin --target ov_gfx_func_tests
```

Run the CTest label:

```bash
ctest --test-dir build-gfx-plugin --output-on-failure -L GFX
```

Run the gtest binary directly:

```bash
find build-gfx-plugin -name ov_gfx_func_tests -type f
DYLD_LIBRARY_PATH=/path/to/openvino/runtime/libs \
  <path-to-ov_gfx_func_tests> --gtest_filter=MetalBasicOps.*
```

## Test Layout
- `tests/unit/`: focused unit tests for plugin logic, MLIR lowering, and helpers
- `tests/integration/`: plugin-level integration checks
- `tests/backends/metal/`: Metal-specific runtime and behavior coverage
- `tests/backends/vulkan/`: Vulkan-specific runtime and behavior coverage
- `tests/shared_tests_instances/`: OpenVINO shared test wiring
- `tests/tools/`: helper tools such as `ov_gfx_compare_runner`

## Typical Test Suites
Examples already present in the tree:
- `GfxBasicOps`
- `MetalBasicOps`
- `MetalPrecisionStudy`
- plugin property and backend-selection tests in `tests/unit/plugin_tests.cpp`

## When To Add Tests
Add or update tests when you change:
- supported ops or their constraints
- property parsing or supported-property lists
- backend selection behavior
- remote context / remote tensor behavior
- stage fusion behavior
- MLIR support probing

## Practical Strategy
- run the narrowest relevant gtest filter first
- then run the broader backend suite
- then run `ctest -L GFX` before finalizing a change

If you change backend-specific code, prefer adding at least:
- one unit or focused regression test
- one end-to-end backend test when behavior is externally visible

## Helpful Notes
- Some tests skip when the corresponding backend is unavailable on the current machine
- Metal tests require a valid Metal runtime environment
- Vulkan tests depend on Vulkan being enabled and available in the build
