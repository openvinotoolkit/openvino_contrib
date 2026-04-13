import os
import subprocess
from setuptools import setup
from torch.utils.cpp_extension import BuildExtension

# Get the directory containing this setup.py file
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

# ── OpenVINO extensions (cmake-based) ─────────────────────────────────────────
OV_EXTENSIONS = [
    {
        'name': 'BEVPool',
        'dir': os.path.join(SCRIPT_DIR, 'openvino_extensions', 'bev_pool'),
        'output': 'libopenvino_bevpool_extension.so',
    },
    {
        'name': 'Voxelize',
        'dir': os.path.join(SCRIPT_DIR, 'openvino_extensions', 'voxelize'),
        'output': 'libopenvino_voxelize_extension.so',
    },
    {
        'name': 'SparseEncoder',
        'dir': os.path.join(SCRIPT_DIR, 'openvino_extensions', 'sparse_encoder'),
        'output': 'libopenvino_sparse_encoder_extension.so',
    },
]


def build_openvino_extension(ext_info: dict) -> bool:
    """Build a single OpenVINO custom extension via cmake."""
    ext_dir = ext_info['dir']
    ext_name = ext_info['name']
    build_dir = os.path.join(ext_dir, 'build')
    output_path = os.path.join(build_dir, ext_info['output'])

    if not os.path.isdir(ext_dir):
        print(f"[openvino_ext] Source directory not found: {ext_dir}, skipping {ext_name}")
        return False

    print(f"\n══════════════════════════════════════════")
    print(f"Building OpenVINO {ext_name} extension")
    print(f"══════════════════════════════════════════")

    os.makedirs(build_dir, exist_ok=True)

    # cmake configure
    cmake_cmd = ['cmake', '..']
    print(f"[openvino_ext] Running: {' '.join(cmake_cmd)}  (in {build_dir})")
    result = subprocess.run(cmake_cmd, cwd=build_dir,
                            capture_output=True, text=True)
    if result.returncode != 0:
        print(f"[openvino_ext] cmake configure failed:\n{result.stderr}")
        return False

    # cmake build
    nproc = os.cpu_count() or 1
    make_cmd = ['make', f'-j{nproc}']
    print(f"[openvino_ext] Running: {' '.join(make_cmd)}  (in {build_dir})")
    result = subprocess.run(make_cmd, cwd=build_dir,
                            capture_output=True, text=True)
    if result.returncode != 0:
        print(f"[openvino_ext] make failed:\n{result.stderr}")
        return False

    if os.path.isfile(output_path):
        print(f"[openvino_ext] ✓ Built: {output_path}")
        return True
    else:
        print(f"[openvino_ext] ✗ Build produced no output .so for {ext_name}")
        return False


def build_all_openvino_extensions():
    """Build all OpenVINO custom extensions."""
    # Check that OpenVINO is importable first
    try:
        import openvino  # noqa: F401
    except ImportError:
        print("[openvino_ext] OpenVINO Python package not found, skipping all extensions")
        return

    for ext_info in OV_EXTENSIONS:
        build_openvino_extension(ext_info)


class BuildExtWithOpenVINO(BuildExtension):
    """Extended build_ext that also builds the OpenVINO extensions via cmake."""

    def run(self):
        # Build all OpenVINO extensions first (cmake-based)
        build_all_openvino_extensions()
        # Then build all the torch/OpenCL extensions as usual
        super().run()


if __name__ == '__main__':
    # The bevfusion/kernels OpenCL extensions (bev_pool_bucket_gpu, voxel_layer,
    # sparse_conv_*, geometry_transform_gpu, etc.) were used by the old
    # bevfusion/pipeline.py path and are no longer needed.  The active inference
    # pipeline (run_inference_standalone.py) uses the OpenVINO custom extensions
    # (BEVPool, Voxelize, SparseEncoder) which are built via cmake inside
    # BuildExtWithOpenVINO.run() → build_all_openvino_extensions().
    setup(
        name='bevfusion_opencl',
        ext_modules=[],          # no torch/OpenCL extensions needed
        cmdclass={'build_ext': BuildExtWithOpenVINO},
        zip_safe=False,
    )
