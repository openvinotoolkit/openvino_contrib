import os

import torch
from setuptools import find_packages, setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension


def make_cuda_ext(name, module, sources, sources_cuda):
    return CUDAExtension(
        name=f"{module}.{name}",
        sources=[os.path.join(*module.split('.'), p) for p in (sources + sources_cuda)],
        define_macros=[('WITH_CUDA', None)],
        extra_compile_args={
            'cxx': ['-O3'],
            'nvcc': [
                '-O3',
                '-D__CUDA_NO_HALF_OPERATORS__',
                '-D__CUDA_NO_HALF_CONVERSIONS__',
                '-D__CUDA_NO_HALF2_OPERATORS__',
            ],
        },
    )


def build_ext_modules():
    has_cuda = torch.cuda.is_available() or os.getenv('FORCE_CUDA', '0') == '1'
    if not has_cuda:
        print('CUDA not available: building Python package without CUDA extensions.')
        print('Runtime will use CPU/OpenVINO fallback paths where implemented.')
        return []

    print('CUDA detected: building CUDA extensions.')
    return [
        make_cuda_ext(
            name='bev_pool_ext',
            module='mmdet3d_plugin.ops.bev_pool',
            sources=[
                'src/bev_pooling.cpp',
                'src/bev_sum_pool.cpp',
                'src/bev_max_pool.cpp',
            ],
            sources_cuda=[
                'src/bev_sum_pool_cuda.cu',
                'src/bev_max_pool_cuda.cu',
            ],
        ),
        make_cuda_ext(
            name='bev_pool_v2_ext',
            module='mmdet3d_plugin.ops.bev_pool_v2',
            sources=['src/bev_pool.cpp'],
            sources_cuda=['src/bev_pool_cuda.cu'],
        ),
        make_cuda_ext(
            name='nearest_assign_ext',
            module='mmdet3d_plugin.ops.nearest_assign',
            sources=['src/nearest_assign.cpp'],
            sources_cuda=['src/nearest_assign_cuda.cu'],
        ),
    ]


if __name__ == '__main__':
    setup(
        name='flashocc_plugin',
        description='FlashOCC plugin package with optional CUDA extensions',
        packages=find_packages(),
        ext_modules=build_ext_modules(),
        cmdclass={'build_ext': BuildExtension.with_options(use_ninja=False)},
        zip_safe=False,
    )
